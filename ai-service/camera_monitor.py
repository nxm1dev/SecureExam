"""
ai-service/camera_monitor.py
─────────────────────────────
Standalone camera monitor sử dụng ExamFaceDetector.

Chạy trực tiếp để test giám sát khuôn mặt từ webcam:
    python camera_monitor.py [--camera 0] [--gpu]

Tích hợp vào ai-service: module này có thể được import và khởi động
song song với FastAPI server.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Thêm thư mục gốc ai-service vào sys.path để import đúng module
_SERVICE_DIR = Path(__file__).resolve().parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

from modules.face.exam_face_detector import BBox, DetectionResult, ExamFaceDetector

# ──────────────────────────────────────────────────────────────────────────────
# Callback implementations (hiển thị trực tiếp, có thể thay bằng HTTP call)
# ──────────────────────────────────────────────────────────────────────────────

# Màu sắc overlay
_COLOR_NORMAL = (0, 220, 80)       # Xanh lá  – bình thường
_COLOR_NO_FACE = (60, 60, 220)     # Đỏ       – không có mặt
_COLOR_MULTIPLE = (0, 165, 255)    # Cam      – nhiều mặt

# Kích thước detector
_DET_SIZE = (640, 640)

# Shared display state (dùng lock vì callback gọi từ thread capture)
_overlay_lock = threading.Lock()
_overlay_state: dict = {
    "label": "Initializing…",
    "color": (200, 200, 200),
    "bbox": None,
    "face_count": 0,
    "last_update": 0.0,
}


def _on_multiple_faces(frame: np.ndarray, face_count: int) -> None:
    """Callback: phát hiện nhiều hơn 1 khuôn mặt."""
    with _overlay_lock:
        _overlay_state["label"] = f"⚠ CẢNH BÁO: {face_count} khuôn mặt!"
        _overlay_state["color"] = _COLOR_MULTIPLE
        _overlay_state["face_count"] = face_count
        _overlay_state["bbox"] = None
        _overlay_state["last_update"] = time.monotonic()
    # Ở đây có thể POST violation lên backend:
    # requests.post("http://localhost:8000/api/violations", json={"type": "multiple_faces", ...})


def _on_no_face(frame: np.ndarray) -> None:
    """Callback: không phát hiện khuôn mặt nào."""
    with _overlay_lock:
        _overlay_state["label"] = "⚠ CẢNH BÁO: Không có thí sinh!"
        _overlay_state["color"] = _COLOR_NO_FACE
        _overlay_state["face_count"] = 0
        _overlay_state["bbox"] = None
        _overlay_state["last_update"] = time.monotonic()


def _on_normal(frame: np.ndarray, bbox: BBox) -> None:
    """Callback: đúng 1 khuôn mặt – trạng thái bình thường."""
    with _overlay_lock:
        _overlay_state["label"] = "✓ Bình thường"
        _overlay_state["color"] = _COLOR_NORMAL
        _overlay_state["face_count"] = 1
        _overlay_state["bbox"] = bbox
        _overlay_state["last_update"] = time.monotonic()


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _normalize_bbox_to_frame(
    bbox: BBox,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """
    Map bbox từ hệ tọa độ detector về hệ tọa độ frame gốc.

    Sửa lỗi lệch lên trên đầu thí sinh do detector thường resize ảnh về
    khung vuông 640x640 theo kiểu letterbox (giữ tỉ lệ + padding).

    Với camera 16:9 như 1280x720:
    - ảnh được scale theo min(640/1280, 640/720) = 0.5
    - ảnh sau scale là 640x360
    - padding top/bottom = 140px mỗi bên
    => bbox phải được trừ padding rồi chia lại scale.
    """
    x1, y1, x2, y2 = map(int, bbox)
    det_w, det_h = _DET_SIZE

    # Nếu bbox trông như đã ở hệ tọa độ frame gốc thì chỉ clamp lại.
    # Điều này giúp an toàn nếu detector trong tương lai trả bbox theo frame.
    if max(x1, x2) > det_w * 1.5 or max(y1, y2) > det_h * 1.5:
        x1 = _clamp(x1, 0, max(frame_w - 1, 0))
        x2 = _clamp(x2, 0, max(frame_w - 1, 0))
        y1 = _clamp(y1, 0, max(frame_h - 1, 0))
        y2 = _clamp(y2, 0, max(frame_h - 1, 0))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    # Inverse letterbox: frame gốc -> det_size
    scale = min(det_w / float(frame_w), det_h / float(frame_h))
    new_w = frame_w * scale
    new_h = frame_h * scale
    pad_x = (det_w - new_w) / 2.0
    pad_y = (det_h - new_h) / 2.0

    # Bỏ padding rồi đưa về frame gốc
    x1 = int(round((x1 - pad_x) / scale))
    x2 = int(round((x2 - pad_x) / scale))
    y1 = int(round((y1 - pad_y) / scale))
    y2 = int(round((y2 - pad_y) / scale))

    # Clamp
    x1 = _clamp(x1, 0, max(frame_w - 1, 0))
    x2 = _clamp(x2, 0, max(frame_w - 1, 0))
    y1 = _clamp(y1, 0, max(frame_h - 1, 0))
    y2 = _clamp(y2, 0, max(frame_h - 1, 0))

    # Đảm bảo đúng thứ tự
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def _draw_overlay(frame: np.ndarray) -> np.ndarray:
    """Vẽ status label và bbox lên frame (non-destructive copy)."""
    display = frame.copy()
    h, w = display.shape[:2]

    with _overlay_lock:
        label: str = _overlay_state["label"]
        color: tuple = _overlay_state["color"]
        bbox: Optional[BBox] = _overlay_state["bbox"]

    # Semi-transparent top banner
    banner_h = 50
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    # Status text
    cv2.putText(
        display, label,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        color, 2, cv2.LINE_AA,
    )

    # Bounding box (khi có 1 mặt)
    if bbox is not None:
        x1, y1, x2, y2 = _normalize_bbox_to_frame(bbox, w, h)

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            display, "Thi sinh",
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            color, 1, cv2.LINE_AA,
        )

    return display


# ──────────────────────────────────────────────────────────────────────────────
# Main monitor loop
# ──────────────────────────────────────────────────────────────────────────────

def run_monitor(
    camera_index: int = 0,
    model_root: str = "~/.insightface",
    ctx_id: int = -1,
    throttle_interval: float = 0.25,
    show_window: bool = True,
) -> None:
    """
    Khởi chạy vòng lặp giám sát camera.

    Parameters
    ----------
    camera_index : int
        Index camera OpenCV (0 = webcam mặc định).
    model_root : str
        Thư mục chứa model InsightFace.
    ctx_id : int
        -1 = CPU, 0 = GPU.
    throttle_interval : float
        Khoảng cách tối thiểu (giây) giữa hai lần inference.
    show_window : bool
        Hiện cửa sổ preview OpenCV (tắt nếu chạy headless).
    """
    print("[ExamMonitor] Khởi tạo ExamFaceDetector…")

    detector = ExamFaceDetector(
        model_root=model_root,
        on_multiple_faces=_on_multiple_faces,
        on_no_face=_on_no_face,
        on_normal=_on_normal,
        det_size=_DET_SIZE,
        det_thresh=0.5,
        throttle_interval=throttle_interval,
        ctx_id=ctx_id,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở camera index={camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[ExamMonitor] Camera sẵn sàng. Nhấn 'q' để thoát.")

    frame_count = 0
    fps_timer = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ExamMonitor] Mất kết nối camera, thử lại…")
                time.sleep(0.1)
                continue

            # ── Gọi ExamFaceDetector ───────────────────────────────────────
            result: DetectionResult = detector.process_frame(frame)

            # ── Tính FPS hiển thị ─────────────────────────────────────────
            frame_count += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                print(
                    f"[ExamMonitor] FPS: {fps:.1f} | "
                    f"Faces: {result.face_count} | "
                    f"Status: {_overlay_state['label']}"
                )
                frame_count = 0
                fps_timer = time.monotonic()

            # ── Hiển thị ──────────────────────────────────────────────────
            if show_window:
                display = _draw_overlay(frame)
                cv2.imshow("SecureExam – Giam sat thi cu", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' hoặc ESC
                    break

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        print("[ExamMonitor] Đã dừng.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SecureExam – Giám sát khuôn mặt kỳ thi")
    parser.add_argument("--camera", type=int, default=0, help="Index camera (mặc định: 0)")
    parser.add_argument("--gpu", action="store_true", help="Dùng GPU (CUDA), mặc định CPU")
    parser.add_argument(
        "--model-root",
        default=str(Path(__file__).resolve().parent / "models_cache"),
        help="Thư mục model InsightFace",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.25,
        help="Khoảng cách giữa 2 lần inference (giây, mặc định 0.25)",
    )
    parser.add_argument("--headless", action="store_true", help="Chạy không hiển thị cửa sổ")
    args = parser.parse_args()

    run_monitor(
        camera_index=args.camera,
        model_root=args.model_root,
        ctx_id=0 if args.gpu else -1,
        throttle_interval=args.throttle,
        show_window=not args.headless,
    )