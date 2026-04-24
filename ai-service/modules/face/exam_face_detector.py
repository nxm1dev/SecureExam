"""
ai-service/modules/face/exam_face_detector.py
─────────────────────────────────────────────
ExamFaceDetector – giám sát khuôn mặt theo thời gian thực trong phòng thi.

Thiết kế:
  - Sử dụng InsightFace FaceAnalysis (buffalo_l) chỉ với task 'detection'.
  - Throttle inference: thực sự chạy model mỗi ~0.25s (4 FPS) dù caller
    gọi process_frame() liên tục ở 30 FPS.
  - Các frame ở giữa trả về kết quả cache của lần inference gần nhất.
  - Callback-based: không print ra màn hình, toàn bộ logic phản ứng nằm
    ở phía ứng dụng gọi.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases cho callback signatures
# ──────────────────────────────────────────────────────────────────────────────
# BBox: [x1, y1, x2, y2] dạng int (pixel coordinates)
BBox = List[int]

OnMultipleFacesCB = Callable[[np.ndarray, int], None]
OnNoFaceCB = Callable[[np.ndarray], None]
OnNormalCB = Callable[[np.ndarray, BBox], None]


# ──────────────────────────────────────────────────────────────────────────────
# Internal result dataclass
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    """Kết quả detect của một lần inference."""
    face_count: int = 0
    bboxes: List[BBox] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────
class ExamFaceDetector:
    """
    Giám sát khuôn mặt trong khung hình camera kỳ thi.

    Parameters
    ----------
    model_root : str
        Thư mục root chứa (hoặc sẽ tải về) model buffalo_l.
        Cấu trúc mong đợi: ``<model_root>/models/buffalo_l/``.
    on_multiple_faces : OnMultipleFacesCB | None
        Callback khi phát hiện > 1 khuôn mặt (nghi ngờ có người nhắc bài).
        Signature: ``(frame: np.ndarray, face_count: int) -> None``
    on_no_face : OnNoFaceCB | None
        Callback khi không phát hiện khuôn mặt nào (nghi ngờ rời vị trí).
        Signature: ``(frame: np.ndarray) -> None``
    on_normal : OnNormalCB | None
        Callback khi chỉ có đúng 1 khuôn mặt (trạng thái bình thường).
        Signature: ``(frame: np.ndarray, bbox: BBox) -> None``
        với ``bbox = [x1, y1, x2, y2]``.
    det_size : tuple[int, int]
        Kích thước ảnh đầu vào cho model detection. Mặc định (640, 640).
    det_thresh : float
        Ngưỡng confidence tối thiểu để chấp nhận một khuôn mặt.
    throttle_interval : float
        Thời gian tối thiểu (giây) giữa hai lần inference thực sự.
        Mặc định 0.25s → ~4 FPS inference.
    resize_width : int
        Chiều rộng resize frame trước khi đưa vào model.
        Chiều cao được tính tự động giữ tỉ lệ khung hình. Mặc định 640.
    ctx_id : int
        0 = GPU (CUDA), -1 = CPU.
    """

    # ── Preprocessing target resolution ───────────────────────────────────────
    _PREPROCESS_W: int = 640
    _PREPROCESS_H: int = 360  # 16:9 ratio

    def __init__(
        self,
        model_root: str = "~/.insightface",
        on_multiple_faces: Optional[OnMultipleFacesCB] = None,
        on_no_face: Optional[OnNoFaceCB] = None,
        on_normal: Optional[OnNormalCB] = None,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        throttle_interval: float = 0.25,
        ctx_id: int = -1,  # -1 = CPU; 0 = GPU
    ) -> None:
        self._on_multiple_faces = on_multiple_faces
        self._on_no_face = on_no_face
        self._on_normal = on_normal

        self._det_size = det_size
        self._det_thresh = det_thresh
        self._throttle_interval = throttle_interval
        self._ctx_id = ctx_id
        self._model_root = model_root

        # Throttle state
        self._last_inference_time: float = 0.0
        self._last_result: DetectionResult = DetectionResult()

        # Lazy-load flag
        self._app = None  # insightface.app.FaceAnalysis

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """Lazy-load InsightFace detection model (thread-safe nếu gọi 1 lần)."""
        if self._app is not None:
            return

        import insightface
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name="buffalo_l",
            root=self._model_root,
            allowed_modules=["detection"],   # Chỉ load detection, bỏ recognition/landmark
        )
        self._app.prepare(
            ctx_id=self._ctx_id,
            det_thresh=self._det_thresh,
            det_size=self._det_size,
        )

    # ── Preprocessing ──────────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """
        Resize frame xuống 640×360 (hoặc giữ tỉ lệ nếu frame không phải 16:9).
        Dùng cv2.INTER_AREA vì đây là downscale – chất lượng tốt nhất.
        """
        h, w = frame.shape[:2]
        target_w = ExamFaceDetector._PREPROCESS_W
        target_h = ExamFaceDetector._PREPROCESS_H

        if w == target_w and h == target_h:
            return frame  # Không cần resize

        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # ── Inference ─────────────────────────────────────────────────────────────
    def _run_inference(self, small_frame: np.ndarray) -> DetectionResult:
        """Chạy InsightFace trên frame đã preprocess và trả về DetectionResult."""
        self._load_model()

        try:
            faces = self._app.get(small_frame)
        except Exception:
            return DetectionResult()

        bboxes: List[BBox] = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            bboxes.append(bbox)

        return DetectionResult(
            face_count=len(bboxes),
            bboxes=bboxes,
            timestamp=time.monotonic(),
        )

    # ── Callback dispatcher ────────────────────────────────────────────────────
    def _dispatch(self, frame: np.ndarray, result: DetectionResult) -> None:
        """Gọi callback phù hợp dựa theo face_count."""
        count = result.face_count

        if count == 0:
            if self._on_no_face is not None:
                self._on_no_face(frame)

        elif count == 1:
            bbox: BBox = result.bboxes[0]
            if self._on_normal is not None:
                self._on_normal(frame, bbox)

        else:  # count > 1
            if self._on_multiple_faces is not None:
                self._on_multiple_faces(frame, count)

    # ── Public API ─────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Xử lý một frame từ OpenCV (BGR numpy array).

        Nếu chưa đến ``throttle_interval`` kể từ lần inference trước, hàm
        trả về kết quả cache và vẫn gọi callback với ``frame`` hiện tại.

        Parameters
        ----------
        frame : np.ndarray
            Frame BGR từ ``cv2.VideoCapture.read()``.

        Returns
        -------
        DetectionResult
            Kết quả detect (có thể là cache từ lần trước).
        """
        now = time.monotonic()
        elapsed = now - self._last_inference_time

        if elapsed >= self._throttle_interval:
            # Thực sự chạy inference
            small = self._preprocess(frame)
            self._last_result = self._run_inference(small)
            self._last_inference_time = now

        # Luôn dispatch với frame gốc (full resolution) để callback có ảnh đẹp
        self._dispatch(frame, self._last_result)
        return self._last_result

    def reset(self) -> None:
        """Xóa state throttle (dùng khi bắt đầu phiên thi mới)."""
        self._last_inference_time = 0.0
        self._last_result = DetectionResult()

    @property
    def last_result(self) -> DetectionResult:
        """Kết quả inference gần nhất (có thể là giá trị khởi tạo nếu chưa detect)."""
        return self._last_result
