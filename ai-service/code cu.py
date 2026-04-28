from __future__ import annotations

"""
ai-service/code cu.py
─────────────────────────────
Multimodal AI Exam Anti-Cheating System.

Gộp 3 phần:
  1. Frontend React/TS (VAD + Camera) – code mẫu trong docstring
  2. Backend Python – ExamFaceDetector nâng cấp (MAR + Pose)
  3. Cross-check Controller – Logic xác nhận chéo
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHẦN 1: FRONTEND (React / TypeScript) – Code mẫu trong docstring       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

FRONTEND_CODE = """
// ─── useExamVAD.ts ── Custom Hook tích hợp @ricky0123/vad-web ───────────────
// npm install @ricky0123/vad-react

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useMicVAD } from '@ricky0123/vad-react';

interface AnalyzeResult {
  status: string;
  message: string;
  level: number;
}

const API_URL = 'http://localhost:8001/api/multimodal/analyze';

export const useExamVAD = (videoRef: React.RefObject<HTMLVideoElement>) => {
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [lastResult, setLastResult] = useState<AnalyzeResult | null>(null);

  // Capture frame từ video và gửi lên backend
  const captureAndSend = useCallback(async (speechDetected: boolean) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const formData = new FormData();
      formData.append('frame', blob, 'frame.jpg');
      formData.append('speech_detected', String(speechDetected));

      try {
        const res = await fetch(API_URL, { method: 'POST', body: formData });
        const data: AnalyzeResult = await res.json();
        setLastResult(data);
      } catch (err) {
        console.error('[ExamVAD] API error:', err);
      }
    }, 'image/jpeg', 0.8);
  }, [videoRef]);

  // Cấu hình VAD
  const vad = useMicVAD({
    startOnLoad: true,
    onSpeechStart: () => {
      setIsSpeaking(true);
      captureAndSend(true);  // Capture ngay khi phát hiện giọng nói
    },
    onSpeechEnd: () => {
      setIsSpeaking(false);
      captureAndSend(false);
    },
    onVADMisfire: () => {
      setIsSpeaking(false);
    },
  });

  return { isSpeaking, lastResult, vad };
};

// ─── ExamMonitorPanel.tsx ── Component hiển thị trạng thái ──────────────────

// import { useExamVAD } from './useExamVAD';
//
// export const ExamMonitorPanel: React.FC = () => {
//   const videoRef = useRef<HTMLVideoElement>(null);
//   const { isSpeaking, lastResult } = useExamVAD(videoRef);
//
//   useEffect(() => {
//     navigator.mediaDevices.getUserMedia({ video: true, audio: true })
//       .then(stream => { if (videoRef.current) videoRef.current.srcObject = stream; });
//   }, []);
//
//   const levelColor = (level?: number) => {
//     if (!level || level === 0) return '#22c55e';
//     if (level === 1) return '#eab308';
//     if (level === 2) return '#f97316';
//     return '#ef4444';
//   };
//
//   return (
//     <div>
//       <video ref={videoRef} autoPlay playsInline muted />
//       <div style={{ background: isSpeaking ? '#ef4444' : '#22c55e', color: '#fff', padding: 8 }}>
//         {isSpeaking ? 'Đang phát hiện giọng nói...' : 'Yên lặng'}
//       </div>
//       {lastResult && (
//         <div style={{ background: levelColor(lastResult.level), color: '#fff', padding: 8 }}>
//           [{lastResult.status}] {lastResult.message}
//         </div>
//       )}
//     </div>
//   );
// };
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHẦN 2: BACKEND – ExamFaceDetector nâng cấp (MAR + Head Pose)          ║
# ╚════════════════════════════════════════════════════════════════════════════╝


import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Type aliases
BBox = List[int]
OnMultipleFacesCB = Callable[[np.ndarray, int], None]
OnNoFaceCB = Callable[[np.ndarray], None]
OnNormalCB = Callable[[np.ndarray, BBox], None]


@dataclass
class DetectionResult:
    """Kết quả detect của một lần inference."""
    face_count: int = 0
    bboxes: List[BBox] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class MultimodalResult:
    """Kết quả phân tích đa phương thức (hình ảnh + âm thanh)."""
    face_count: int = 0
    bboxes: List[BBox] = field(default_factory=list)
    # MAR (Mouth Aspect Ratio)
    is_mouth_open: bool = False
    mar_value: float = 0.0
    # Head Pose
    is_looking_away: bool = False
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    # Landmarks raw (cho debug)
    has_landmarks: bool = False


class ExamFaceDetector:
    """
    Giám sát khuôn mặt + phân tích MAR & Head Pose cho hệ thống chống gian lận.

    Nâng cấp so với bản gốc:
      - Bật module 'landmark_2d_106' để có 106 điểm landmark
      - Hàm calculate_mar() tính Mouth Aspect Ratio
      - Hàm analyze_pose() phân tích góc quay đầu
    """

    _PREPROCESS_W: int = 640
    _PREPROCESS_H: int = 360

    def __init__(
        self,
        model_root: str = "~/.insightface",
        on_multiple_faces: Optional[OnMultipleFacesCB] = None,
        on_no_face: Optional[OnNoFaceCB] = None,
        on_normal: Optional[OnNormalCB] = None,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        throttle_interval: float = 0.25,
        ctx_id: int = -1,
        # ── Ngưỡng mới cho Multimodal ──
        mar_threshold: float = 0.20,
        yaw_threshold: float = 30.0,
        pitch_threshold: float = 20.0,
    ) -> None:
        self._on_multiple_faces = on_multiple_faces
        self._on_no_face = on_no_face
        self._on_normal = on_normal

        self._det_size = det_size
        self._det_thresh = det_thresh
        self._throttle_interval = throttle_interval
        self._ctx_id = ctx_id
        self._model_root = model_root

        # ── Ngưỡng cấu hình (có thể chỉnh runtime) ──
        # MAR > 0.20 → miệng mở. Tăng nếu false-positive nhiều.
        self.mar_threshold: float = mar_threshold
        # |yaw| > 30° → quay ngang. Giảm nếu muốn nhạy hơn.
        self.yaw_threshold: float = yaw_threshold
        # |pitch| > 20° → cúi/ngửa quá mức.
        self.pitch_threshold: float = pitch_threshold

        # Throttle state
        self._last_inference_time: float = 0.0
        self._last_result: DetectionResult = DetectionResult()
        self._last_multimodal: MultimodalResult = MultimodalResult()

        # Lazy-load
        self._app = None

    # ── Model loading ──────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """Lazy-load InsightFace với detection + landmark_2d_106."""
        if self._app is not None:
            return

        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name="buffalo_l",
            root=self._model_root,
            # QUAN TRỌNG: Thêm 'landmark_2d_106' để có 106 điểm landmark
            allowed_modules=["detection", "landmark_2d_106"],
        )
        self._app.prepare(
            ctx_id=self._ctx_id,
            det_thresh=self._det_thresh,
            det_size=self._det_size,
        )

    # ── Preprocessing ──────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        tw, th = ExamFaceDetector._PREPROCESS_W, ExamFaceDetector._PREPROCESS_H
        if w == tw and h == th:
            return frame
        return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

    # ── MAR Calculation ────────────────────────────────────────────────────
    def calculate_mar(self, landmarks: Optional[np.ndarray]) -> Tuple[bool, float]:
        """
        Tính Mouth Aspect Ratio (MAR) từ InsightFace 106-point landmarks.

        Công thức:
            MAR = khoảng_cách_dọc_môi / khoảng_cách_ngang_khóe_miệng

        InsightFace 106-point landmark indices cho vùng miệng:
            - Khóe miệng trái:  điểm 84
            - Khóe miệng phải:  điểm 90
            - Môi trên (giữa):  điểm 87
            - Môi dưới (giữa):  điểm 93

        Returns:
            (is_mouth_open, mar_value)
        """
        if landmarks is None or len(landmarks) < 106:
            return False, 0.0

        # Khóe miệng trái và phải
        left_corner = landmarks[84]   # Khóe trái
        right_corner = landmarks[90]  # Khóe phải

        # Khoảng cách ngang (horizontal) giữa 2 khóe miệng
        horizontal = np.linalg.norm(left_corner - right_corner)
        if horizontal < 1e-6:
            return False, 0.0

        # Điểm giữa môi trên và môi dưới
        top_lip_center = landmarks[87]     # Giữa môi trên
        bottom_lip_center = landmarks[93]  # Giữa môi dưới

        # Khoảng cách dọc (vertical) giữa môi trên và môi dưới
        vertical = np.linalg.norm(top_lip_center - bottom_lip_center)

        mar = float(vertical / horizontal)
        is_mouth_open = mar > self.mar_threshold

        return is_mouth_open, mar

    # ── Head Pose Analysis ─────────────────────────────────────────────────
    def analyze_pose(self, pose: Optional[np.ndarray]) -> Tuple[bool, Tuple[float, float, float]]:
        """
        Phân tích góc quay đầu từ InsightFace.

        InsightFace trả về pose = [pitch, yaw, roll] (đơn vị: độ).
          - pitch: cúi (âm) / ngửa (dương)
          - yaw:   quay trái (âm) / quay phải (dương)
          - roll:  nghiêng đầu

        Returns:
            (is_looking_away, (pitch, yaw, roll))
        """
        if pose is None or len(pose) < 3:
            return False, (0.0, 0.0, 0.0)

        pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])

        # Thí sinh "nhìn đi chỗ khác" khi yaw hoặc pitch vượt ngưỡng
        is_looking_away = abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold

        return is_looking_away, (pitch, yaw, roll)

    # ── Inference ──────────────────────────────────────────────────────────
    def _run_inference(self, small_frame: np.ndarray) -> Tuple[DetectionResult, MultimodalResult]:
        """Chạy InsightFace và trích xuất BBox + MAR + Pose."""
        self._load_model()

        try:
            faces = self._app.get(small_frame)
        except Exception:
            return DetectionResult(), MultimodalResult()

        bboxes: List[BBox] = []
        for face in faces:
            bboxes.append(face.bbox.astype(int).tolist())

        det = DetectionResult(face_count=len(bboxes), bboxes=bboxes, timestamp=time.monotonic())

        # Phân tích MAR và Pose cho khuôn mặt đầu tiên (thí sinh chính)
        mm = MultimodalResult(face_count=len(bboxes), bboxes=bboxes)

        if faces:
            face = faces[0]

            # MAR từ landmark_2d_106
            lmk = getattr(face, "landmark_2d_106", None)
            if lmk is not None:
                mm.has_landmarks = True
                mm.is_mouth_open, mm.mar_value = self.calculate_mar(lmk)

            # Head Pose
            pose = getattr(face, "pose", None)
            if pose is not None:
                mm.is_looking_away, (mm.pitch, mm.yaw, mm.roll) = self.analyze_pose(pose)

        return det, mm

    # ── Callback dispatcher ────────────────────────────────────────────────
    def _dispatch(self, frame: np.ndarray, result: DetectionResult) -> None:
        count = result.face_count
        if count == 0:
            if self._on_no_face:
                self._on_no_face(frame)
        elif count == 1:
            if self._on_normal:
                self._on_normal(frame, result.bboxes[0])
        else:
            if self._on_multiple_faces:
                self._on_multiple_faces(frame, count)

    # ── Public API ─────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Xử lý frame (có throttle). Tương thích API cũ."""
        now = time.monotonic()
        if now - self._last_inference_time >= self._throttle_interval:
            small = self._preprocess(frame)
            self._last_result, self._last_multimodal = self._run_inference(small)
            self._last_inference_time = now

        self._dispatch(frame, self._last_result)
        return self._last_result

    def process_frame_multimodal(self, frame: np.ndarray) -> MultimodalResult:
        """Xử lý frame và trả về kết quả đa phương thức (MAR + Pose)."""
        now = time.monotonic()
        if now - self._last_inference_time >= self._throttle_interval:
            small = self._preprocess(frame)
            self._last_result, self._last_multimodal = self._run_inference(small)
            self._last_inference_time = now

        self._dispatch(frame, self._last_result)
        return self._last_multimodal

    def reset(self) -> None:
        self._last_inference_time = 0.0
        self._last_result = DetectionResult()
        self._last_multimodal = MultimodalResult()

    @property
    def last_result(self) -> DetectionResult:
        return self._last_result

    @property
    def last_multimodal(self) -> MultimodalResult:
        return self._last_multimodal


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHẦN 3: CROSS-CHECK CONTROLLER – Logic xác nhận chéo                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class CheatingVerdict:
    """Kết quả phân tích hành vi gian lận."""
    status: str       # Mã trạng thái (NORMAL, MILD_WARNING, WARNING_L1, WARNING_L2)
    message: str      # Mô tả tiếng Việt
    level: int         # 0=bình thường, 1=nhẹ, 2=mức 1, 3=mức 2 khẩn cấp
    details: Dict[str, Any] = field(default_factory=dict)


def analyze_cheating_behavior(
    frame: np.ndarray,
    speech_detected: bool,
    detector: ExamFaceDetector,
) -> CheatingVerdict:
    """
    Hàm xác nhận chéo (Cross-check) giữa VAD (âm thanh) và Camera (hình ảnh).

    Ma trận logic:
    ┌──────────────────┬────────────────┬──────────────────┬─────────────────────────────┐
    │ speech_detected  │ is_mouth_open  │ is_looking_away  │ Kết luận                    │
    ├──────────────────┼────────────────┼──────────────────┼─────────────────────────────┤
    │ False            │ *              │ *                │ Bình thường (bỏ qua)        │
    │ True             │ True           │ False            │ Cảnh báo nhẹ (đọc nhẩm)    │
    │ True             │ True           │ True             │ Mức 1 (quay sang nói)       │
    │ True             │ False          │ *                │ Mức 2 (người khác nhắc bài) │
    └──────────────────┴────────────────┴──────────────────┴─────────────────────────────┘

    Args:
        frame: Frame BGR từ camera/upload.
        speech_detected: Cờ VAD từ client (True = có tiếng nói).
        detector: Instance ExamFaceDetector đã khởi tạo.

    Returns:
        CheatingVerdict chứa status, message, level và details.
    """
    # ── Không có tiếng nói → Bình thường ──
    if not speech_detected:
        return CheatingVerdict(
            status="NORMAL",
            message="Bình thường – không phát hiện tiếng nói.",
            level=0,
        )

    # ── Có tiếng nói → Phân tích hình ảnh ──
    mm: MultimodalResult = detector.process_frame_multimodal(frame)

    details: Dict[str, Any] = {
        "face_count": mm.face_count,
        "mar": mm.mar_value,
        "is_mouth_open": mm.is_mouth_open,
        "is_looking_away": mm.is_looking_away,
        "pose": {"pitch": mm.pitch, "yaw": mm.yaw, "roll": mm.roll},
        "has_landmarks": mm.has_landmarks,
    }

    # Không thấy mặt nhưng có tiếng nói → cũng đáng ngờ
    if mm.face_count == 0:
        return CheatingVerdict(
            status="WARNING_NO_FACE",
            message="Có tiếng nói nhưng không tìm thấy khuôn mặt trong khung hình.",
            level=2,
            details=details,
        )

    # ── Áp dụng ma trận logic ──

    if mm.is_mouth_open and not mm.is_looking_away:
        # speech=True, mouth=open, looking=straight → đọc nhẩm
        return CheatingVerdict(
            status="MILD_WARNING",
            message="Cảnh báo nhẹ: Thí sinh đang đọc nhẩm.",
            level=1,
            details=details,
        )

    if mm.is_mouth_open and mm.is_looking_away:
        # speech=True, mouth=open, looking=away → quay sang nói chuyện
        return CheatingVerdict(
            status="WARNING_LEVEL_1",
            message="Cảnh báo Mức 1: Thí sinh đang quay sang nói chuyện.",
            level=2,
            details=details,
        )

    if not mm.is_mouth_open:
        # speech=True, mouth=closed → có người khác nhắc bài
        return CheatingVerdict(
            status="WARNING_LEVEL_2_URGENT",
            message="Cảnh báo Mức 2 Khẩn cấp: Có tiếng người khác nhắc bài!",
            level=3,
            details=details,
        )

    # Fallback (không bao giờ đến đây theo logic trên)
    return CheatingVerdict(status="UNKNOWN", message="Không xác định.", level=0, details=details)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CAMERA MONITOR GUI – Giao dien giam sat thoi gian thuc                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

import argparse
import os
import sys
import threading
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Mau sac overlay ───────────────────────────────────────────────────────────
_COLOR_NORMAL = (0, 220, 80)       # Xanh la  - binh thuong
_COLOR_NO_FACE = (60, 60, 220)     # Do       - khong co mat
_COLOR_MULTIPLE = (0, 165, 255)    # Cam      - nhieu mat
_COLOR_WARNING_L1 = (0, 255, 255)  # Vang     - canh bao nhe
_COLOR_WARNING_L2 = (0, 140, 255)  # Cam dam  - muc 1
_COLOR_WARNING_L3 = (0, 0, 255)    # Do       - muc 2 khan cap

_DET_SIZE = (640, 640)
_PREPROCESS_SIZE = (640, 360)

# ── Shared overlay state (thread-safe) ─────────────────────────────────────
_overlay_lock = threading.Lock()
_overlay_state: dict = {
    "label": "Initializing...",
    "color": (200, 200, 200),
    "bbox": None,
    "face_count": 0,
    "last_update": 0.0,
    "verdict": None,         # CheatingVerdict hien tai
    "multimodal": None,      # MultimodalResult hien tai
    "speech_on": False,
}

# ── Callbacks cho ExamFaceDetector ─────────────────────────────────────────

def _on_multiple_faces(frame: np.ndarray, face_count: int) -> None:
    with _overlay_lock:
        _overlay_state["label"] = f"CANH BAO: {face_count} khuon mat!"
        _overlay_state["color"] = _COLOR_MULTIPLE
        _overlay_state["face_count"] = face_count
        _overlay_state["bbox"] = None
        _overlay_state["last_update"] = time.monotonic()


def _on_no_face(frame: np.ndarray) -> None:
    with _overlay_lock:
        _overlay_state["label"] = "CANH BAO: Khong co thi sinh!"
        _overlay_state["color"] = _COLOR_NO_FACE
        _overlay_state["face_count"] = 0
        _overlay_state["bbox"] = None
        _overlay_state["last_update"] = time.monotonic()


def _on_normal(frame: np.ndarray, bbox: BBox) -> None:
    with _overlay_lock:
        _overlay_state["label"] = "Binh thuong"
        _overlay_state["color"] = _COLOR_NORMAL
        _overlay_state["face_count"] = 1
        _overlay_state["bbox"] = bbox
        _overlay_state["last_update"] = time.monotonic()


# ── Font helpers ───────────────────────────────────────────────────────────

def _load_unicode_font(size: int = 22) -> ImageFont.FreeTypeFont:
    """Tai font ho tro Unicode (tieng Viet)."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for fp in candidates:
        if os.path.isfile(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


# Cache font
_FONT_BANNER = _load_unicode_font(24)
_FONT_LABEL = _load_unicode_font(18)
_FONT_SMALL = _load_unicode_font(14)


def _put_text_unicode(
    img: np.ndarray,
    text: str,
    pos: tuple,
    font: ImageFont.FreeTypeFont,
    color_bgr: tuple,
) -> np.ndarray:
    """Ve text Unicode len anh OpenCV (BGR) bang PIL."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=font, fill=color_rgb)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img, result)
    return img


# ── Bbox normalization ────────────────────────────────────────────────────

def _normalize_bbox_to_frame(
    bbox: BBox, frame_w: int, frame_h: int
) -> tuple:
    """Map bbox tu he toa do detector (640x640) ve frame goc."""
    x1, y1, x2, y2 = map(int, bbox)
    det_w, det_h = _DET_SIZE
    pp_w, pp_h = _PREPROCESS_SIZE

    lb_scale = min(det_w / float(pp_w), det_h / float(pp_h))
    pad_x = (det_w - pp_w * lb_scale) / 2.0
    pad_y = (det_h - pp_h * lb_scale) / 2.0

    pp_x1 = (x1 - pad_x) / lb_scale
    pp_y1 = (y1 - pad_y) / lb_scale
    pp_x2 = (x2 - pad_x) / lb_scale
    pp_y2 = (y2 - pad_y) / lb_scale

    sx = frame_w / float(pp_w)
    sy = frame_h / float(pp_h)

    return (
        max(0, int(pp_x1 * sx)),
        max(0, int(pp_y1 * sy)),
        min(frame_w, int(pp_x2 * sx)),
        min(frame_h, int(pp_y2 * sy)),
    )


# ── Draw overlay ──────────────────────────────────────────────────────────

def _level_color(level: int) -> tuple:
    """Tra ve mau BGR theo muc canh bao."""
    if level <= 0:
        return _COLOR_NORMAL
    elif level == 1:
        return _COLOR_WARNING_L1
    elif level == 2:
        return _COLOR_WARNING_L2
    else:
        return _COLOR_WARNING_L3


def _draw_overlay(frame: np.ndarray) -> np.ndarray:
    """Ve status label, bbox, va panel multimodal len frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    with _overlay_lock:
        label = _overlay_state["label"]
        color = _overlay_state["color"]
        bbox = _overlay_state["bbox"]
        verdict: Optional[CheatingVerdict] = _overlay_state["verdict"]
        mm: Optional[MultimodalResult] = _overlay_state["multimodal"]
        speech_on = _overlay_state["speech_on"]

    # ── Top banner (semi-transparent) ──────────────────────────────────────
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
    _put_text_unicode(display, label, (16, 12), _FONT_BANNER, color)

    # ── Bounding box ──────────────────────────────────────────────────────
    if bbox is not None:
        x1, y1, x2, y2 = _normalize_bbox_to_frame(bbox, w, h)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        _put_text_unicode(display, "Thi sinh", (x1, max(y1 - 24, 2)), _FONT_LABEL, color)

    # ── Panel Multimodal (goc duoi ben trai) ──────────────────────────────
    panel_w = 420
    panel_h = 130
    panel_y = h - panel_h - 10
    panel_x = 10

    # Semi-transparent panel background
    panel_overlay = display.copy()
    cv2.rectangle(panel_overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(panel_overlay, 0.7, display, 0.3, 0, display)

    # Speech status indicator
    speech_color = (0, 0, 255) if speech_on else (0, 220, 80)
    speech_text = "[S] Speech: ON  (nhan 's' de tat)" if speech_on else "[S] Speech: OFF (nhan 's' de bat)"
    _put_text_unicode(display, speech_text, (panel_x + 8, panel_y + 8), _FONT_SMALL, speech_color)

    # MAR & Pose info
    if mm is not None and mm.face_count > 0:
        mar_text = f"MAR: {mm.mar_value:.3f} | Mouth: {'OPEN' if mm.is_mouth_open else 'Closed'}"
        _put_text_unicode(display, mar_text, (panel_x + 8, panel_y + 32), _FONT_SMALL, (200, 200, 200))

        pose_text = f"Yaw: {mm.yaw:.1f}  Pitch: {mm.pitch:.1f}  Roll: {mm.roll:.1f}"
        look_color = (0, 0, 255) if mm.is_looking_away else (0, 220, 80)
        look_text = " | LOOKING AWAY!" if mm.is_looking_away else " | OK"
        _put_text_unicode(display, pose_text + look_text, (panel_x + 8, panel_y + 56), _FONT_SMALL, look_color)
    else:
        _put_text_unicode(display, "MAR: --- | Pose: ---", (panel_x + 8, panel_y + 32), _FONT_SMALL, (120, 120, 120))

    # Verdict
    if verdict is not None and speech_on:
        v_color = _level_color(verdict.level)
        v_text = f"[Lv.{verdict.level}] {verdict.message}"
        _put_text_unicode(display, v_text, (panel_x + 8, panel_y + 85), _FONT_LABEL, v_color)
    else:
        _put_text_unicode(display, "Khong co tieng noi - Binh thuong", (panel_x + 8, panel_y + 85), _FONT_LABEL, _COLOR_NORMAL)

    # Hotkey hint (goc duoi phai)
    hint = "Q: Thoat | S: Toggle Speech"
    _put_text_unicode(display, hint, (w - 280, h - 25), _FONT_SMALL, (150, 150, 150))

    return display


# ── Main monitor loop ─────────────────────────────────────────────────────

def run_monitor(
    camera_index: int = 0,
    model_root: str = "~/.insightface",
    ctx_id: int = -1,
    throttle_interval: float = 0.25,
    simulate_speech: bool = False,
) -> None:
    """
    Khoi chay vong lap giam sat camera voi giao dien multimodal.

    Phim tat:
      - 's': Toggle simulate speech (bat/tat gia lap co tieng noi)
      - 'q' / ESC: Thoat
    """
    print("[ExamMonitor] Khoi tao ExamFaceDetector (Multimodal)...")

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
        raise RuntimeError(f"Khong the mo camera index={camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    speech_on = simulate_speech
    frame_count = 0
    fps_timer = time.monotonic()

    print("[ExamMonitor] Camera san sang. Nhan 'q' de thoat, 's' de toggle speech.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ExamMonitor] Mat ket noi camera, thu lai...")
                time.sleep(0.1)
                continue

            # ── Chay multimodal inference ──────────────────────────────────
            mm: MultimodalResult = detector.process_frame_multimodal(frame)

            # ── Cross-check logic ──────────────────────────────────────────
            verdict: CheatingVerdict = analyze_cheating_behavior(
                frame, speech_on, detector
            )

            # ── Cap nhat overlay state ─────────────────────────────────────
            with _overlay_lock:
                _overlay_state["verdict"] = verdict
                _overlay_state["multimodal"] = mm
                _overlay_state["speech_on"] = speech_on

            # ── Tinh FPS ───────────────────────────────────────────────────
            frame_count += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                print(
                    f"[ExamMonitor] FPS: {fps:.1f} | "
                    f"Faces: {mm.face_count} | "
                    f"MAR: {mm.mar_value:.3f} | "
                    f"Speech: {'ON' if speech_on else 'OFF'} | "
                    f"Verdict: Lv.{verdict.level}"
                )
                frame_count = 0
                fps_timer = time.monotonic()

            # ── Hien thi ──────────────────────────────────────────────────
            display = _draw_overlay(frame)
            cv2.imshow("SecureExam - Giam sat thi cu (Multimodal)", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # 'q' hoac ESC
                break
            if key == ord("s"):  # Toggle simulate speech
                speech_on = not speech_on
                print(f"[ExamMonitor] Speech simulate: {'ON' if speech_on else 'OFF'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[ExamMonitor] Da dung.")


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SecureExam - Giam sat khuon mat ky thi (Multimodal AI)"
    )
    parser.add_argument("--camera", type=int, default=0, help="Index camera (mac dinh: 0)")
    parser.add_argument("--gpu", action="store_true", help="Dung GPU (CUDA), mac dinh CPU")
    parser.add_argument(
        "--model-root",
        default=str(Path(__file__).resolve().parent / "models_cache"),
        help="Thu muc model InsightFace",
    )
    parser.add_argument(
        "--throttle", type=float, default=0.25,
        help="Khoang cach giua 2 lan inference (giay, mac dinh 0.25)",
    )
    parser.add_argument(
        "--simulate-speech", action="store_true",
        help="Bat dau voi speech=ON de test logic cross-check",
    )
    args = parser.parse_args()

    run_monitor(
        camera_index=args.camera,
        model_root=args.model_root,
        ctx_id=0 if args.gpu else -1,
        throttle_interval=args.throttle,
        simulate_speech=args.simulate_speech,
    )