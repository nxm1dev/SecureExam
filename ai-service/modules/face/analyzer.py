"""
ai-service/modules/face/analyzer.py
─────────────────────────────────────
Orchestrates face detection + recognition for a single frame.

Architecture (tách biệt Detection vs Recognition):
  ┌─────────────────────────────────────────────────────────┐
  │  analyze()                                              │
  │                                                         │
  │  1. Decode base64 JPEG frame                            │
  │                                                         │
  │  2. Detection  ← ExamFaceDetector (buffalo_l, det-only) │
  │     • Throttled / cached internally                     │
  │     • Fires on_multiple_faces / on_no_face / on_normal  │
  │     • Returns face_count + bboxes                       │
  │                                                         │
  │  3. Recognition  ← FaceRecognizer (buffalo_l, full)     │
  │     • ONLY runs when reference_embedding_b64 supplied   │
  │     • OR when caller explicitly requests embedding      │
  │     • Skipped entirely during normal surveillance loop  │
  └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from core.config import get_camera_config, get_settings
from core.logger import get_logger
from modules.face.exam_face_detector import (
    BBox,
    DetectionResult,
    ExamFaceDetector,
    OnMultipleFacesCB,
    OnNoFaceCB,
    OnNormalCB,
)
from modules.face.recognizer import get_recognizer

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FaceAnalysisResult:
    """Kết quả phân tích một camera frame."""

    # ── Detection (luôn có) ───────────────────────────────────────────────────
    face_count: int = 0
    face_detected: bool = False
    multiple_faces: bool = False
    # Detected bounding boxes  [{x1, y1, x2, y2}]
    bboxes: list[dict] = field(default_factory=list)

    # ── Recognition (chỉ khi được yêu cầu) ───────────────────────────────────
    identity_checked: bool = False
    identity_match: bool = False
    identity_distance: float = 1.0
    # Embedding của khuôn mặt chính (dùng cho luồng đăng ký thí sinh)
    embedding_b64: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────
class FaceAnalyzer:
    """
    Orchestrator phân tích khuôn mặt cho một HTTP request.

    Detection:
        Dùng ``ExamFaceDetector`` (buffalo_l, detection-only).
        Với API calls, throttle_interval=0 → mỗi frame được inference ngay.

    Recognition:
        Dùng ``FaceRecognizer`` (buffalo_l, detection+recognition).
        Chỉ kích hoạt khi:
          (a) ``reference_embedding_b64`` được truyền vào  → identity check
          (b) ``extract_embedding=True``                   → lấy embedding mới

    Callbacks:
        Người dùng có thể đăng ký callbacks vào ``ExamFaceDetector`` bên trong
        để nhận sự kiện giám sát (on_multiple_faces, on_no_face, on_normal)
        ngay trong luồng HTTP phân tích.
    """

    def __init__(
        self,
        on_multiple_faces: Optional[OnMultipleFacesCB] = None,
        on_no_face: Optional[OnNoFaceCB] = None,
        on_normal: Optional[OnNormalCB] = None,
    ) -> None:
        settings = get_settings()
        camera_cfg = get_camera_config()

        det_cfg = camera_cfg.get("detection", {})
        self._max_faces: int = det_cfg.get("max_allowed_faces", 1)
        min_conf: float = det_cfg.get("min_face_confidence", 0.5)

        # ── Detection engine (nhẹ, chỉ detection ONNX) ────────────────────────
        self._detector = ExamFaceDetector(
            model_root=settings.model_cache_dir,
            on_multiple_faces=on_multiple_faces,
            on_no_face=on_no_face,
            on_normal=on_normal,
            det_size=(640, 640),
            det_thresh=min_conf,
            # API mode: không throttle – mỗi HTTP request là 1 frame độc lập
            throttle_interval=0.0,
            ctx_id=-1,  # CPU; đặt 0 nếu có GPU
        )

        # ── Recognition engine (nặng, chỉ load khi cần) ──────────────────────
        self._recognizer = get_recognizer()

        log.info("FaceAnalyzer ready (detection-first architecture)")

    # ── Internal helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _decode_frame(frame_b64: str) -> Optional[np.ndarray]:
        """Giải mã base64 JPEG → BGR numpy array. Trả về None nếu lỗi."""
        try:
            img_bytes = base64.b64decode(frame_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("OpenCV could not decode image")
            return frame
        except Exception as e:
            log.warning("Frame decode failed", error=str(e))
            return None

    # ── Public API ─────────────────────────────────────────────────────────────
    def analyze(
        self,
        frame_b64: str,
        reference_embedding_b64: Optional[str] = None,
        extract_embedding: bool = False,
    ) -> FaceAnalysisResult:
        """
        Phân tích một frame base64-JPEG.

        Parameters
        ----------
        frame_b64 : str
            Ảnh JPEG mã hoá base64.
        reference_embedding_b64 : str | None
            Embedding tham chiếu (từ DB) để so sánh danh tính thí sinh.
            Nếu None → bỏ qua bước recognition hoàn toàn.
        extract_embedding : bool
            Nếu True, luôn trích xuất embedding khuôn mặt chính
            (dùng cho luồng đăng ký). Tự động set True khi
            ``reference_embedding_b64`` được cung cấp.

        Returns
        -------
        FaceAnalysisResult
        """
        result = FaceAnalysisResult()

        # ── 1. Decode ──────────────────────────────────────────────────────────
        frame = self._decode_frame(frame_b64)
        if frame is None:
            return result  # Empty result on bad frame

        # ── 2. Detection (ExamFaceDetector) ────────────────────────────────────
        det: DetectionResult = self._detector.process_frame(frame)
        # Callbacks (on_multiple_faces / on_no_face / on_normal) đã được gọi
        # bên trong process_frame() ↑

        result.face_count = det.face_count
        result.face_detected = det.face_count > 0
        result.multiple_faces = det.face_count > self._max_faces
        result.bboxes = [
            {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
            for b in det.bboxes
        ]

        if not result.face_detected:
            return result  # Không có mặt → dừng sớm, bỏ qua recognition

        # ── 3. Recognition (chỉ khi cần) ──────────────────────────────────────
        need_recognition = extract_embedding or (reference_embedding_b64 is not None)
        if not need_recognition:
            return result

        emb = self._recognizer.extract_embedding(frame)
        if emb is None:
            log.debug("Recognition: no embedding extracted")
            return result

        result.embedding_b64 = self._recognizer.embedding_to_b64(emb)

        if reference_embedding_b64:
            result.identity_checked = True
            result.identity_match, result.identity_distance = (
                self._recognizer.is_same_person(emb, reference_embedding_b64)
            )
            log.debug(
                "Identity check",
                match=result.identity_match,
                distance=round(result.identity_distance, 4),
            )

        return result

    @property
    def exam_detector(self) -> ExamFaceDetector:
        """
        Trả về ExamFaceDetector bên trong.
        Dùng khi muốn chạy camera stream trực tiếp thay vì qua HTTP API.
        """
        return self._detector


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────────────────
_analyzer: Optional[FaceAnalyzer] = None
_analyzer_lock = threading.Lock()


def get_analyzer(
    on_multiple_faces: Optional[OnMultipleFacesCB] = None,
    on_no_face: Optional[OnNoFaceCB] = None,
    on_normal: Optional[OnNormalCB] = None,
) -> FaceAnalyzer:
    """
    Trả về singleton FaceAnalyzer.

    Lần đầu gọi với callbacks sẽ khởi tạo instance với callbacks đó.
    Các lần sau trả về instance đã có (callbacks không đổi).
    """
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:  # Double-checked locking
                _analyzer = FaceAnalyzer(
                    on_multiple_faces=on_multiple_faces,
                    on_no_face=on_no_face,
                    on_normal=on_normal,
                )
    return _analyzer
