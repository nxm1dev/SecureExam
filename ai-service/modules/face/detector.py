"""
ai-service/modules/face/detector.py
─────────────────────────────────────
Thin wrapper duy trì backward-compatibility cho FaceBox và FaceDetector.

Sau khi refactor sang ExamFaceDetector, module này:
  - Giữ lại FaceBox (NamedTuple) để tests và code cũ không bị vỡ.
  - FaceDetector.detect() vẫn hoạt động nhưng bên trong dùng ExamFaceDetector.
  - Không còn là entry-point chính – FaceAnalyzer dùng ExamFaceDetector trực tiếp.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional

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

log = get_logger(__name__)


class FaceBox(NamedTuple):
    """Detected face bounding box. score=1.0 (placeholder – det_score trong ExamFaceDetector)."""
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0


class FaceDetector:
    """
    Backward-compatible wrapper quanh ExamFaceDetector.

    Dùng detect() cho HTTP API hoặc test. Mỗi call là một frame độc lập
    (throttle_interval=0) nên luôn chạy inference ngay.
    """

    def __init__(
        self,
        on_multiple_faces: Optional[OnMultipleFacesCB] = None,
        on_no_face: Optional[OnNoFaceCB] = None,
        on_normal: Optional[OnNormalCB] = None,
    ) -> None:
        self._config = get_camera_config()
        self._min_confidence: float = (
            self._config.get("detection", {}).get("min_face_confidence", 0.5)
        )
        settings = get_settings()
        os.makedirs(settings.model_cache_dir, exist_ok=True)

        self._exam_detector = ExamFaceDetector(
            model_root=settings.model_cache_dir,
            on_multiple_faces=on_multiple_faces,
            on_no_face=on_no_face,
            on_normal=on_normal,
            det_size=(640, 640),
            det_thresh=self._min_confidence,
            throttle_interval=0.0,   # API mode: always infer immediately
            ctx_id=-1,               # CPU
        )

    # ── Internal model shim (for tests that mock detector._model) ─────────────
    @property
    def _model(self):
        """Legacy shim – tests may patch this."""
        return self._exam_detector._app

    @_model.setter
    def _model(self, value):
        """Allow tests to inject a mock FaceAnalysis instance directly."""
        self._exam_detector._app = value

    # ── Public API ─────────────────────────────────────────────────────────────
    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        """
        Detect faces in a BGR numpy image (as returned by OpenCV).

        Returns:
            List of FaceBox named tuples (score always 1.0 – use det_score
            from ExamFaceDetector.last_result if precise confidence needed).
        """
        try:
            result: DetectionResult = self._exam_detector.process_frame(frame_bgr)
        except Exception as e:
            log.warning("Face detection inference error", error=str(e))
            return []

        return [
            FaceBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], score=1.0)
            for b in result.bboxes
        ]

    @property
    def exam_detector(self) -> ExamFaceDetector:
        """Expose the underlying ExamFaceDetector for camera-stream usage."""
        return self._exam_detector


# ── Module-level singleton ────────────────────────────────────────────────────
_detector = FaceDetector()


def get_detector() -> FaceDetector:
    return _detector
