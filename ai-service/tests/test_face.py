"""
ai-service/tests/test_face.py
──────────────────────────────
Tests for ExamFaceDetector, FaceDetector, and FaceAnalyzer.

Architecture tested:
  ExamFaceDetector  – detection-only, throttle, callback dispatch
  FaceDetector      – backward-compat wrapper
  FaceAnalyzer      – orchestrator (detection-first, lazy recognition)

All tests use mocks – no real camera or ONNX model required.
Run: python -m pytest tests/test_face.py -v
"""

from __future__ import annotations

import base64
import io
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _make_blank_frame_b64(width: int = 640, height: int = 480) -> str:
    """Base64-encoded blank PNG (no cv2 required)."""
    from PIL import Image
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    img = Image.fromarray(frame, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _mock_insight_face(bboxes: list[list[float]]) -> MagicMock:
    """
    Build a MagicMock that mimics insightface.app.FaceAnalysis.get().
    Each entry in bboxes is [x1, y1, x2, y2].
    """
    faces = []
    for b in bboxes:
        f = MagicMock()
        f.bbox = np.array(b, dtype=float)
        f.det_score = 0.95
        faces.append(f)

    mock_app = MagicMock()
    mock_app.get.return_value = faces
    return mock_app


# ──────────────────────────────────────────────────────────────────────────────
# ExamFaceDetector tests
# ──────────────────────────────────────────────────────────────────────────────
class TestExamFaceDetector(unittest.TestCase):
    """Unit tests for ExamFaceDetector (detection-only core)."""

    def _make_detector(self, bboxes: list[list[float]], **kwargs):
        from modules.face.exam_face_detector import ExamFaceDetector
        det = ExamFaceDetector(throttle_interval=0.0, **kwargs)
        det._app = _mock_insight_face(bboxes)
        return det

    # ── Basic counts ──────────────────────────────────────────────────────────
    def test_no_face_returns_zero_count(self):
        det = self._make_detector([])
        result = det.process_frame(_blank_frame())
        self.assertEqual(result.face_count, 0)
        self.assertEqual(result.bboxes, [])

    def test_one_face_returns_one_count(self):
        det = self._make_detector([[10, 10, 100, 100]])
        result = det.process_frame(_blank_frame())
        self.assertEqual(result.face_count, 1)
        self.assertEqual(len(result.bboxes), 1)

    def test_two_faces_returns_two_count(self):
        det = self._make_detector([
            [10, 10, 100, 100],
            [200, 10, 300, 100],
        ])
        result = det.process_frame(_blank_frame())
        self.assertEqual(result.face_count, 2)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def test_on_no_face_callback_called(self):
        cb = MagicMock()
        det = self._make_detector([], on_no_face=cb)
        det.process_frame(_blank_frame())
        cb.assert_called_once()

    def test_on_normal_callback_called_with_bbox(self):
        cb = MagicMock()
        det = self._make_detector([[10, 20, 110, 120]], on_normal=cb)
        det.process_frame(_blank_frame())
        cb.assert_called_once()
        _, bbox = cb.call_args[0]
        self.assertEqual(len(bbox), 4)  # [x1, y1, x2, y2]

    def test_on_multiple_faces_callback_called_with_count(self):
        cb = MagicMock()
        det = self._make_detector(
            [[10, 10, 100, 100], [200, 10, 300, 100]],
            on_multiple_faces=cb,
        )
        det.process_frame(_blank_frame())
        cb.assert_called_once()
        _, face_count = cb.call_args[0]
        self.assertEqual(face_count, 2)

    def test_callbacks_not_required(self):
        """No callbacks registered → process_frame should not raise."""
        det = self._make_detector([[10, 10, 100, 100]])
        det.process_frame(_blank_frame())  # Must not raise

    # ── Throttle ──────────────────────────────────────────────────────────────
    def test_throttle_uses_cached_result(self):
        """Second call within throttle window must NOT call model again."""
        from modules.face.exam_face_detector import ExamFaceDetector
        det = ExamFaceDetector(throttle_interval=60.0)  # Very long window
        det._app = _mock_insight_face([[10, 10, 100, 100]])

        det.process_frame(_blank_frame())  # First call: runs inference
        det._app.get.reset_mock()

        det.process_frame(_blank_frame())  # Second call: should be cached
        det._app.get.assert_not_called()

    def test_throttle_runs_inference_after_interval(self):
        """After throttle window expires, inference should run again."""
        from modules.face.exam_face_detector import ExamFaceDetector
        det = ExamFaceDetector(throttle_interval=0.0)  # Always infer
        det._app = _mock_insight_face([[10, 10, 100, 100]])

        det.process_frame(_blank_frame())
        call_count_1 = det._app.get.call_count

        det.process_frame(_blank_frame())
        call_count_2 = det._app.get.call_count

        self.assertEqual(call_count_2, call_count_1 + 1)

    # ── Reset ─────────────────────────────────────────────────────────────────
    def test_reset_clears_state(self):
        from modules.face.exam_face_detector import DetectionResult
        det = self._make_detector([[10, 10, 100, 100]])
        det.process_frame(_blank_frame())
        det.reset()
        self.assertEqual(det.last_result.face_count, 0)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    def test_preprocess_downsizes_large_frame(self):
        from modules.face.exam_face_detector import ExamFaceDetector
        large = np.zeros((1080, 1920, 3), dtype=np.uint8)
        small = ExamFaceDetector._preprocess(large)
        self.assertEqual(small.shape[1], ExamFaceDetector._PREPROCESS_W)
        self.assertEqual(small.shape[0], ExamFaceDetector._PREPROCESS_H)

    def test_preprocess_skips_already_sized_frame(self):
        from modules.face.exam_face_detector import ExamFaceDetector
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = ExamFaceDetector._preprocess(frame)
        self.assertIs(result, frame)  # Same object (no copy)


# ──────────────────────────────────────────────────────────────────────────────
# FaceDetector (backward-compat wrapper) tests
# ──────────────────────────────────────────────────────────────────────────────
class TestFaceDetector(unittest.TestCase):
    """Test backward-compatible FaceDetector wrapper."""

    def _make_detector(self, bboxes: list[list[float]]):
        from modules.face.detector import FaceDetector
        det = FaceDetector()
        det._model = _mock_insight_face(bboxes)  # Uses property shim
        return det

    def test_no_face_returns_empty_list(self):
        det = self._make_detector([])
        self.assertEqual(det.detect(_blank_frame()), [])

    def test_one_face_returns_one_facebox(self):
        from modules.face.detector import FaceBox
        det = self._make_detector([[100, 100, 300, 300]])
        result = det.detect(_blank_frame())
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], FaceBox)

    def test_two_faces_returns_two_faceboxes(self):
        det = self._make_detector([
            [10, 10, 100, 100],
            [200, 10, 300, 100],
        ])
        result = det.detect(_blank_frame())
        self.assertEqual(len(result), 2)

    def test_facebox_has_score_field(self):
        """FaceBox must still have score for backward compat."""
        from modules.face.detector import FaceBox
        box = FaceBox(x1=0, y1=0, x2=100, y2=100, score=0.9)
        self.assertAlmostEqual(box.score, 0.9)

    def test_model_shim_allows_mock_injection(self):
        """Setting detector._model should update the underlying _app."""
        from modules.face.detector import FaceDetector
        det = FaceDetector()
        mock_app = _mock_insight_face([[10, 10, 100, 100]])
        det._model = mock_app
        self.assertIs(det._exam_detector._app, mock_app)


# ──────────────────────────────────────────────────────────────────────────────
# FaceAnalyzer tests
# ──────────────────────────────────────────────────────────────────────────────
class TestFaceAnalyzer(unittest.TestCase):
    """Integration tests for FaceAnalyzer orchestrator."""

    def _make_analyzer(self, bboxes: list[list[float]]):
        """Create FaceAnalyzer with mocked ExamFaceDetector."""
        from modules.face.analyzer import FaceAnalyzer
        from modules.face.exam_face_detector import DetectionResult

        analyzer = FaceAnalyzer()
        # Inject mock detection result
        mock_det = MagicMock()
        mock_det.process_frame.return_value = DetectionResult(
            face_count=len(bboxes),
            bboxes=[list(map(int, b)) for b in bboxes],
        )
        analyzer._detector = mock_det
        # Mock recognizer (should NOT be called unless explicitly needed)
        analyzer._recognizer = MagicMock()
        analyzer._recognizer.extract_embedding.return_value = None
        return analyzer

    def test_bad_frame_returns_empty_result(self):
        from modules.face.analyzer import FaceAnalyzer
        analyzer = FaceAnalyzer()
        result = analyzer.analyze(frame_b64="not_valid_base64!!!!")
        self.assertFalse(result.face_detected)
        self.assertEqual(result.face_count, 0)

    def test_no_face_sets_correct_flags(self):
        analyzer = self._make_analyzer([])
        result = analyzer.analyze(frame_b64=_make_blank_frame_b64())
        self.assertFalse(result.face_detected)
        self.assertFalse(result.multiple_faces)
        self.assertEqual(result.face_count, 0)

    def test_one_face_sets_face_detected_true(self):
        analyzer = self._make_analyzer([[10, 10, 100, 100]])
        result = analyzer.analyze(frame_b64=_make_blank_frame_b64())
        self.assertTrue(result.face_detected)
        self.assertFalse(result.multiple_faces)
        self.assertEqual(result.face_count, 1)

    def test_two_faces_sets_multiple_faces_flag(self):
        analyzer = self._make_analyzer([
            [10, 10, 100, 100],
            [200, 10, 300, 100],
        ])
        analyzer._max_faces = 1
        result = analyzer.analyze(frame_b64=_make_blank_frame_b64())
        self.assertTrue(result.multiple_faces)
        self.assertEqual(result.face_count, 2)

    def test_recognition_skipped_without_request(self):
        """Recognizer must NOT be called during normal surveillance (no reference)."""
        analyzer = self._make_analyzer([[10, 10, 100, 100]])
        analyzer.analyze(frame_b64=_make_blank_frame_b64())
        analyzer._recognizer.extract_embedding.assert_not_called()

    def test_recognition_runs_when_reference_provided(self):
        """Recognizer MUST run when reference_embedding_b64 is given."""
        analyzer = self._make_analyzer([[10, 10, 100, 100]])
        # Provide a fake embedding
        fake_emb = np.random.rand(512).astype(np.float32)
        analyzer._recognizer.extract_embedding.return_value = fake_emb
        analyzer._recognizer.embedding_to_b64.return_value = "fake_b64"
        analyzer._recognizer.is_same_person.return_value = (True, 0.2)

        result = analyzer.analyze(
            frame_b64=_make_blank_frame_b64(),
            reference_embedding_b64="some_ref_b64",
        )

        analyzer._recognizer.extract_embedding.assert_called_once()
        self.assertTrue(result.identity_checked)
        self.assertTrue(result.identity_match)

    def test_extract_embedding_flag_triggers_recognition(self):
        """extract_embedding=True should force recognition even without reference."""
        analyzer = self._make_analyzer([[10, 10, 100, 100]])
        fake_emb = np.random.rand(512).astype(np.float32)
        analyzer._recognizer.extract_embedding.return_value = fake_emb
        analyzer._recognizer.embedding_to_b64.return_value = "fake_b64"

        result = analyzer.analyze(
            frame_b64=_make_blank_frame_b64(),
            extract_embedding=True,
        )

        analyzer._recognizer.extract_embedding.assert_called_once()
        self.assertIsNotNone(result.embedding_b64)
        self.assertFalse(result.identity_checked)  # No reference given

    def test_no_face_skips_recognition_entirely(self):
        """When no face detected, recognizer must never be called."""
        analyzer = self._make_analyzer([])
        analyzer.analyze(
            frame_b64=_make_blank_frame_b64(),
            reference_embedding_b64="some_ref",
        )
        analyzer._recognizer.extract_embedding.assert_not_called()

    def test_bboxes_in_result(self):
        """Result bboxes should match detected faces."""
        analyzer = self._make_analyzer([[10, 20, 110, 120]])
        result = analyzer.analyze(frame_b64=_make_blank_frame_b64())
        self.assertEqual(len(result.bboxes), 1)
        self.assertEqual(result.bboxes[0]["x1"], 10)
        self.assertEqual(result.bboxes[0]["y1"], 20)


if __name__ == "__main__":
    unittest.main()
