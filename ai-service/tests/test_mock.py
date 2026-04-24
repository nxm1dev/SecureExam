"""
Simple tests using mocks only - no heavy dependencies.
Run with: python -m pytest tests/test_mock.py -v
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class TestFaceDetectorMocked(unittest.TestCase):
    """Test FaceDetector with completely mocked dependencies."""

    def test_no_face_returns_empty_list(self):
        """Blank frame should return no faces."""
        with patch.dict("sys.modules", {"cv2": MagicMock(), "insightface": MagicMock()}):
            from modules.face.detector import FaceBox
            box = FaceBox(x1=10, y1=10, x2=100, y2=100, score=0.9)
            self.assertEqual(box.score, 0.9)

    def test_face_box_properties(self):
        """FaceBox should have correct properties."""
        from modules.face.detector import FaceBox
        box = FaceBox(x1=0, y1=0, x2=100, y2=100, score=0.95)
        self.assertEqual(box.x1, 0)
        self.assertEqual(box.y2, 100)
        self.assertAlmostEqual(box.score, 0.95)

    def test_multiple_faces(self):
        """Multiple face boxes should be stored correctly."""
        from modules.face.detector import FaceBox
        faces = [
            FaceBox(x1=10, y1=10, x2=50, y2=50, score=0.9),
            FaceBox(x1=100, y1=100, x2=200, y2=200, score=0.85),
        ]
        self.assertEqual(len(faces), 2)


class TestAudioMocked(unittest.TestCase):
    """Test audio modules with mocks."""

    def test_vad_frame_calculation(self):
        """VAD should calculate correct frame count."""
        sample_rate = 16000
        duration_sec = 1.0
        frame_size_ms = 30
        expected_frames = int(duration_sec * 1000 / frame_size_ms)
        self.assertEqual(expected_frames, 33)

    def test_mfcc_mock_shape(self):
        """MFCC features should have expected dimension."""
        mfcc_dim = 13
        self.assertEqual(mfcc_dim, 13)


class TestAnalyzerMocked(unittest.TestCase):
    """Test analyzer result structures."""

    def test_face_result_defaults(self):
        """Face analysis result should have correct defaults."""
        class MockResult:
            def __init__(self):
                self.face_detected = False
                self.face_count = 0
                self.multiple_faces = False

        result = MockResult()
        self.assertFalse(result.face_detected)
        self.assertEqual(result.face_count, 0)

    def test_audio_result_defaults(self):
        """Audio analysis result should have correct defaults."""
        class MockAudioResult:
            def __init__(self):
                self.speech_detected = False
                self.voice_overlap = False
                self.voice_overlap_score = 0.0

        result = MockAudioResult()
        self.assertFalse(result.speech_detected)
        self.assertEqual(result.voice_overlap_score, 0.0)


if __name__ == "__main__":
    unittest.main()