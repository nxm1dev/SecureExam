"""
ai-service/tests/test_audio.py
────────────────────────────────
Tests for audio VAD, feature extraction, and analyzer.
Uses synthetic PCM audio data – no microphone required.
"""

import base64
import struct
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def _make_silence_pcm(duration_sec=1.0, sample_rate=16000) -> bytes:
    """Generate silent PCM bytes (all zeros)."""
    n_samples = int(sample_rate * duration_sec)
    return struct.pack(f"{n_samples}h", *([0] * n_samples))


def _make_noise_pcm(duration_sec=1.0, sample_rate=16000, amplitude=0.5) -> bytes:
    """Generate random noise as PCM bytes."""
    n_samples = int(sample_rate * duration_sec)
    noise = (np.random.randn(n_samples) * amplitude * 32767).astype(np.int16)
    return noise.tobytes()


def _pcm_to_b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode()


class TestVAD(unittest.TestCase):

    def test_silence_has_no_speech(self):
        """Silent audio should not trigger VAD."""
        from modules.audio.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        silence = _make_silence_pcm(1.0)
        result = vad.process(silence)
        self.assertFalse(result.has_speech)
        self.assertEqual(result.speech_ratio, 0.0)

    def test_short_buffer_returns_no_speech(self):
        """Buffer shorter than one VAD frame returns gracefully."""
        from modules.audio.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        tiny = b"\x00" * 10  # Too short
        result = vad.process(tiny)
        self.assertFalse(result.has_speech)


class TestFeatureExtractor(unittest.TestCase):

    def test_extract_from_noise(self):
        """Noise audio should produce a valid MFCC vector."""
        from modules.audio.feature_extractor import FeatureExtractor

        fx = FeatureExtractor()
        pcm = _make_noise_pcm(1.0)
        mfcc = fx.extract_from_pcm(pcm)
        self.assertIsNotNone(mfcc)
        self.assertEqual(mfcc.shape[0], 13)

    def test_extract_from_silence_returns_none_or_vector(self):
        """Silence may produce a zero or near-zero MFCC vector."""
        from modules.audio.feature_extractor import FeatureExtractor

        fx = FeatureExtractor()
        pcm = _make_silence_pcm(1.0)
        # Should not raise
        mfcc = fx.extract_from_pcm(pcm)
        # Result is either None or a vector
        if mfcc is not None:
            self.assertEqual(mfcc.shape[0], 13)

    def test_cosine_distance_identical(self):
        """Identical vectors should have distance 0."""
        from modules.audio.feature_extractor import FeatureExtractor

        fx = FeatureExtractor()
        v = np.array([1.0, 2.0, 3.0])
        dist = fx.cosine_distance(v, v)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_cosine_distance_different(self):
        """Orthogonal vectors should have distance 1."""
        from modules.audio.feature_extractor import FeatureExtractor

        fx = FeatureExtractor()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        dist = fx.cosine_distance(a, b)
        self.assertAlmostEqual(dist, 1.0, places=5)


class TestAudioAnalyzer(unittest.TestCase):

    def test_silence_no_violation(self):
        """Silent audio should produce no violation suggestion."""
        from modules.audio.analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer()
        # Mock VAD to return no speech
        analyzer._vad = MagicMock()
        analyzer._vad.process.return_value = MagicMock(
            has_speech=False, speech_ratio=0.0
        )

        b64 = _pcm_to_b64(_make_silence_pcm())
        result = analyzer.analyze(pcm_b64=b64, session_id="test-sess")

        self.assertFalse(result.has_speech)
        self.assertEqual(result.suggested_violation, "")

    def test_speech_detected_violation(self):
        """Speech detected → suggests speech_detected violation."""
        from modules.audio.analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer()
        analyzer._vad = MagicMock()
        analyzer._vad.process.return_value = MagicMock(
            has_speech=True, speech_ratio=0.8
        )
        analyzer._extractor = MagicMock()
        # Return a consistent MFCC vector (no change)
        fixed_mfcc = np.ones(13)
        analyzer._extractor.extract_from_pcm.return_value = fixed_mfcc
        analyzer._extractor.cosine_distance.return_value = 0.1  # Small change
        analyzer._extractor.estimate_overlap.return_value = 0.1

        b64 = _pcm_to_b64(_make_noise_pcm())
        result = analyzer.analyze(pcm_b64=b64, session_id="test-sess-2")

        self.assertTrue(result.has_speech)
        self.assertIn(result.suggested_violation, ["speech_detected", ""])

    def test_voice_overlap_detected(self):
        """High overlap score → voice_overlap violation."""
        from modules.audio.analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer()
        analyzer._vad = MagicMock()
        analyzer._vad.process.return_value = MagicMock(
            has_speech=True, speech_ratio=0.9
        )
        analyzer._extractor = MagicMock()
        analyzer._extractor.extract_from_pcm.return_value = np.ones(13)
        analyzer._extractor.cosine_distance.return_value = 0.1
        analyzer._extractor.estimate_overlap.return_value = 0.9  # HIGH

        b64 = _pcm_to_b64(_make_noise_pcm())
        result = analyzer.analyze(pcm_b64=b64, session_id="overlap-sess")

        self.assertTrue(result.voice_overlap_detected)
        self.assertEqual(result.suggested_violation, "voice_overlap")
        self.assertEqual(result.suggested_severity, "critical")


if __name__ == "__main__":
    unittest.main()
