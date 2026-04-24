"""
ai-service/modules/audio/vad.py
────────────────────────────────
Voice Activity Detection using WebRTC VAD.
WebRTC VAD is Google's lightweight C-based VAD – extremely fast on CPU.

Splits audio into 30ms frames and classifies each as speech / non-speech.
Returns the ratio of speech frames in a chunk for downstream use.
"""

import struct
from typing import NamedTuple

import webrtcvad

from core.config import get_audio_config
from core.logger import get_logger

log = get_logger(__name__)


class VADResult(NamedTuple):
    """Result of VAD analysis on an audio chunk."""
    has_speech: bool
    speech_ratio: float     # 0.0 – 1.0
    total_frames: int
    speech_frames: int


class VoiceActivityDetector:
    """
    Wraps WebRTC VAD.
    Audio must be 16 kHz, 16-bit PCM mono.
    """

    SUPPORTED_SAMPLE_RATE = 16000
    SUPPORTED_FRAME_MS = [10, 20, 30]  # WebRTC VAD only supports these

    def __init__(self):
        cfg = get_audio_config()
        self._mode: int = cfg.get("vad", {}).get("mode", 3)
        self._frame_ms: int = cfg.get("chunk_duration_ms", 30)
        self._speech_ratio_threshold: float = (
            cfg.get("vad", {}).get("speech_ratio_threshold", 0.3)
        )

        if self._frame_ms not in self.SUPPORTED_FRAME_MS:
            log.warning(
                "Invalid VAD frame duration, defaulting to 30ms",
                requested=self._frame_ms,
            )
            self._frame_ms = 30

        self._vad = webrtcvad.Vad(self._mode)
        self._frame_bytes = self.SUPPORTED_SAMPLE_RATE * 2 * self._frame_ms // 1000

    def process(self, pcm_bytes: bytes) -> VADResult:
        """
        Process raw 16kHz 16-bit PCM mono bytes.

        Args:
            pcm_bytes: Raw audio bytes.

        Returns:
            VADResult with speech detection summary.
        """
        if len(pcm_bytes) < self._frame_bytes:
            return VADResult(
                has_speech=False, speech_ratio=0.0, total_frames=0, speech_frames=0
            )

        total_frames = 0
        speech_frames = 0

        # Slide through the buffer in fixed-size frames
        for start in range(0, len(pcm_bytes) - self._frame_bytes + 1, self._frame_bytes):
            frame = pcm_bytes[start : start + self._frame_bytes]
            total_frames += 1
            try:
                if self._vad.is_speech(frame, self.SUPPORTED_SAMPLE_RATE):
                    speech_frames += 1
            except Exception:
                pass  # Malformed frame – skip

        ratio = speech_frames / total_frames if total_frames > 0 else 0.0
        has_speech = ratio >= self._speech_ratio_threshold

        return VADResult(
            has_speech=has_speech,
            speech_ratio=ratio,
            total_frames=total_frames,
            speech_frames=speech_frames,
        )


# Module-level singleton
_vad = VoiceActivityDetector()


def get_vad() -> VoiceActivityDetector:
    return _vad
