"""
ai-service/modules/audio/analyzer.py
──────────────────────────────────────
Audio analysis orchestrator.
Combines VAD + MFCC feature extraction to detect:
  1. Speech presence
  2. Rapid speaker changes (possible multiple speakers taking turns)
  3. Voice overlap (multiple people talking simultaneously)

State is maintained per session in a lightweight in-memory store.
The store is automatically pruned on session end.
"""

import base64
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.config import get_audio_config
from core.logger import get_logger
from modules.audio.feature_extractor import get_extractor
from modules.audio.vad import VADResult, get_vad

log = get_logger(__name__)

# ── Per-session sliding window ────────────────────────────────────
# Stores recent MFCC vectors + timestamps for change detection
MAX_WINDOW = 20  # Keep last 20 chunk vectors per session


@dataclass
class AudioAnalysisResult:
    """Result of analyzing one audio chunk."""
    has_speech: bool = False
    speech_ratio: float = 0.0
    # Speaker change detection
    speaker_change_detected: bool = False
    change_distance: float = 0.0
    # Multi-speaker detection
    rapid_changes_detected: bool = False
    rapid_change_count: int = 0
    # Overlap detection
    voice_overlap_detected: bool = False
    overlap_score: float = 0.0
    # Suggested violation type (empty = no violation)
    suggested_violation: str = ""
    suggested_severity: str = "low"


class SessionAudioState:
    """Sliding window of MFCC vectors for a single session."""

    def __init__(self):
        self.vectors: deque = deque(maxlen=MAX_WINDOW)       # (timestamp, ndarray)
        self.change_times: deque = deque(maxlen=MAX_WINDOW)  # timestamps of speaker changes


class AudioAnalyzer:
    """
    Stateful audio analyzer – maintains per-session MFCC history.
    Not thread-safe – use one instance per service worker.
    """

    def __init__(self):
        self._vad = get_vad()
        self._extractor = get_extractor()
        self._cfg = get_audio_config()

        anomaly_cfg = self._cfg.get("anomaly", {})
        self._change_threshold: float = anomaly_cfg.get("change_threshold", 0.35)
        self._rapid_change_count: int = anomaly_cfg.get("rapid_change_count", 3)
        self._rapid_window_sec: float = anomaly_cfg.get("rapid_change_window_seconds", 15)
        self._overlap_energy_ratio: float = anomaly_cfg.get("overlap_energy_ratio", 0.6)

        # Session state store: session_id → SessionAudioState
        self._sessions: dict[str, SessionAudioState] = {}

    def _get_state(self, session_id: str) -> SessionAudioState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionAudioState()
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Release memory when session ends."""
        self._sessions.pop(session_id, None)

    def analyze(
        self,
        pcm_b64: str,
        session_id: str,
    ) -> AudioAnalysisResult:
        """
        Analyze one audio chunk for a session.

        Args:
            pcm_b64: Base64-encoded 16kHz 16-bit mono PCM bytes.
            session_id: Unique session identifier for state tracking.

        Returns:
            AudioAnalysisResult with detected anomalies.
        """
        result = AudioAnalysisResult()

        # ── Decode PCM ──────────────────────────────────────────────
        try:
            pcm_bytes = base64.b64decode(pcm_b64)
        except Exception as e:
            log.warning("PCM decode failed", error=str(e))
            return result

        # ── VAD: Is there speech? ───────────────────────────────────
        vad_result: VADResult = self._vad.process(pcm_bytes)
        result.has_speech = vad_result.has_speech
        result.speech_ratio = vad_result.speech_ratio

        if not vad_result.has_speech:
            # No speech – nothing further to do
            return result

        # ── MFCC extraction ─────────────────────────────────────────
        mfcc = self._extractor.extract_from_pcm(pcm_bytes)
        if mfcc is None:
            return result

        state = self._get_state(session_id)
        now = time.time()

        # ── Speaker change detection ────────────────────────────────
        if state.vectors:
            prev_time, prev_mfcc = state.vectors[-1]
            dist = self._extractor.cosine_distance(mfcc, prev_mfcc)
            result.change_distance = dist

            if dist > self._change_threshold:
                result.speaker_change_detected = True
                state.change_times.append(now)

        state.vectors.append((now, mfcc))

        # ── Rapid change detection ──────────────────────────────────
        # Count changes within the rolling time window
        cutoff = now - self._rapid_window_sec
        recent_changes = [t for t in state.change_times if t > cutoff]
        result.rapid_change_count = len(recent_changes)

        if result.rapid_change_count >= self._rapid_change_count:
            result.rapid_changes_detected = True

        # ── Overlap detection ───────────────────────────────────────
        overlap_score = self._extractor.estimate_overlap(pcm_bytes)
        result.overlap_score = overlap_score
        result.voice_overlap_detected = overlap_score >= self._overlap_energy_ratio

        # ── Suggest violation type ──────────────────────────────────
        severity_cfg = self._cfg.get("severity", {})
        if result.voice_overlap_detected:
            result.suggested_violation = "voice_overlap"
            result.suggested_severity = severity_cfg.get("voice_overlap", "critical")
        elif result.rapid_changes_detected:
            result.suggested_violation = "rapid_voice_change"
            result.suggested_severity = severity_cfg.get("rapid_changes", "high")
        elif result.speaker_change_detected:
            result.suggested_violation = "multiple_voices"
            result.suggested_severity = severity_cfg.get("multiple_voices", "high")
        elif result.has_speech:
            result.suggested_violation = "speech_detected"
            result.suggested_severity = severity_cfg.get("speech_detected", "low")

        return result


# Module-level singleton
_audio_analyzer = AudioAnalyzer()


def get_audio_analyzer() -> AudioAnalyzer:
    return _audio_analyzer
