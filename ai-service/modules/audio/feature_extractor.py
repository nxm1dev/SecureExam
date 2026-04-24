"""
ai-service/modules/audio/feature_extractor.py
──────────────────────────────────────────────
Extracts MFCC features from audio chunks for speaker change detection.

MFCCs (Mel-Frequency Cepstral Coefficients) are a compact representation
of the spectral envelope of audio. They are widely used for speaker
identification because different voices produce different MFCC patterns.

No speaker enrollment is required – we only compare adjacent chunks.
"""

import base64
import io
from typing import Optional

import librosa
import numpy as np

from core.config import get_audio_config
from core.logger import get_logger

log = get_logger(__name__)


class FeatureExtractor:
    """
    Extracts mean MFCC vector from a PCM audio chunk.
    The mean vector (shape: n_mfcc,) is used as a compact fingerprint
    for speaker change detection via cosine similarity.
    """

    def __init__(self):
        cfg = get_audio_config()
        self._sample_rate: int = cfg.get("sample_rate", 16000)
        self._n_mfcc: int = cfg.get("features", {}).get("n_mfcc", 13)
        self._hop_length: int = cfg.get("features", {}).get("hop_length", 512)

    def extract_from_pcm(self, pcm_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract mean MFCC from raw 16kHz 16-bit mono PCM bytes.

        Returns:
            Mean MFCC vector of shape (n_mfcc,), or None on error.
        """
        try:
            # Convert int16 PCM bytes → float32 waveform
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            audio /= 32768.0  # Normalize to [-1, 1]

            if len(audio) < self._hop_length:
                return None

            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self._sample_rate,
                n_mfcc=self._n_mfcc,
                hop_length=self._hop_length,
            )
            # Return mean over time axis – compact single vector
            return np.mean(mfccs, axis=1)

        except Exception as e:
            log.warning("MFCC extraction failed", error=str(e))
            return None

    def extract_from_b64(self, pcm_b64: str) -> Optional[np.ndarray]:
        """Decode base64 PCM and extract MFCC."""
        try:
            pcm_bytes = base64.b64decode(pcm_b64)
            return self.extract_from_pcm(pcm_bytes)
        except Exception as e:
            log.warning("Base64 decode for feature extraction failed", error=str(e))
            return None

    def vector_to_b64(self, vector: np.ndarray) -> str:
        """Serialize MFCC vector to base64 for transport."""
        return base64.b64encode(vector.astype(np.float32).tobytes()).decode()

    def b64_to_vector(self, b64: str) -> np.ndarray:
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype=np.float32)

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance between two vectors. 0 = identical, 2 = opposite."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 2.0
        return float(1.0 - np.dot(a, b) / (norm_a * norm_b))

    def estimate_overlap(self, pcm_bytes: bytes) -> float:
        """
        Estimate the probability of voice overlap by checking energy
        distribution across low and high frequency bands simultaneously.

        Returns:
            Overlap score [0, 1]. >0.6 suggests overlapping voices.
        """
        try:
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            audio /= 32768.0

            # Short-time Fourier transform
            stft = np.abs(librosa.stft(audio))
            # Low band: 0–500 Hz, High band: 500–4000 Hz
            n_low = int(500 / (self._sample_rate / 2) * stft.shape[0])
            n_high = min(int(4000 / (self._sample_rate / 2) * stft.shape[0]), stft.shape[0])

            low_energy = np.mean(stft[:n_low, :])
            high_energy = np.mean(stft[n_low:n_high, :])

            if high_energy == 0:
                return 0.0
            # Overlap indicated when both bands are active at similar levels
            ratio = low_energy / (high_energy + 1e-6)
            # Normalize to [0, 1]
            return float(min(ratio, 1.0))

        except Exception as e:
            log.warning("Overlap estimation failed", error=str(e))
            return 0.0


# Module-level singleton
_extractor = FeatureExtractor()


def get_extractor() -> FeatureExtractor:
    return _extractor
