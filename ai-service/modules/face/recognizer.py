"""
ai-service/modules/face/recognizer.py
───────────────────────────────────────
Face recognition using InsightFace embeddings.
Computes 512-dim face embedding and compares to a reference embedding
via cosine similarity.

Does NOT require any server-side training. The reference embedding is
pre-computed from the candidate's registration photo and stored in the
backend DB, then passed in each analysis request.
"""

import base64
from typing import Optional

import numpy as np

from core.config import get_camera_config, get_settings
from core.logger import get_logger

log = get_logger(__name__)


class FaceRecognizer:
    """
    Wraps InsightFace recognition model for embedding extraction.
    Uses cosine distance for identity comparison.
    """

    def __init__(self):
        self._model = None
        self._config = get_camera_config()
        thresh = self._config.get("recognition", {}).get("similarity_threshold", 0.4)
        self._threshold: float = thresh  # cosine distance threshold

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import insightface
            from insightface.app import FaceAnalysis

            settings = get_settings()
            log.info("Loading InsightFace recognition model...")
            self._model = FaceAnalysis(
                name="buffalo_l",
                root=settings.model_cache_dir,
                allowed_modules=["detection", "recognition"],
            )
            self._model.prepare(ctx_id=-1, det_size=(640, 640))
            log.info("InsightFace recognition model loaded (buffalo_l)")
        except Exception as e:
            log.error("Failed to load recognition model", error=str(e))
            raise RuntimeError(f"Recognition model load failed: {e}") from e

    def extract_embedding(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-dim face embedding from the largest face in the frame.
        Returns None if no face is detected.
        """
        self._load_model()
        try:
            faces = self._model.get(frame_bgr)
            if not faces:
                return None
            # Use the face with highest detection score
            best = max(faces, key=lambda f: f.det_score)
            return best.embedding  # shape (512,)
        except Exception as e:
            log.warning("Embedding extraction error", error=str(e))
            return None

    def embedding_to_b64(self, embedding: np.ndarray) -> str:
        """Serialize embedding to base64 string for storage in DB."""
        return base64.b64encode(embedding.astype(np.float32).tobytes()).decode()

    def b64_to_embedding(self, b64: str) -> np.ndarray:
        """Deserialize embedding from base64 string."""
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype=np.float32)

    def cosine_distance(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Return cosine distance [0, 2]; lower = more similar."""
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            return 2.0
        similarity = np.dot(emb_a, emb_b) / (norm_a * norm_b)
        return float(1.0 - similarity)

    def is_same_person(
        self, embedding: np.ndarray, reference_b64: str
    ) -> tuple[bool, float]:
        """
        Compare extracted embedding to the reference.

        Returns:
            (is_match: bool, distance: float)
        """
        reference = self.b64_to_embedding(reference_b64)
        distance = self.cosine_distance(embedding, reference)
        return distance <= self._threshold, distance


# Module-level singleton
_recognizer = FaceRecognizer()


def get_recognizer() -> FaceRecognizer:
    return _recognizer
