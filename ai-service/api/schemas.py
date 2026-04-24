"""
ai-service/api/schemas.py
──────────────────────────
Pydantic request/response schemas for the AI service API.
"""

from typing import Any, Optional
from pydantic import BaseModel


# ──────────────────────────────────────────
# Face analysis
# ──────────────────────────────────────────

class FaceAnalyzeRequest(BaseModel):
    """
    Single frame analysis request.

    frame_b64             : Base64-encoded JPEG image.
    reference_embedding_b64: Optional candidate reference embedding (from DB).
                            When provided, identity verification runs automatically.
    extract_embedding     : Force embedding extraction even without a reference.
                            Use True for the candidate registration flow.
    """
    frame_b64: str
    reference_embedding_b64: Optional[str] = None
    extract_embedding: bool = False


class FaceBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class FaceAnalyzeResponse(BaseModel):
    face_count: int
    face_detected: bool
    multiple_faces: bool
    identity_checked: bool
    identity_match: bool
    identity_distance: float
    embedding_b64: Optional[str] = None  # For registration flow
    bboxes: list[FaceBox]


# ──────────────────────────────────────────
# Audio analysis
# ──────────────────────────────────────────

class AudioAnalyzeRequest(BaseModel):
    """
    Audio chunk analysis request.
    pcm_b64: Base64-encoded 16kHz 16-bit mono PCM bytes.
    session_id: Used to maintain state across chunks.
    """
    pcm_b64: str
    session_id: str


class AudioAnalyzeResponse(BaseModel):
    has_speech: bool
    speech_ratio: float
    speaker_change_detected: bool
    change_distance: float
    rapid_changes_detected: bool
    rapid_change_count: int
    voice_overlap_detected: bool
    overlap_score: float
    suggested_violation: str
    suggested_severity: str


class SessionClearRequest(BaseModel):
    session_id: str
