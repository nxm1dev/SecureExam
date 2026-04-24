"""
ai-service/api/routes/audio.py
───────────────────────────────
Audio analysis endpoints.

POST /analyze/audio          – Analyze an audio chunk
POST /analyze/audio/clear    – Clear session audio state
"""

from fastapi import APIRouter, HTTPException

from api.schemas import AudioAnalyzeRequest, AudioAnalyzeResponse, SessionClearRequest
from core.logger import get_logger
from modules.audio.analyzer import AudioAnalysisResult, get_audio_analyzer

log = get_logger(__name__)
router = APIRouter(prefix="/analyze/audio", tags=["audio"])


@router.post("/", response_model=AudioAnalyzeResponse)
async def analyze_audio(req: AudioAnalyzeRequest) -> AudioAnalyzeResponse:
    """
    Analyze one audio chunk for a given session.
    Maintains per-session MFCC history for change detection.
    """
    try:
        analyzer = get_audio_analyzer()
        result: AudioAnalysisResult = analyzer.analyze(
            pcm_b64=req.pcm_b64,
            session_id=req.session_id,
        )
        return AudioAnalyzeResponse(
            has_speech=result.has_speech,
            speech_ratio=result.speech_ratio,
            speaker_change_detected=result.speaker_change_detected,
            change_distance=result.change_distance,
            rapid_changes_detected=result.rapid_changes_detected,
            rapid_change_count=result.rapid_change_count,
            voice_overlap_detected=result.voice_overlap_detected,
            overlap_score=result.overlap_score,
            suggested_violation=result.suggested_violation,
            suggested_severity=result.suggested_severity,
        )
    except Exception as e:
        log.error("Audio analysis error", error=str(e))
        raise HTTPException(status_code=500, detail="Audio analysis failed")


@router.post("/clear")
async def clear_session_state(req: SessionClearRequest):
    """
    Release in-memory audio state for a session after it ends.
    Call this when ending an exam session.
    """
    get_audio_analyzer().clear_session(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}
