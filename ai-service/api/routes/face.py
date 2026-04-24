"""
ai-service/api/routes/face.py
──────────────────────────────
Face analysis endpoints.

POST /analyze/face          – Surveillance analysis (detection + optional recognition)
POST /analyze/face/register – Extract embedding from registration photo
"""

from fastapi import APIRouter, HTTPException

from api.schemas import FaceAnalyzeRequest, FaceAnalyzeResponse, FaceBox
from core.logger import get_logger
from modules.face.analyzer import FaceAnalysisResult, get_analyzer

log = get_logger(__name__)
router = APIRouter(prefix="/analyze/face", tags=["face"])


@router.post("/", response_model=FaceAnalyzeResponse)
async def analyze_face(req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
    """
    Analyze a single camera frame.

    Detection always runs (ExamFaceDetector, detection-only model).
    Recognition runs only when:
      - ``reference_embedding_b64`` is provided  → identity verification
      - ``extract_embedding`` is True             → registration flow
    """
    try:
        analyzer = get_analyzer()
        result: FaceAnalysisResult = analyzer.analyze(
            frame_b64=req.frame_b64,
            reference_embedding_b64=req.reference_embedding_b64,
            extract_embedding=req.extract_embedding,
        )
        return FaceAnalyzeResponse(
            face_count=result.face_count,
            face_detected=result.face_detected,
            multiple_faces=result.multiple_faces,
            identity_checked=result.identity_checked,
            identity_match=result.identity_match,
            identity_distance=result.identity_distance,
            embedding_b64=result.embedding_b64,
            bboxes=[
                FaceBox(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"])
                for b in result.bboxes
            ],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error("Face analysis error", error=str(e))
        raise HTTPException(status_code=500, detail="Face analysis failed")
