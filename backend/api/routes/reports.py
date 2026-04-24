"""
backend/api/routes/reports.py
───────────────────────────────
Post-exam report endpoints.
GET /reports/{session_id} – Full report for a session
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import ReportOut
from core.database import get_db
from services.report_service import ReportService

router = APIRouter(prefix="/reports", tags=["reports"])
_service = ReportService()


@router.get("/{session_id}", response_model=ReportOut)
async def get_report(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Generate a comprehensive report for a completed session.
    Includes: violation counts, severity breakdown, type grouping, timeline.
    """
    report = await _service.generate(db, session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Session not found")
    return report
