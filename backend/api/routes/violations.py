"""
backend/api/routes/violations.py
──────────────────────────────────
Violation log endpoints.
POST /violations/              – Record a violation
GET  /violations/{session_id}  – Get all violations for a session
"""

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import ViolationCreate, ViolationOut
from core.database import get_db
from services.violation_service import ViolationService

router = APIRouter(prefix="/violations", tags=["violations"])
_service = ViolationService()


@router.post("/", response_model=ViolationOut, status_code=201)
async def log_violation(data: ViolationCreate, db: AsyncSession = Depends(get_db)):
    """Record a single violation event from the desktop app."""
    violation = await _service.create(db, data)
    return violation


@router.post("/batch", response_model=list[ViolationOut], status_code=201)
async def log_violations_batch(
    items: list[ViolationCreate], db: AsyncSession = Depends(get_db)
):
    """Batch-record multiple violations in one request (reduces HTTP overhead)."""
    results = []
    for item in items:
        v = await _service.create(db, item)
        results.append(v)
    return results


@router.get("/session/{session_id}", response_model=list[ViolationOut])
async def get_session_violations(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    return await _service.list_by_session(db, session_id)
