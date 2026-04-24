"""
backend/services/violation_service.py
──────────────────────────────────────
Business logic for recording and querying violations.
"""

import uuid
from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.violation import Violation
from api.schemas import ViolationCreate


class ViolationService:

    async def create(self, db: AsyncSession, data: ViolationCreate) -> Violation:
        """Record a single violation event."""
        violation = Violation(
            session_id=data.session_id,
            user_id=data.user_id,
            event_type=data.event_type,
            severity=data.severity,
            event_metadata=data.event_metadata,
        )
        db.add(violation)
        await db.flush()
        await db.refresh(violation)
        return violation

    async def list_by_session(
        self, db: AsyncSession, session_id: uuid.UUID
    ) -> list[Violation]:
        result = await db.execute(
            select(Violation)
            .where(Violation.session_id == session_id)
            .order_by(Violation.occurred_at.asc())
        )
        return list(result.scalars().all())

    async def count_by_severity(
        self, db: AsyncSession, session_id: uuid.UUID
    ) -> dict[str, int]:
        """Return {severity: count} for a session."""
        result = await db.execute(
            select(Violation.severity, func.count(Violation.id))
            .where(Violation.session_id == session_id)
            .group_by(Violation.severity)
        )
        return {row[0]: row[1] for row in result.all()}

    async def count_by_type(
        self, db: AsyncSession, session_id: uuid.UUID
    ) -> list[dict]:
        """Return [{event_type, severity, count}] for report grouping."""
        result = await db.execute(
            select(
                Violation.event_type,
                Violation.severity,
                func.count(Violation.id).label("count"),
            )
            .where(Violation.session_id == session_id)
            .group_by(Violation.event_type, Violation.severity)
            .order_by(func.count(Violation.id).desc())
        )
        return [
            {"event_type": r[0], "severity": r[1], "count": r[2]}
            for r in result.all()
        ]
