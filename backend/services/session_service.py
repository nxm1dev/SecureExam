"""
backend/services/session_service.py
────────────────────────────────────
Business logic for managing exam sessions.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.session import Session
from api.schemas import SessionCreate


class SessionService:
    """CRUD operations for exam sessions."""

    async def create(self, db: AsyncSession, data: SessionCreate) -> Session:
        """Start a new exam session."""
        session = Session(
            user_id=data.user_id,
            exam_url=data.exam_url,
            status="active",
        )
        db.add(session)
        await db.flush()
        await db.refresh(session)
        return session

    async def get(self, db: AsyncSession, session_id: uuid.UUID) -> Session | None:
        result = await db.execute(select(Session).where(Session.id == session_id))
        return result.scalar_one_or_none()

    async def list_by_user(
        self, db: AsyncSession, user_id: uuid.UUID
    ) -> list[Session]:
        result = await db.execute(
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.started_at.desc())
        )
        return list(result.scalars().all())

    async def end_session(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        status: str = "completed",
    ) -> Session | None:
        """Mark a session as ended and update violation counters."""
        session = await self.get(db, session_id)
        if not session:
            return None

        # Import here to avoid circular import
        from services.violation_service import ViolationService
        v_svc = ViolationService()
        counts = await v_svc.count_by_severity(db, session_id)

        await db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(
                status=status,
                ended_at=datetime.now(timezone.utc),
                total_violations=sum(counts.values()),
                critical_count=counts.get("critical", 0),
                high_count=counts.get("high", 0),
                medium_count=counts.get("medium", 0),
                low_count=counts.get("low", 0),
            )
        )
        return await self.get(db, session_id)
