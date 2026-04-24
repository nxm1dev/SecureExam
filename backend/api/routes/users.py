"""
backend/api/routes/users.py
────────────────────────────
User management endpoints.
POST /users/            – Register candidate
GET  /users/{id}        – Get user info
PUT  /users/{id}/face   – Upload face embedding for recognition
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import FaceEmbeddingUpdate, UserCreate, UserOut
from core.database import get_db
from models.user import User

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/", response_model=UserOut, status_code=201)
async def create_user(data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user (candidate or admin)."""
    # Check email uniqueness
    existing = await db.execute(select(User).where(User.email == data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(**data.model_dump())
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


@router.get("/by-email/{email}", response_model=UserOut)
async def get_user_by_email(email: str, db: AsyncSession = Depends(get_db)):
    """Look up a user by email address (used as fallback when registration returns 409)."""
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/{user_id}", response_model=UserOut)
async def get_user(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/{user_id}/face", response_model=UserOut)
async def update_face_embedding(
    user_id: uuid.UUID,
    data: FaceEmbeddingUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Store a candidate's face embedding after registration photo upload."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.face_embedding = data.face_embedding
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user
