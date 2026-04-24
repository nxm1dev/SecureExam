"""
backend/main.py
FastAPI application entry point.
Mounts all routers and sets up CORS, middlewares, and startup hooks.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import reports, sessions, users, violations
from core.config import get_settings
from core.database import Base, engine
from core.logger import get_logger, setup_logging
from models import session, user, violation  # noqa: F401

settings = get_settings()
setup_logging(settings.log_level)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: ensure the database schema exists.
    Shutdown: dispose DB engine pool.
    """
    log.info("SecureExam Backend starting", port=8000)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()
    log.info("SecureExam Backend stopped")


app = FastAPI(
    title="SecureExam Backend",
    description="Anti-fraud exam session management API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(sessions.router)
app.include_router(violations.router)
app.include_router(reports.router)


@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint for Docker and desktop app."""
    return {"status": "ok", "service": "backend"}
