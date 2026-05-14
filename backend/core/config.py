"""
backend/core/config.py
──────────────────────
Loads and validates backend configuration from environment variables
and the shared YAML config files.
"""

import os
from functools import lru_cache
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None
from pydantic_settings import BaseSettings


def _discover_config_dir() -> Path:
    env_dir = os.getenv("EXAMAC_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "config"
        if candidate.exists():
            return candidate

    return current.parents[2] / "config"


CONFIG_DIR = _discover_config_dir()


def _load_yaml(filename: str) -> dict:
    """Load a YAML file from the shared config directory."""
    path = CONFIG_DIR / filename
    if not path.exists() or yaml is None:
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """
    Backend service settings.
    Values come from environment variables first, then defaults below.
    """

    # Database
    database_url: str = (
        "postgresql+asyncpg://postgres.oyfsjrywxxfndcwjyopi:Nguyenminhnhat@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
    )

    # AI service URL
    ai_service_url: str = "https://nx-m1-ea.hf.space"

    # JWT / Auth (minimal for MVP)
    secret_key: str = "change-me-in-production-please"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 8  # 8 hours

    # Logging
    log_level: str = "INFO"

    # CORS – desktop app origin
    cors_origins: list[str] = [
        "http://localhost:3000",
        "app://ExamAC",
        "file://",
        "null",
    ]

    model_config = {"env_file": ".env", "case_sensitive": False}


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_app_config() -> dict:
    """Return parsed app.yaml."""
    return _load_yaml("app.yaml")


def get_whitelist() -> list[str]:
    """Return list of allowed URL patterns from whitelist.yaml."""
    data = _load_yaml("whitelist.yaml")
    return data.get("whitelist", [])
