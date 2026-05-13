"""
ai-service/core/config.py
──────────────────────────
Loads camera and audio configuration from YAML files.
All thresholds come from config/ directory, never hard-coded here.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

def _discover_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "config").exists():
            return parent
    return current.parents[2]


def _discover_config_dir() -> Path:
    env_dir = os.getenv("EXAMAC_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    return _discover_repo_root() / "config"


REPO_ROOT = _discover_repo_root()
CONFIG_DIR = _discover_config_dir()


def _load_yaml(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


class AISettings(BaseSettings):
    """Runtime settings – primarily from env vars."""

    # Where InsightFace downloads model weights
    model_cache_dir: str = os.getenv(
        "MODEL_CACHE_DIR",
        str(REPO_ROOT / "ai-service" / "models_cache"),
    )

    log_level: str = "INFO"

    # CORS origins (desktop app)
    cors_origins: list[str] = [
        "http://localhost:3000",
        "app://ExamAC",
        "file://",
        "null",
    ]

    model_config = {"env_file": ".env", "case_sensitive": False}


@lru_cache
def get_settings() -> AISettings:
    return AISettings()


@lru_cache
def get_camera_config() -> dict:
    return _load_yaml("camera.yaml").get("camera", {})


@lru_cache
def get_audio_config() -> dict:
    return _load_yaml("audio.yaml").get("audio", {})
