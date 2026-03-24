"""Pydantic models for CLI configuration persistence."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SetupType(str, Enum):
    SINGLE = "single"
    TWO_MACHINE = "two-machine"


class InferenceBackend(str, Enum):
    MLX = "mlx"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai-compatible"


class ModelProfile(str, Enum):
    LIGHT = "light"
    BALANCED = "balanced"
    CUSTOM = "custom"


VALID_FEATURES = ("voice", "search", "rag")

FEATURE_DESCRIPTIONS = {
    "voice": "Voice conversation (Whisper STT + Kokoro TTS) — requires Docker + ffmpeg",
    "search": "Web search (SearXNG) — requires Docker",
    "rag": "Document memory (RAG) — downloads embedding model on first use",
}


class MolebieConfig(BaseModel):
    """Persistent configuration saved by the install wizard."""

    version: int = Field(default=2, description="Schema version for future migrations")

    # Setup type
    setup_type: SetupType = SetupType.SINGLE
    gpu_ip: str = "localhost"
    server_ip: str = "localhost"

    # Inference backend
    inference_backend: InferenceBackend = InferenceBackend.MLX
    inference_url: Optional[str] = Field(
        default=None, description="Custom URL for openai-compatible backend"
    )
    inference_api_key: Optional[str] = Field(
        default=None, description="API key for commercial backends"
    )
    thinking_model: Optional[str] = Field(
        default=None, description="Override thinking tier model name"
    )
    instant_model: Optional[str] = Field(
        default=None, description="Override instant tier model name"
    )

    # Model profile
    model_profile: ModelProfile = ModelProfile.BALANCED

    # Features
    voice_enabled: bool = False
    search_enabled: bool = True
    rag_enabled: bool = True

    # State
    installed: bool = False
    last_install_at: Optional[str] = Field(
        default=None, description="ISO timestamp of last install"
    )
