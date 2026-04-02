"""Pydantic models for CLI configuration persistence."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class SetupType(str, Enum):
    SINGLE = "single"
    DISTRIBUTED = "distributed"


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

    model_config = {"protected_namespaces": ()}

    version: int = Field(default=3, description="Schema version for future migrations")

    # Setup type
    setup_type: SetupType = SetupType.SINGLE

    # Per-service: does it run on THIS machine?
    run_inference: bool = True
    run_gateway: bool = True
    run_webapp: bool = True

    # Per-service: remote host (only meaningful when run_* is False)
    inference_host: str = "localhost"
    gateway_host: str = "localhost"
    webapp_host: str = "localhost"

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

    @model_validator(mode="after")
    def _at_least_one_local_service(self) -> "MolebieConfig":
        if self.setup_type == SetupType.DISTRIBUTED:
            if not any([self.run_inference, self.run_gateway, self.run_webapp]):
                raise ValueError("At least one service must run on this machine")
        return self
