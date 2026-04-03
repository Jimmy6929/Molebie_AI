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

    # ── Connectivity graph ───────────────────────────────────
    # Defines which remote services each local service actually
    # needs to connect to.  Used by install review, run, doctor.
    #
    # Rules:
    #   Gateway  → Inference  (forwards LLM requests)
    #   Webapp   → Gateway    (calls the API)
    #   Inference → nothing   (passive — only listens)

    def required_remote_endpoints(self) -> list[tuple[str, str]]:
        """Return ``(name, url)`` pairs for remote services this machine must reach.

        Only includes services that a LOCAL service actually connects to.
        A pure LLM-server machine returns an empty list because it only listens.
        """
        if self.setup_type != SetupType.DISTRIBUTED:
            return []

        endpoints: list[tuple[str, str]] = []

        # Gateway connects to Inference
        if self.run_gateway and not self.run_inference:
            host = self.inference_host
            if self.inference_backend == InferenceBackend.MLX:
                endpoints.append(("MLX Thinking", f"http://{host}:8080/v1/models"))
                endpoints.append(("MLX Instant", f"http://{host}:8081/v1/models"))
            elif self.inference_backend == InferenceBackend.OLLAMA:
                endpoints.append(("Ollama", f"http://{host}:11434/v1/models"))
            elif self.inference_backend == InferenceBackend.OPENAI_COMPATIBLE and self.inference_url:
                endpoints.append(("Inference", f"{self.inference_url}/v1/models"))

        # Webapp connects to Gateway
        if self.run_webapp and not self.run_gateway:
            endpoints.append(("Gateway", f"http://{self.gateway_host}:8000/health"))

        return endpoints

    def relevant_remote_hosts(self) -> list[tuple[str, str]]:
        """Return ``(label, host)`` pairs for remote hosts worth displaying.

        Only includes hosts that a LOCAL service needs to know about —
        i.e. hosts that appear in :meth:`required_remote_endpoints`.
        """
        if self.setup_type != SetupType.DISTRIBUTED:
            return []

        hosts: list[tuple[str, str]] = []

        if self.run_gateway and not self.run_inference:
            hosts.append(("LLM Server host", self.inference_host))
        if self.run_webapp and not self.run_gateway:
            hosts.append(("Gateway host", self.gateway_host))

        return hosts

    def local_service_names(self) -> list[str]:
        """Return human-readable names of services that run on this machine."""
        return [name for name, runs in [
            ("LLM Server", self.run_inference),
            ("Gateway", self.run_gateway),
            ("Webapp", self.run_webapp),
        ] if runs]
