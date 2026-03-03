"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.

Supports two-tier inference: Instant (fast, always-on) and Thinking
(stronger, scale-to-zero). Both use OpenAI-compatible /v1/chat/completions
and work with any open-source model served via vLLM, TGI, Ollama, etc.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local", "../.env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore NEXT_PUBLIC_* and other webapp-only vars
    )
    
    # Application
    app_name: str = "AI Assistant Gateway"
    debug: bool = True
    
    # Supabase Configuration
    supabase_url: str = "http://127.0.0.1:54321"
    supabase_anon_key: str = ""  # Set via SUPABASE_ANON_KEY env var
    supabase_service_role_key: str = ""  # Set via SUPABASE_SERVICE_ROLE_KEY env var
    
    # JWT Configuration (for local Supabase, this is the default secret)
    jwt_secret: str = ""  # Set via JWT_SECRET env var
    jwt_algorithm: str = "HS256"
    
    # ── Inference Endpoints ────────────────────────────────────
    # Both tiers use OpenAI-compatible APIs and work with ANY
    # open-source model (Llama, Mistral, Qwen, DeepSeek, etc.)
    
    # Instant tier — fast, always-on (7B-13B models)
    inference_instant_url: str = ""
    inference_instant_model: str = ""          # e.g. mistralai/Mistral-7B-Instruct-v0.3
    inference_instant_max_tokens: int = 2048
    inference_instant_temperature: float = 0.7
    inference_instant_timeout: float = 120.0   # seconds
    
    # Thinking tier — stronger, scale-to-zero (30B-70B models)
    inference_thinking_url: str = ""
    inference_thinking_model: str = ""         # e.g. Qwen/Qwen2.5-72B-Instruct
    inference_thinking_max_tokens: int = 4096
    inference_thinking_temperature: float = 0.5  # more focused for reasoning
    inference_thinking_timeout: float = 300.0    # 5 min — cold start + reasoning
    
    # Legacy / shared fallback (used when per-mode settings are empty)
    inference_model_name: str = "default"
    inference_max_tokens: int = 2048
    inference_temperature: float = 0.7
    inference_timeout: float = 120.0
    inference_stream: bool = True
    
    # ── Routing ────────────────────────────────────────────────
    routing_default_mode: str = "instant"
    routing_thinking_fallback_to_instant: bool = True  # Fallback if thinking is down
    routing_cold_start_timeout: float = 60.0           # Max wait for serverless cold start
    
    # ── Cost Controls ─────────────────────────────────────────
    thinking_daily_request_limit: int = 100   # Max thinking requests per user per day
    thinking_max_concurrent: int = 2          # Max concurrent thinking requests
    
    # Database URL (for direct connections if needed)
    database_url: str = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
    
    # ── Helpers ────────────────────────────────────────────────
    
    def get_model_for_mode(self, mode: str) -> str:
        """Return the model name for the given mode, with fallback."""
        if mode == "thinking":
            return self.inference_thinking_model or self.inference_model_name
        return self.inference_instant_model or self.inference_model_name
    
    def get_max_tokens_for_mode(self, mode: str) -> int:
        """Return max tokens for the given mode."""
        if mode == "thinking":
            return self.inference_thinking_max_tokens
        return self.inference_instant_max_tokens
    
    def get_temperature_for_mode(self, mode: str) -> float:
        """Return temperature for the given mode."""
        if mode == "thinking":
            return self.inference_thinking_temperature
        return self.inference_instant_temperature
    
    def get_timeout_for_mode(self, mode: str) -> float:
        """Return timeout for the given mode."""
        if mode == "thinking":
            return self.inference_thinking_timeout
        return self.inference_instant_timeout


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
