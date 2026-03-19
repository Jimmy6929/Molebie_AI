"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.

Supports two-tier inference: Instant (fast, always-on) and Thinking
(stronger, scale-to-zero). Both use OpenAI-compatible /v1/chat/completions
and work with any open-source model served via MLX, vLLM, TGI, etc.
"""

from functools import lru_cache
from typing import Optional
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
    # Both tiers use OpenAI-compatible APIs via mlx_lm.server
    # and work with ANY MLX-quantized model from mlx-community.
    
    # Instant tier — fast, always-on (mlx_vlm.server for VLMs)
    inference_instant_url: str = ""
    inference_instant_model: str = ""          # (unused — instant tier currently empty)
    inference_instant_api_prefix: str = ""     # "" for mlx_vlm, "/v1" for mlx_lm
    inference_instant_max_tokens: int = 2048
    inference_instant_temperature: float = 0.7
    inference_instant_top_p: float = 0.8
    inference_instant_top_k: int = 20
    inference_instant_timeout: float = 120.0   # seconds
    inference_instant_enable_thinking: bool = False  # no CoT for fast tier
    
    # Thinking tier — deeper reasoning (mlx_vlm.server for Qwen 3.5)
    inference_thinking_url: str = ""
    inference_thinking_model: str = ""         # e.g. mlx-community/Qwen3.5-9B-4bit
    inference_thinking_api_prefix: str = ""    # "" for mlx_vlm, "/v1" for mlx_lm
    inference_thinking_max_tokens: int = 4096
    inference_thinking_temperature: float = 0.6  # Qwen 3.5 recommended for thinking
    inference_thinking_top_p: float = 0.95
    inference_thinking_top_k: int = 20
    inference_thinking_timeout: float = 300.0    # 5 min — cold start + reasoning
    inference_thinking_enable_thinking: bool = True  # CoT reasoning for deep tier
    inference_thinking_budget: int = 2048            # max tokens for the <think> block
    inference_thinking_harder_max_tokens: int = 28672
    inference_thinking_harder_budget: int = 8192
    
    # Legacy / shared fallback (used when per-mode settings are empty)
    inference_model_name: str = "default"
    inference_max_tokens: int = 2048
    inference_temperature: float = 0.7
    inference_timeout: float = 120.0
    inference_stream: bool = True
    
    # ── Routing ────────────────────────────────────────────────
    routing_default_mode: str = "thinking"
    routing_thinking_fallback_to_instant: bool = True  # Fallback if thinking is down
    routing_cold_start_timeout: float = 60.0           # Max wait for serverless cold start
    
    # ── Cost Controls ─────────────────────────────────────────
    thinking_daily_request_limit: int = 100   # Max thinking requests per user per day
    thinking_max_concurrent: int = 2          # Max concurrent thinking requests
    
    # ── Kokoro TTS (local text-to-speech) ─────────────────────
    kokoro_tts_url: str = "http://localhost:8880"
    kokoro_tts_default_voice: str = "bm_george"

    # ── SearXNG Web Search ─────────────────────────────────────
    searxng_url: str = "http://localhost:8888"
    web_search_enabled: bool = True
    web_search_timeout: float = 5.0
    web_search_max_results: int = 6
    web_search_snippet_max_chars: int = 800

    # ── RAG / Embeddings ─────────────────────────────────────
    rag_enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims, ~80MB, fast load
    embedding_local_only: bool = False  # Use cached model only (6–7x faster load); set True after first run
    embedding_preload: bool = False  # Load embedding model at startup (trade startup time for faster first chat)
    rag_match_count: int = 5
    rag_match_threshold: float = 0.4
    rag_max_context_chars: int = 4000
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50

    # Database URL (for direct connections if needed)
    database_url: str = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
    
    # ── Helpers ────────────────────────────────────────────────
    
    def get_api_prefix_for_mode(self, mode: str) -> str:
        """Return the API path prefix for the given mode ('/v1' or '')."""
        if mode in ("thinking", "thinking_harder"):
            return self.inference_thinking_api_prefix
        return self.inference_instant_api_prefix
    
    def get_model_for_mode(self, mode: str) -> str:
        """Return the model name for the given mode, with fallback."""
        if mode in ("thinking", "thinking_harder"):
            return self.inference_thinking_model or self.inference_model_name
        return self.inference_instant_model or self.inference_model_name
    
    def get_max_tokens_for_mode(self, mode: str) -> int:
        """Return max tokens for the given mode."""
        if mode == "thinking_harder":
            return self.inference_thinking_harder_max_tokens
        if mode == "thinking":
            return self.inference_thinking_max_tokens
        return self.inference_instant_max_tokens
    
    def get_temperature_for_mode(self, mode: str) -> float:
        """Return temperature for the given mode."""
        if mode in ("thinking", "thinking_harder"):
            return self.inference_thinking_temperature
        return self.inference_instant_temperature
    
    def get_timeout_for_mode(self, mode: str) -> float:
        """Return timeout for the given mode."""
        if mode in ("thinking", "thinking_harder"):
            return self.inference_thinking_timeout
        return self.inference_instant_timeout

    def get_top_p_for_mode(self, mode: str) -> float:
        """Return top_p for the given mode."""
        if mode in ("thinking", "thinking_harder"):
            return self.inference_thinking_top_p
        return self.inference_instant_top_p

    def get_top_k_for_mode(self, mode: str) -> int:
        """Return top_k for the given mode."""
        if mode in ("thinking", "thinking_harder"):
            return self.inference_thinking_top_k
        return self.inference_instant_top_k
    
    def get_enable_thinking_for_mode(self, mode: str) -> bool:
        """Return whether to enable chain-of-thought for the given mode."""
        if mode in ("thinking", "thinking_harder"):
            return self.inference_thinking_enable_thinking
        return self.inference_instant_enable_thinking
    
    def get_thinking_budget_for_mode(self, mode: str) -> Optional[int]:
        """Return the thinking token budget, or None for non-thinking modes."""
        if mode == "thinking_harder":
            return self.inference_thinking_harder_budget
        if mode == "thinking":
            return self.inference_thinking_budget
        return None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
