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
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # ── Assistant Identity ──────────────────────────────────────
    assistant_name: str = "Assistant"
    prompt_dir: str = "prompts"  # relative to gateway/ directory

    # ── Inference API Key (for commercial backends like OpenAI) ──
    inference_api_key: str = ""
    
    # Data directory (SQLite database + local file storage)
    data_dir: str = "data"

    # Authentication mode: "single" (password only) or "multi" (email + password)
    auth_mode: str = "single"

    # JWT Configuration
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
    inference_thinking_model: str = ""         # e.g. mlx-community/Qwen3.5-9B-MLX-4bit
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
    web_search_fetch_full_content: bool = True
    web_search_full_content_count: int = 3
    web_search_full_content_max_chars: int = 2000
    web_search_full_content_timeout: float = 4.0

    # ── Intent Classification ────────────────────────────────
    web_search_llm_classify: bool = True
    web_search_classify_timeout: float = 3.0
    web_search_classify_max_tokens: int = 3

    # ── RAG / Embeddings ─────────────────────────────────────
    rag_enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # default; override via EMBEDDING_MODEL in .env
    embedding_local_only: bool = False  # Use cached model only (6–7x faster load); set True after first run
    embedding_preload: bool = False  # Load embedding model at startup (trade startup time for faster first chat)
    rag_match_count: int = 20  # over-fetch for reranking
    rag_match_threshold: float = 0.3
    rag_max_context_chars: int = 12000
    rag_chunk_size: int = 1024
    rag_chunk_overlap: int = 128

    # ── Hybrid Search ─────────────────────────────────────────
    rag_hybrid_enabled: bool = True
    rag_vector_weight: float = 0.7
    rag_text_weight: float = 0.3
    rag_rrf_k: int = 60

    # ── Contextual Retrieval ──────────────────────────────────
    rag_contextual_retrieval_enabled: bool = True
    rag_context_max_doc_chars: int = 8000  # max doc chars sent to LLM for context generation
    rag_context_llm_mode: str = "instant"  # LLM tier for context generation
    rag_context_max_tokens: int = 150  # max tokens per context prefix

    # ── Cross-Encoder Reranking ───────────────────────────────
    rag_reranker_enabled: bool = True
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rag_rerank_top_k: int = 5
    rag_reranker_preload: bool = False

    # ── RAG Metrics ───────────────────────────────────────────
    rag_metrics_enabled: bool = True
    rag_metrics_log_console: bool = True

    # ── Session Document Attachments ──────────────────────────────
    session_doc_max_chars: int = 12000  # max extracted text per attachment

    # ── Conversation Summarisation (8a) ─────────────────────────
    summary_enabled: bool = True
    summary_trigger_threshold: int = 16    # unsummarised messages before compressing
    summary_recent_messages: int = 10      # recent messages kept raw (not summarised)
    summary_max_input_chars: int = 8000    # max chars of messages sent to summariser LLM
    summary_max_output_tokens: int = 300   # max tokens for LLM summary response
    summary_llm_mode: str = "instant"      # LLM tier for summarisation

    # ── Structured Memory (8b) ──────────────────────────────────
    memory_enabled: bool = True
    memory_extract_interval: int = 6       # extract facts every N messages
    memory_max_facts_per_extraction: int = 5
    memory_dedup_threshold: float = 0.9    # cosine sim above this = duplicate
    memory_retrieval_threshold: float = 0.5
    memory_retrieval_top_k: int = 5
    memory_max_per_user: int = 200         # max stored memories per user
    memory_llm_mode: str = "instant"       # LLM tier for extraction
    memory_extract_max_tokens: int = 400   # max tokens for extraction response

    # ── RAG Query Rewriting (8c) ────────────────────────────────
    rag_query_rewrite_enabled: bool = True
    rag_query_rewrite_timeout: float = 3.0  # hard timeout; fallback to original
    rag_query_rewrite_max_tokens: int = 50  # short rewritten query
    rag_query_rewrite_llm_mode: str = "instant"

    # ── Vision / Image ──────────────────────────────────────────
    vision_max_image_size: int = 5 * 1024 * 1024  # 5 MB max base64 payload
    vision_allowed_types: str = "image/jpeg,image/png,image/gif,image/webp"

    # Legacy (kept for backwards-compat env files, unused)
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    database_url: str = ""
    
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
