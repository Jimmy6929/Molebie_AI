"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.

Supports two-tier inference: Instant (fast, always-on) and Thinking
(stronger, scale-to-zero). Both use OpenAI-compatible /v1/chat/completions
and work with any open-source model served via MLX, vLLM, TGI, etc.
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
    # Qwen3.5 non-thinking preset: presence_penalty=1.5 reduces repetition
    # without the language-mixing seen above 1.5. repetition_penalty=1.0 is
    # neutral — Qwen ships tuned, so over-penalising hurts coherence.
    inference_instant_presence_penalty: float = 1.5
    inference_instant_repetition_penalty: float = 1.0
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
    # Thinking-mode preset: penalties at 0 — repetition control during a CoT
    # block tends to truncate reasoning. Repetition control belongs on the
    # final answer, which the strict-grounding template already constrains.
    inference_thinking_presence_penalty: float = 0.0
    inference_thinking_repetition_penalty: float = 1.0
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

    # Hard ceiling on presence_penalty: above 1.5 Qwen3.5 starts mixing
    # languages mid-response. Applied as a clamp in the helper, not a
    # validator, so misconfigured .env files don't crash the service.
    inference_max_presence_penalty: float = 1.5

    # Which OpenAI-compatible backend the inference URLs point at. Controls
    # only the on-the-wire field name for repetition penalty:
    #   * "auto" / "mlx" / "vllm" / "ollama" → ``repetition_penalty``
    #   * "llamacpp"                          → ``repeat_penalty``
    # Default "auto" sends both names so a tuned value is honoured by either
    # server family. Set explicitly when running a llama.cpp-only deployment
    # to keep the wire payload minimal.
    inference_backend_flavor: str = "auto"

    # When True, thinking-tier requests with RAG context disable CoT —
    # 9B "burns thousands of thinking tokens in circles" on retrieval Q&A.
    # See task 1.3 in tasks/hallucination-mitigation/phase-1-foundation.md.
    inference_thinking_auto_disable_for_rag: bool = True

    # ── Self-Consistency (Phase 2 task 2.3) ───────────────────────────
    # Sample N responses for verifiable queries (factual/numeric/yes-no)
    # and majority-vote. Reduces hallucinations on the queries small models
    # are most likely to fabricate on.
    self_consistency_enabled: bool = False
    self_consistency_samples: int = 5         # N total samples
    self_consistency_early_stop: int = 3      # ESC: stop after this many agree
    self_consistency_max_concurrent: int = 3  # cap parallel inference calls

    # ── Tool Calling (Phase 2 task 2.2) ────────────────────────────────
    # Off by default: tool calling needs a backend that actually supports
    # OpenAI-format tools (vLLM with --tool-call-parser hermes, Ollama via
    # /v1, llama.cpp with --jinja). mlx_vlm support is patchy — verify
    # against your backend before flipping this on.
    tool_calling_enabled: bool = False
    tool_calling_max_iterations: int = 4   # cap the tool-call → execute → call loop

    # ── Chain-of-Verification (Phase 3 task 3.1) ──────────────────────
    # Post-generation: decompose response into atomic claims, verify each
    # claim against the cited chunk in a separate inference context, flag
    # unsupported claims inline. Factored variant — verifier doesn't see
    # the original generation context.
    # Off by default: 5–9× extra inference calls on applicable responses.
    cove_enabled: bool = False
    cove_min_response_chars: int = 500       # only verify long responses
    cove_max_claims: int = 8                 # cap per-response claim count
    cove_verifier_max_concurrent: int = 4    # bounded concurrency on verifier
    cove_verifier_temperature: float = 0.0   # deterministic verification

    # ── Grounding Judge (Phase 3 task 3.2) ────────────────────────────
    # Reuses Qwen3-Reranker yes/no head as a fast grounding gate. No
    # extra LLM calls — one reranker forward pass per claim. Catches
    # off-topic fabrications cheaply; CoVe still needed for numeric
    # substitution attacks within an otherwise-relevant chunk.
    judge_enabled: bool = False
    judge_threshold: float = 0.5             # below = flagged
    judge_min_response_chars: int = 200

    # ── SelfCheckGPT (Phase 3 task 3.3) ───────────────────────────────
    # Reference-free consistency: sample N additional responses at
    # higher temperature, score sentences by cross-sample disagreement
    # using NLI (DeBERTa-v3-MNLI when selfcheckgpt is installed; pure-
    # function token-overlap fallback otherwise). Fills the no-RAG case
    # CoVe and the judge can't reach.
    selfcheck_enabled: bool = False
    selfcheck_samples: int = 3
    selfcheck_temperature: float = 0.7
    selfcheck_threshold: float = 0.5         # NLI: contradiction probability
    selfcheck_min_response_chars: int = 200
    selfcheck_max_concurrent: int = 3

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
    # Same model family as the generator (Qwen) → tokenizer alignment →
    # better retrieval. 1024-dim, 32K context, instruction-aware. Outputs
    # match the schema's vec table dim (must be kept in sync — see
    # ``embedding_dim`` and scripts/migrate_embedding_dim.py).
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    # Dim of the embedding model's output. Used to create the sqlite-vec
    # virtual tables. If this changes, run scripts/migrate_embedding_dim.py
    # then POST /documents/reindex/full.
    embedding_dim: int = 1024
    embedding_local_only: bool = True  # Use cached model only; avoids HuggingFace Hub check (offline-safe)
    embedding_preload: bool = False  # Load embedding model at startup (trade startup time for faster first chat)
    # Bumped from 20: we over-fetch more candidates so the new reranker
    # has more material to work with; the rerank top-k decides what
    # actually reaches the prompt.
    rag_match_count: int = 30
    rag_match_threshold: float = 0.3
    rag_max_context_chars: int = 12000
    # Smaller chunks → less noise per retrieval hit. 512/64 is the
    # build-plan target; the previous 1024/128 produced overly broad
    # chunks that diluted reranker scores. After changing these, run
    # POST /documents/reindex/full to rebuild existing chunks.
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64

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
    # Qwen3-Reranker: same family as the generator, beats BGE-reranker-v2-m3
    # on MTEB-R/CMTEB/MTEB-Code, 32K ctx, Apache 2.0. Loaded via the
    # AutoModelForCausalLM yes/no-logit path inside RerankerService;
    # CrossEncoder-style models still work via the same service.
    rag_reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    # Bumped from 5: more chunks reach the prompt for better coverage now
    # that the reranker is stronger and chunks are smaller (512 vs 1024).
    rag_rerank_top_k: int = 8
    rag_reranker_preload: bool = False

    # ── RAG Metrics ───────────────────────────────────────────
    rag_metrics_enabled: bool = True
    rag_metrics_log_console: bool = True

    # ── Documents ─────────────────────────────────────────────────
    document_max_file_size: int = 50 * 1024 * 1024  # 50 MB; override via DOCUMENT_MAX_FILE_SIZE
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

    def get_presence_penalty_for_mode(self, mode: str) -> float:
        """Return presence_penalty for the given mode, clamped at the
        Qwen-safe ceiling (above 1.5 causes language mixing)."""
        raw = (
            self.inference_thinking_presence_penalty
            if mode in ("thinking", "thinking_harder")
            else self.inference_instant_presence_penalty
        )
        return min(raw, self.inference_max_presence_penalty)

    def get_repetition_penalty_for_mode(self, mode: str) -> float:
        """Return repetition_penalty for the given mode."""
        if mode in ("thinking", "thinking_harder"):
            return self.inference_thinking_repetition_penalty
        return self.inference_instant_repetition_penalty

    def get_enable_thinking_for_mode(self, mode: str) -> bool:
        """Return whether to enable chain-of-thought for the given mode."""
        if mode in ("thinking", "thinking_harder"):
            return self.inference_thinking_enable_thinking
        return self.inference_instant_enable_thinking

    def get_thinking_budget_for_mode(self, mode: str) -> int | None:
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
