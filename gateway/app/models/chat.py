"""
Chat request and response models.

Supports two-tier inference (Instant + Thinking) with any open-source
model. Mode selection, fallback info, and latency are surfaced in
responses so the frontend can display appropriate indicators.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMode(str, Enum):
    """Inference mode for chat."""
    INSTANT = "instant"
    THINKING = "thinking"
    THINKING_HARDER = "thinking_harder"


class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    message: str = Field(..., min_length=1, max_length=32000, description="User message")
    session_id: str | None = Field(None, description="Existing session ID, or null for new session")
    mode: ChatMode = Field(ChatMode.INSTANT, description="Inference mode: instant (fast) or thinking (deeper reasoning)")
    conversation_mode: bool = Field(False, description="When true, use voice conversation system prompt")
    image: str | None = Field(None, description="Base64-encoded image as data URI (data:image/...;base64,...)")
    web_search: bool = Field(False, description="When true, force web search for this message")


class ChatMessage(BaseModel):
    """A single chat message."""
    model_config = ConfigDict(serialize_by_alias=True)
    id: str
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    mode_used: str | None = None
    inference_model: str | None = Field(None, alias="model_used")
    reasoning_content: str | None = None
    image_id: str | None = None
    sources: list[dict[str, str]] | None = None
    created_at: datetime


class InferenceMetadata(BaseModel):
    """Metadata about the inference call, returned alongside the response."""
    mode_used: str                          # actual mode used (may differ if fallback)
    model: str | None = None             # model name that served the request
    fallback_used: bool = False             # True if thinking fell back to instant
    original_mode: str | None = None     # original requested mode (if fallback)
    latency_ms: int | None = None        # total round-trip time in ms
    tokens_used: int | None = None       # total tokens (prompt + completion)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    rag_metrics: dict[str, Any] | None = None  # RAG pipeline metrics (timing, scores)


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    session_id: str
    message: ChatMessage
    session_title: str | None = None
    inference: InferenceMetadata | None = None  # rich metadata about the inference call


class SessionInfo(BaseModel):
    """Chat session information."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    is_archived: bool = False
    is_pinned: bool = False


class SessionRenameRequest(BaseModel):
    """Request body for PATCH /chat/sessions/{id}."""
    title: str = Field(..., min_length=1, max_length=200, description="New session title")


class SessionPinRequest(BaseModel):
    """Request body for PATCH /chat/sessions/{id}/pin."""
    is_pinned: bool = Field(..., description="Whether to pin or unpin the session")


class SessionListResponse(BaseModel):
    """Response for listing sessions."""
    sessions: list[SessionInfo]


class TTSRequest(BaseModel):
    """Request body for POST /chat/tts."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: str = Field("bm_george", description="Kokoro voice ID")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
