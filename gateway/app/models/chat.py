"""
Chat request and response models.

Supports two-tier inference (Instant + Thinking) with any open-source
model. Mode selection, fallback info, and latency are surfaced in
responses so the frontend can display appropriate indicators.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ChatMode(str, Enum):
    """Inference mode for chat."""
    INSTANT = "instant"
    THINKING = "thinking"


class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    message: str = Field(..., min_length=1, max_length=32000, description="User message")
    session_id: Optional[str] = Field(None, description="Existing session ID, or null for new session")
    mode: ChatMode = Field(ChatMode.INSTANT, description="Inference mode: instant (fast) or thinking (deeper reasoning)")


class ChatMessage(BaseModel):
    """A single chat message."""
    id: str
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    mode_used: Optional[str] = None
    model_used: Optional[str] = None
    reasoning_content: Optional[str] = None
    created_at: datetime


class InferenceMetadata(BaseModel):
    """Metadata about the inference call, returned alongside the response."""
    mode_used: str                          # actual mode used (may differ if fallback)
    model: Optional[str] = None             # model name that served the request
    fallback_used: bool = False             # True if thinking fell back to instant
    original_mode: Optional[str] = None     # original requested mode (if fallback)
    latency_ms: Optional[int] = None        # total round-trip time in ms
    tokens_used: Optional[int] = None       # total tokens (prompt + completion)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    session_id: str
    message: ChatMessage
    session_title: Optional[str] = None
    inference: Optional[InferenceMetadata] = None  # rich metadata about the inference call


class SessionInfo(BaseModel):
    """Chat session information."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    is_archived: bool = False


class SessionRenameRequest(BaseModel):
    """Request body for PATCH /chat/sessions/{id}."""
    title: str = Field(..., min_length=1, max_length=200, description="New session title")


class SessionListResponse(BaseModel):
    """Response for listing sessions."""
    sessions: List[SessionInfo]
