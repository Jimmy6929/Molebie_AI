"""
Pydantic models for the documents / RAG endpoints.
"""

from datetime import datetime

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    """Metadata about an uploaded document."""
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    created_at: datetime
    processed_at: datetime | None = None


class DocumentListResponse(BaseModel):
    """Response for GET /documents."""
    documents: list[DocumentInfo]


class UploadResponse(BaseModel):
    """Response for POST /documents/upload."""
    id: str
    filename: str
    status: str
    chunks: int = 0
    message: str = "Document uploaded and processed"


class SessionAttachmentInfo(BaseModel):
    """Metadata about a document attached to a chat session."""
    id: str
    session_id: str
    filename: str
    content_length: int
    file_size: int
    truncated: bool = False
    created_at: datetime


class AttachResponse(BaseModel):
    """Response for POST /chat/sessions/{session_id}/attach."""
    id: str
    filename: str
    content_length: int
    session_id: str
    truncated: bool = False
    message: str = "Document attached to session"


class SessionAttachmentListResponse(BaseModel):
    """Response for GET /chat/sessions/{session_id}/attachments."""
    attachments: list[SessionAttachmentInfo]
