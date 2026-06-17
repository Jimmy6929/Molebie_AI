"""
Pydantic models for the documents / RAG endpoints.
"""

from datetime import datetime

from pydantic import BaseModel, Field


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


class BrainInfo(BaseModel):
    """A user-defined brain: a named bucket of vault folders."""
    id: str
    name: str
    folders: list[str] = []
    doc_count: int = 0
    # Folders the brain references that currently hold no documents (e.g. the
    # folder was renamed/deleted in the vault) — surfaced so the user re-points.
    missing_folders: list[str] = []


class BrainListResponse(BaseModel):
    """Response for GET /documents/brains."""
    brains: list[BrainInfo]


class FolderInfo(BaseModel):
    """A top-level vault folder available to add to a brain."""
    folder: str
    doc_count: int


class FolderListResponse(BaseModel):
    """Response for GET /documents/folders (the brain folder-picker source)."""
    folders: list[FolderInfo]


class CreateBrainRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class RenameBrainRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class BrainFolderRequest(BaseModel):
    folder: str = Field(..., min_length=1)


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
