"""
Document management endpoints for RAG.

Handles file upload, processing, listing, and deletion.
Files are stored in Supabase Storage; metadata and chunks in Postgres.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.middleware.auth import JWTPayload, get_current_user
from app.models.documents import (
    AttachResponse,
    DocumentInfo,
    DocumentListResponse,
    SessionAttachmentInfo,
    SessionAttachmentListResponse,
    UploadResponse,
)
from app.services.document_processor import extract_text
from app.services.database import DatabaseService, get_database_service
from app.services.document_processor import DocumentProcessor, get_document_processor

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_TYPES: Dict[str, str] = {
    "text/plain": "txt",
    "text/markdown": "md",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ── Supabase Storage helpers ─────────────────────────────────────────────

def _storage_headers(settings: Settings) -> dict:
    """Headers for Supabase Storage API calls using service role."""
    return {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
    }


def _upload_to_storage(
    settings: Settings,
    user_id: str,
    filename: str,
    data: bytes,
    content_type: str,
) -> str:
    """Upload file to Supabase Storage. Returns the storage path."""
    storage_path = f"{user_id}/{uuid.uuid4().hex}_{filename}"
    url = f"{settings.supabase_url}/storage/v1/object/documents/{storage_path}"

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            url,
            content=data,
            headers={
                **_storage_headers(settings),
                "Content-Type": content_type,
            },
        )
        if not resp.is_success:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Storage upload failed: {resp.status_code} {resp.text[:300]}",
            )
    return storage_path


def _download_from_storage(settings: Settings, storage_path: str) -> bytes:
    """Download file bytes from Supabase Storage."""
    url = f"{settings.supabase_url}/storage/v1/object/documents/{storage_path}"
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(url, headers=_storage_headers(settings))
        resp.raise_for_status()
        return resp.content


def _delete_from_storage(settings: Settings, storage_path: str) -> None:
    """Delete a file from Supabase Storage."""
    url = f"{settings.supabase_url}/storage/v1/object/documents/{storage_path}"
    with httpx.Client(timeout=30.0) as client:
        client.delete(url, headers=_storage_headers(settings))


# ── Document DB helpers (service-role for status updates) ────────────────

def _db_headers_service(settings: Settings) -> dict:
    return {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _insert_document(settings: Settings, doc: dict) -> dict:
    url = f"{settings.supabase_url}/rest/v1/documents"
    with httpx.Client() as client:
        resp = client.post(url, json=doc, headers=_db_headers_service(settings))
        resp.raise_for_status()
        return resp.json()[0]


def _update_document_status(
    settings: Settings, doc_id: str, doc_status: str, processed_at: str = None
):
    url = f"{settings.supabase_url}/rest/v1/documents?id=eq.{doc_id}"
    body: dict = {"status": doc_status}
    if processed_at:
        body["processed_at"] = processed_at
    with httpx.Client() as client:
        resp = client.patch(url, json=body, headers=_db_headers_service(settings))
        resp.raise_for_status()


def _insert_chunks(settings: Settings, chunks: List[dict]):
    url = f"{settings.supabase_url}/rest/v1/document_chunks"
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=chunks, headers=_db_headers_service(settings))
        resp.raise_for_status()


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user: JWTPayload = Depends(get_current_user),
    processor: DocumentProcessor = Depends(get_document_processor),
):
    """
    Upload a document for RAG processing.

    Accepts TXT, MD, PDF, DOCX (up to 50 MB). The file is stored in
    Supabase Storage, then text is extracted, chunked, embedded, and
    inserted into document_chunks for vector search.
    """
    settings = get_settings()
    user_id = user.user_id

    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed: {', '.join(ALLOWED_TYPES.values())}",
        )

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(data)} bytes). Max: {MAX_FILE_SIZE} bytes",
        )
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    filename = file.filename or "document"

    # 1. Upload to storage
    storage_path = _upload_to_storage(settings, user_id, filename, data, content_type)

    # 2. Create document row (pending)
    doc_row = _insert_document(settings, {
        "user_id": user_id,
        "filename": filename,
        "storage_path": storage_path,
        "file_type": content_type,
        "file_size": len(data),
        "status": "pending",
    })
    doc_id = doc_row["id"]

    # 3. Process: extract → chunk → embed
    try:
        _update_document_status(settings, doc_id, "processing")
        print(f"[documents] Processing {filename} ({len(data)} bytes, {content_type})")
        chunk_pairs = processor.process(data, content_type)
        print(f"[documents] Processing complete: {len(chunk_pairs)} chunk+embedding pairs")
    except Exception as exc:
        print(f"[documents] Processing FAILED: {type(exc).__name__}: {exc}")
        _update_document_status(settings, doc_id, "failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Processing failed: {exc}",
        )

    # 4. Insert chunks with embeddings (batched to avoid oversized payloads)
    try:
        chunk_rows = [
            {
                "document_id": doc_id,
                "user_id": user_id,
                "content": text,
                "embedding": embedding,
                "chunk_index": idx,
            }
            for idx, (text, embedding) in enumerate(chunk_pairs)
        ]
        batch_size = 20
        for i in range(0, len(chunk_rows), batch_size):
            batch = chunk_rows[i : i + batch_size]
            _insert_chunks(settings, batch)
            print(f"[documents] Inserted chunks {i}–{i + len(batch) - 1}")
    except Exception as exc:
        print(f"[documents] Chunk insert FAILED: {type(exc).__name__}: {exc}")
        _update_document_status(settings, doc_id, "failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store chunks: {exc}",
        )

    # 5. Mark completed
    _update_document_status(
        settings,
        doc_id,
        "completed",
        processed_at=datetime.now(timezone.utc).isoformat(),
    )

    return UploadResponse(
        id=doc_id,
        filename=filename,
        status="completed",
        chunks=len(chunk_pairs),
        message=f"Document processed: {len(chunk_pairs)} chunks created",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
):
    """List all documents uploaded by the current user."""
    token = user.raw_token
    settings = get_settings()
    url = (
        f"{settings.supabase_url}/rest/v1/documents"
        f"?user_id=eq.{user.user_id}&order=created_at.desc"
    )
    with httpx.Client() as client:
        resp = client.get(url, headers={
            "apikey": settings.supabase_anon_key,
            "Authorization": f"Bearer {token}",
        })
        resp.raise_for_status()
        rows = resp.json()

    return DocumentListResponse(
        documents=[
            DocumentInfo(
                id=r["id"],
                filename=r["filename"],
                file_type=r["file_type"],
                file_size=r["file_size"],
                status=r["status"],
                created_at=r["created_at"],
                processed_at=r.get("processed_at"),
            )
            for r in rows
        ]
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    """Delete a document, its chunks, and the storage file."""
    settings = get_settings()
    token = user.raw_token

    # Fetch document (with user RLS)
    url = (
        f"{settings.supabase_url}/rest/v1/documents"
        f"?id=eq.{document_id}&user_id=eq.{user.user_id}&select=*"
    )
    with httpx.Client() as client:
        resp = client.get(url, headers={
            "apikey": settings.supabase_anon_key,
            "Authorization": f"Bearer {token}",
        })
        resp.raise_for_status()
        rows = resp.json()

    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    storage_path = rows[0]["storage_path"]

    # Delete chunks (service role — cascades from documents FK, but explicit is safer)
    with httpx.Client() as client:
        client.delete(
            f"{settings.supabase_url}/rest/v1/document_chunks?document_id=eq.{document_id}",
            headers=_db_headers_service(settings),
        )

    # Delete document row
    with httpx.Client() as client:
        client.delete(
            f"{settings.supabase_url}/rest/v1/documents?id=eq.{document_id}",
            headers=_db_headers_service(settings),
        )

    # Delete from storage
    try:
        _delete_from_storage(settings, storage_path)
    except Exception:
        pass  # file may already be gone


# ── Session Attachment Endpoints ("Attach to Chat") ──────────────────────


@router.post(
    "/sessions/{session_id}/attach",
    response_model=AttachResponse,
    tags=["Session Attachments"],
)
async def attach_document_to_session(
    session_id: str,
    file: UploadFile = File(...),
    user: JWTPayload = Depends(get_current_user),
):
    """
    Attach a document to a chat session.

    Extracts full text, stores it in session_documents.
    The text is injected into the system prompt so the AI reads everything.
    No embeddings or chunking — just full text.
    """
    settings = get_settings()
    user_id = user.user_id
    token = user.raw_token
    max_chars = settings.session_doc_max_chars

    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed: {', '.join(ALLOWED_TYPES.values())}",
        )

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(data)} bytes). Max: {MAX_FILE_SIZE} bytes",
        )
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    filename = file.filename or "document"

    # Verify the session belongs to this user
    session_url = (
        f"{settings.supabase_url}/rest/v1/chat_sessions"
        f"?id=eq.{session_id}&user_id=eq.{user_id}&select=id"
    )
    with httpx.Client() as client:
        resp = client.get(session_url, headers={
            "apikey": settings.supabase_anon_key,
            "Authorization": f"Bearer {token}",
        })
        resp.raise_for_status()
        if not resp.json():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

    # Extract text
    try:
        text = extract_text(data, content_type)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Text extraction failed: {exc}",
        )

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the document",
        )

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    # Insert into session_documents (use user token for RLS)
    insert_url = f"{settings.supabase_url}/rest/v1/session_documents"
    row = {
        "session_id": session_id,
        "user_id": user_id,
        "filename": filename,
        "content": text,
        "file_size": len(data),
    }
    with httpx.Client() as client:
        resp = client.post(insert_url, json=row, headers={
            "apikey": settings.supabase_anon_key,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        })
        if not resp.is_success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store attachment: {resp.text[:300]}",
            )
        created = resp.json()[0]

    msg = "Document attached to session"
    if truncated:
        msg = (
            f"Document truncated at {max_chars:,} chars. "
            "For larger documents, use 'Add to Memory' for search-based retrieval."
        )

    return AttachResponse(
        id=created["id"],
        filename=filename,
        content_length=len(text),
        session_id=session_id,
        truncated=truncated,
        message=msg,
    )


@router.get(
    "/sessions/{session_id}/attachments",
    response_model=SessionAttachmentListResponse,
    tags=["Session Attachments"],
)
async def list_session_attachments(
    session_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    """List documents attached to a chat session."""
    settings = get_settings()
    token = user.raw_token

    url = (
        f"{settings.supabase_url}/rest/v1/session_documents"
        f"?session_id=eq.{session_id}&user_id=eq.{user.user_id}"
        f"&select=id,session_id,filename,file_size,created_at,content"
        f"&order=created_at.asc"
    )
    with httpx.Client() as client:
        resp = client.get(url, headers={
            "apikey": settings.supabase_anon_key,
            "Authorization": f"Bearer {token}",
        })
        resp.raise_for_status()
        rows = resp.json()

    return SessionAttachmentListResponse(
        attachments=[
            SessionAttachmentInfo(
                id=r["id"],
                session_id=r["session_id"],
                filename=r["filename"],
                content_length=len(r.get("content", "")),
                file_size=r["file_size"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
    )


@router.delete(
    "/sessions/{session_id}/attachments/{attachment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Session Attachments"],
)
async def remove_session_attachment(
    session_id: str,
    attachment_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    """Remove a document attachment from a chat session."""
    settings = get_settings()
    token = user.raw_token

    url = (
        f"{settings.supabase_url}/rest/v1/session_documents"
        f"?id=eq.{attachment_id}&session_id=eq.{session_id}&user_id=eq.{user.user_id}"
    )
    with httpx.Client() as client:
        resp = client.delete(url, headers={
            "apikey": settings.supabase_anon_key,
            "Authorization": f"Bearer {token}",
        })
