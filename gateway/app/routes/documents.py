"""
Document management endpoints for RAG.

Handles file upload, processing, listing, and deletion.
Files are stored locally; metadata and chunks in SQLite.
"""

from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config import get_settings
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
from app.services.storage import LocalStorageService, get_storage_service
from app.services.document_processor import DocumentProcessor, get_document_processor

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_TYPES: Dict[str, str] = {
    "text/plain": "txt",
    "text/markdown": "md",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user: JWTPayload = Depends(get_current_user),
    processor: DocumentProcessor = Depends(get_document_processor),
):
    """
    Upload a document for RAG processing.

    Accepts TXT, MD, PDF, DOCX (up to 50 MB). The file is stored locally,
    then text is extracted, chunked, embedded, and inserted into
    document_chunks for vector search.
    """
    settings = get_settings()
    if not settings.rag_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document upload requires RAG to be enabled. Set RAG_ENABLED=true in .env.local and restart.",
        )
    storage = get_storage_service()
    db = get_database_service()
    user_id = user.user_id

    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed: {', '.join(ALLOWED_TYPES.values())}",
        )

    data = await file.read()
    max_size = settings.document_max_file_size
    if len(data) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(data)} bytes). Max: {max_size} bytes",
        )
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    filename = file.filename or "document"

    # 1. Upload to local storage
    storage_path = storage.upload_document(user_id, filename, data, content_type)

    # 2. Create document row (pending)
    doc_row = await db.insert_document(user_id, filename, storage_path, content_type, len(data))
    doc_id = doc_row["id"]

    # 3. Process: extract → chunk → embed (with optional contextual retrieval)
    try:
        await db.update_document_status(doc_id, "processing")
        print(f"[documents] Processing {filename} ({len(data)} bytes, {content_type})")

        if settings.rag_contextual_retrieval_enabled:
            chunk_triples = await processor.process_async(data, content_type)
        else:
            chunk_triples = processor.process(data, content_type)

        print(f"[documents] Processing complete: {len(chunk_triples)} chunk+embedding pairs")
    except Exception as exc:
        print(f"[documents] Processing FAILED: {type(exc).__name__}: {exc}")
        await db.update_document_status(doc_id, "failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Processing failed: {exc}",
        )

    # 4. Insert chunks with embeddings (batched to avoid oversized payloads)
    try:
        chunk_rows = []
        for text, embedding, meta in chunk_triples:
            row = {
                "document_id": doc_id,
                "user_id": user_id,
                "content": text,
                "embedding": embedding,
                "chunk_index": meta.get("chunk_index", 0),
                "metadata": {"heading": meta.get("heading")},
            }
            # Store contextualized version separately if available
            if meta.get("content_contextualized"):
                row["content_contextualized"] = meta["content_contextualized"]
            chunk_rows.append(row)

        batch_size = 20
        for i in range(0, len(chunk_rows), batch_size):
            batch = chunk_rows[i : i + batch_size]
            await db.insert_chunks(batch)
            print(f"[documents] Inserted chunks {i}–{i + len(batch) - 1}")
    except Exception as exc:
        print(f"[documents] Chunk insert FAILED: {type(exc).__name__}: {exc}")
        await db.update_document_status(doc_id, "failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store chunks: {exc}",
        )

    # 5. Mark completed
    await db.update_document_status(
        doc_id,
        "completed",
        processed_at=datetime.now(timezone.utc).isoformat(),
    )

    return UploadResponse(
        id=doc_id,
        filename=filename,
        status="completed",
        chunks=len(chunk_triples),
        message=f"Document processed: {len(chunk_triples)} chunks created",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
):
    """List all documents uploaded by the current user."""
    rows = await db.list_documents(user.user_id)

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
    db = get_database_service()
    storage = get_storage_service()

    doc = await db.get_document(document_id, user.user_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    storage_path = doc["storage_path"]

    # Delete document row and associated chunks from DB
    await db.delete_document(document_id, user.user_id)

    # Delete from local storage
    storage.delete_document(storage_path)


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
    db = get_database_service()
    user_id = user.user_id
    max_chars = settings.session_doc_max_chars

    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed: {', '.join(ALLOWED_TYPES.values())}",
        )

    data = await file.read()
    max_size = settings.document_max_file_size
    if len(data) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(data)} bytes). Max: {max_size} bytes",
        )
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    filename = file.filename or "document"

    # Verify the session belongs to this user
    session = await db.get_session(session_id, user_id)
    if not session:
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

    # Insert into session_documents
    created = await db.insert_session_document(session_id, user_id, filename, text, len(data))

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
    db = get_database_service()
    rows = await db.list_session_documents(session_id, user.user_id)

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
    db = get_database_service()
    await db.delete_session_document(attachment_id, session_id, user.user_id)


# ── RAG Evaluation Endpoint ──────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import List as TypingList


class EvalTestCase(BaseModel):
    query: str
    expected_doc_ids: TypingList[str] = Field(default_factory=list)


class EvalRequest(BaseModel):
    test_cases: TypingList[EvalTestCase]


@router.post("/evaluate", tags=["RAG Evaluation"])
async def evaluate_rag(
    request: EvalRequest,
    user: JWTPayload = Depends(get_current_user),
):
    """
    Evaluate RAG pipeline quality with test cases.

    Each test case provides a query and expected document IDs.
    Returns hit rate, MRR, and per-query details including timing.
    """
    from app.services.rag_eval import evaluate_queries

    cases = [{"query": tc.query, "expected_doc_ids": tc.expected_doc_ids} for tc in request.test_cases]
    results = await evaluate_queries(cases, user.user_id)
    return results
