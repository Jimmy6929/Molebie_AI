"""
Folder-ingest endpoints.

Lets a user drop an entire folder into Brain. The frontend posts a manifest,
streams up file bytes in batches, then watches a single SSE stream for
per-file progress. A background worker (`ingest_worker`) handles the
extract → chunk → embed pipeline sequentially per job, with state persisted
to SQLite so the job survives a server restart.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.middleware.auth import (
    JWTPayload,
    get_current_user,
    get_current_user_query_or_header,
)
from app.models.folder_ingest import (
    AcceptedFile,
    CancelFolderJobResponse,
    FolderJobSnapshot,
    RejectedFile,
    StartFolderJobRequest,
    StartFolderJobResponse,
    UploadFolderBatchResponse,
)
from app.services.database import get_database_service
from app.services.ingest_worker import get_ingest_worker
from app.services.storage import get_storage_service

router = APIRouter(prefix="/documents/folder", tags=["Folder Ingest"])


# Accepted extensions (must mirror webapp/src/lib/folderManifest.ts)
_ALLOWED_EXTENSIONS = frozenset({
    "txt", "md", "markdown",
    "pdf", "docx",
    "py", "js", "ts", "tsx", "jsx",
    "json", "yaml", "yml", "toml",
    "html", "htm", "css", "sql", "sh",
    "go", "rs", "java", "c", "cpp", "h", "hpp",
    "rb", "php", "csv",
    "ini", "cfg", "conf",
})


def _split_globs(s: str) -> tuple[list[str], list[str]]:
    """Return (segment_names, globs) parsed from a comma-separated ignore string.

    A bare token like `node_modules` matches an exact path segment.
    A token containing `*` (e.g. `*.lock`) is treated as a basename glob.
    """
    names: list[str] = []
    globs: list[str] = []
    for raw in s.split(","):
        token = raw.strip()
        if not token:
            continue
        if "*" in token or "?" in token:
            globs.append(token)
        else:
            names.append(token)
    return names, globs


def _path_is_ignored(rel: str, names: list[str], globs: list[str]) -> bool:
    segments = rel.split("/")
    name_set = set(names)
    if any(s in name_set for s in segments):
        return True
    base = segments[-1] if segments else rel
    return any(fnmatch.fnmatch(base, g) for g in globs)


def _ext_of(rel: str) -> str:
    dot = rel.rfind(".")
    return rel[dot + 1:].lower() if dot >= 0 else ""


def _classify(rel: str, size: int, max_size: int, names: list[str], globs: list[str]) -> str | None:
    """Return None if accepted, otherwise a rejection reason string."""
    if _path_is_ignored(rel, names, globs):
        return "ignored"
    if size > max_size:
        return "too_large"
    ext = _ext_of(rel)
    if not ext or ext not in _ALLOWED_EXTENSIONS:
        return "unsupported_type"
    return None


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post("/start", response_model=StartFolderJobResponse)
async def start_folder_job(
    body: StartFolderJobRequest,
    user: JWTPayload = Depends(get_current_user),
):
    settings = get_settings()
    if not settings.folder_ingest_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Folder ingest disabled",
        )
    if not settings.rag_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG must be enabled to ingest folders",
        )

    db = get_database_service()
    user_id = user.user_id

    # Refuse if there's already a non-terminal job for this user.
    active = await db.get_active_ingest_job(user_id)
    if active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "A folder ingest is already in progress",
                "job_id": active["id"],
                "status": active["status"],
            },
        )

    if not body.files:
        raise HTTPException(status_code=400, detail="Manifest is empty")
    if len(body.files) > settings.folder_ingest_max_files:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files (limit {settings.folder_ingest_max_files})",
        )

    names, globs = _split_globs(settings.folder_ingest_default_ignore)
    max_size = settings.document_max_file_size

    rejected: list[RejectedFile] = []
    accepted_for_db: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for entry in body.files:
        rel = entry.relative_path.strip().lstrip("/")
        if not rel:
            rejected.append(RejectedFile(relative_path=entry.relative_path, reason="empty_path"))
            continue
        if ".." in rel.split("/"):
            rejected.append(RejectedFile(relative_path=rel, reason="path_traversal"))
            continue
        if rel in seen_paths:
            rejected.append(RejectedFile(relative_path=rel, reason="duplicate_path"))
            continue
        reason = _classify(rel, entry.size, max_size, names, globs)
        if reason:
            rejected.append(RejectedFile(relative_path=rel, reason=reason))
            continue
        seen_paths.add(rel)
        accepted_for_db.append({
            "relative_path": rel,
            "file_size": entry.size,
            "content_type": entry.content_type,
        })

    if not accepted_for_db:
        raise HTTPException(
            status_code=400,
            detail={"message": "No files accepted", "rejected": [r.model_dump() for r in rejected]},
        )

    job = await db.create_ingest_job(user_id, body.root_label, accepted_for_db)
    accepted = [
        AcceptedFile(file_id=a["file_id"], relative_path=a["relative_path"], size=a["size"])
        for a in job["accepted_files"]
    ]

    return StartFolderJobResponse(
        job_id=job["id"],
        accepted_files=accepted,
        rejected_files=rejected,
        total_accepted_bytes=sum(a.size for a in accepted),
    )


@router.post("/{job_id}/upload", response_model=UploadFolderBatchResponse)
async def upload_folder_batch(
    job_id: str,
    files: list[UploadFile] = File(...),
    relative_paths: list[str] = Form(...),
    user: JWTPayload = Depends(get_current_user),
):
    if len(files) != len(relative_paths):
        raise HTTPException(
            status_code=400,
            detail=f"Mismatched files ({len(files)}) vs relative_paths ({len(relative_paths)})",
        )

    db = get_database_service()
    storage = get_storage_service()
    settings = get_settings()
    user_id = user.user_id

    job = await db.get_ingest_job(job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")

    accepted_ids: list[str] = []
    rejected: list[RejectedFile] = []

    for upload, raw_rel in zip(files, relative_paths, strict=False):
        rel = raw_rel.strip().lstrip("/")
        file_row = await db.get_ingest_file_by_path(job_id, rel)
        if not file_row:
            rejected.append(RejectedFile(relative_path=rel, reason="not_in_manifest"))
            continue
        if file_row["status"] != "pending":
            # Already received or in flight — idempotent skip
            accepted_ids.append(file_row["id"])
            continue

        data = await upload.read()
        if not data:
            await db.update_ingest_file_status(
                file_row["id"], "failed", error_message="empty_file"
            )
            rejected.append(RejectedFile(relative_path=rel, reason="empty_file"))
            continue
        if len(data) > settings.document_max_file_size:
            await db.update_ingest_file_status(
                file_row["id"], "failed", error_message="too_large"
            )
            rejected.append(RejectedFile(relative_path=rel, reason="too_large"))
            continue

        sha = hashlib.sha256(data).hexdigest()
        filename = rel.rsplit("/", 1)[-1] or rel
        content_type = upload.content_type or file_row.get("content_type") or "text/plain"
        storage_path = await asyncio.to_thread(
            storage.upload_document, user_id, filename, data, content_type
        )
        await db.update_ingest_file_status(
            file_row["id"],
            "uploaded",
            storage_path=storage_path,
            file_hash=sha,
            content_type=content_type,
        )
        accepted_ids.append(file_row["id"])

    # Wake the worker (idempotent — safe even if already running).
    if accepted_ids:
        await get_ingest_worker().ensure_worker_started(job_id)

    return UploadFolderBatchResponse(accepted=accepted_ids, rejected=rejected)


@router.post("/{job_id}/cancel", response_model=CancelFolderJobResponse)
async def cancel_folder_job(
    job_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    db = get_database_service()
    cancelled = await db.set_ingest_job_cancelled(job_id, user.user_id)
    if not cancelled:
        job = await db.get_ingest_job(job_id, user.user_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return CancelFolderJobResponse(job_id=job_id, status=job["status"])

    # Nudge the worker / SSE listeners.
    snap = await db.get_ingest_job(job_id, user.user_id)
    if snap:
        await get_ingest_worker().emit(
            job_id,
            "job_cancelled",
            {
                "job_id": job_id,
                "processed_files": snap["processed_files"],
                "failed_files": snap["failed_files"],
            },
        )
    return CancelFolderJobResponse(job_id=job_id, status="cancelled")


def _to_snapshot(job: dict[str, Any], last_event_id: int) -> FolderJobSnapshot:
    return FolderJobSnapshot(
        job_id=job["id"],
        status=job["status"],
        root_label=job["root_label"],
        total_files=job["total_files"],
        processed_files=job["processed_files"],
        failed_files=job["failed_files"],
        skipped_files=job["skipped_files"],
        total_bytes=job["total_bytes"],
        processed_bytes=job["processed_bytes"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        last_event_id=last_event_id,
    )


@router.get("/active", response_model=FolderJobSnapshot | None)
async def get_active_folder_job(
    user: JWTPayload = Depends(get_current_user),
):
    db = get_database_service()
    job = await db.get_active_ingest_job(user.user_id)
    if not job:
        return None
    last_id = await db.get_ingest_max_event_id(job["id"])
    return _to_snapshot(job, last_id)


@router.get("/{job_id}", response_model=FolderJobSnapshot)
async def get_folder_job(
    job_id: str,
    user: JWTPayload = Depends(get_current_user),
):
    db = get_database_service()
    job = await db.get_ingest_job(job_id, user.user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    last_id = await db.get_ingest_max_event_id(job_id)
    return _to_snapshot(job, last_id)


_TERMINAL_EVENTS = {"job_completed", "job_failed", "job_cancelled"}


def _format_sse(event_id: int, event_type: str, payload: Any) -> str:
    """Render one SSE frame. Payload may already be a JSON string (from DB) or a dict."""
    if isinstance(payload, str):
        data_str = payload
    else:
        data_str = json.dumps(payload)
    return f"id: {event_id}\nevent: {event_type}\ndata: {data_str}\n\n"


@router.get("/{job_id}/events")
async def stream_folder_job_events(
    job_id: str,
    request: Request,
    user: JWTPayload = Depends(get_current_user_query_or_header),
):
    db = get_database_service()
    job = await db.get_ingest_job(job_id, user.user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Last-Event-ID can come from header (auto-reconnect) or query (initial reattach).
    raw = (
        request.headers.get("last-event-id")
        or request.query_params.get("last_event_id")
        or "0"
    )
    try:
        last_event_id = int(raw)
    except ValueError:
        last_event_id = 0

    worker = get_ingest_worker()

    async def gen():
        # 1. Replay events the client may have missed.
        events = await db.list_ingest_events_after(job_id, last_event_id)
        terminal_seen = False
        for e in events:
            yield _format_sse(e["id"], e["event_type"], e["payload"])
            if e["event_type"] in _TERMINAL_EVENTS:
                terminal_seen = True

        # 2. If the job is already finished, end the stream.
        if terminal_seen or job["status"] in ("completed", "failed", "cancelled"):
            return

        # 3. Subscribe live; intersperse heartbeats every 15s.
        queue = worker.subscribe(job_id)
        try:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield _format_sse(evt["id"], evt["event_type"], evt["payload"])
                if evt["event_type"] in _TERMINAL_EVENTS:
                    return
        finally:
            worker.unsubscribe(job_id, queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
