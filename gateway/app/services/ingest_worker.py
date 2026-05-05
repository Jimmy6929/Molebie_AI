"""
Background worker for folder-ingest jobs.

One asyncio.Task per active ingest_jobs row. Picks up files in `uploaded`
state, runs them through the existing extract → chunk → embed pipeline, and
emits SSE events both to live subscriber queues (for connected clients) and
to the `ingest_job_events` table (for `Last-Event-ID` replay on reconnect).

Sequential within a job — the embedder is the bottleneck on local CPU/MPS.
Cancellation is polled between files. Server restart picks up where it left
off via `resume_running_jobs()` in the FastAPI lifespan.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings
from app.services.database import get_database_service
from app.services.document_processor import get_document_processor
from app.services.markdown_meta import extract_md_metadata
from app.services.storage import get_storage_service


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_markdown(content_type: str | None, relative_path: str) -> bool:
    if content_type and content_type.lower() in ("text/markdown", "md"):
        return True
    rp = relative_path.lower()
    return rp.endswith(".md") or rp.endswith(".markdown")


_QUEUE_MAX = 1024


class IngestWorker:
    """Singleton coordinator for folder-ingest background jobs."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._progress_state: dict[str, dict[str, float]] = {}
        self._lock = asyncio.Lock()

    # ---------------------- Subscriber API (used by SSE route) ----------------------

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._subscribers.setdefault(job_id, set()).add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        bucket = self._subscribers.get(job_id)
        if not bucket:
            return
        bucket.discard(q)
        if not bucket:
            self._subscribers.pop(job_id, None)

    # ---------------------- Worker control ----------------------

    async def ensure_worker_started(self, job_id: str) -> None:
        """Start the worker task for `job_id` if not already running."""
        async with self._lock:
            existing = self._tasks.get(job_id)
            if existing and not existing.done():
                return
            if existing and existing.done():
                self._tasks.pop(job_id, None)
            task = asyncio.create_task(self._run_job_safe(job_id), name=f"ingest:{job_id}")
            self._tasks[job_id] = task

    async def resume_running_jobs(self) -> None:
        """Restart workers for jobs left in pending/running state across restart."""
        try:
            db = get_database_service()
            jobs = await db.list_unfinished_ingest_jobs()
            for job in jobs:
                await db.reset_processing_files_to_uploaded(job["id"])
                await self.ensure_worker_started(job["id"])
            if jobs:
                print(f"[ingest_worker] Resumed {len(jobs)} unfinished job(s)")
        except Exception as exc:  # noqa: BLE001
            print(f"[ingest_worker] resume failed: {type(exc).__name__}: {exc}")

    # ---------------------- Event emission ----------------------

    async def emit(self, job_id: str, event_type: str, payload: dict[str, Any]) -> int:
        """Persist an event and fan out to live subscribers. Returns the event id."""
        db = get_database_service()
        event_id = await db.insert_ingest_event(job_id, event_type, payload)
        envelope = {"id": event_id, "event_type": event_type, "payload": payload}
        for q in list(self._subscribers.get(job_id, ())):
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                # Drop oldest, push new — clients fall back to replay on reconnect
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(envelope)
                except asyncio.QueueFull:
                    pass
        return event_id

    async def _maybe_emit_progress(self, job_id: str) -> None:
        settings = get_settings()
        state = self._progress_state.setdefault(
            job_id, {"last_files": 0.0, "last_time": 0.0}
        )
        db = get_database_service()
        snap = await db.get_ingest_job(job_id)
        if not snap:
            return
        done = snap["processed_files"] + snap["failed_files"] + snap["skipped_files"]
        now = time.monotonic()
        if (
            done - state["last_files"] >= settings.folder_ingest_progress_interval_files
            or now - state["last_time"] >= settings.folder_ingest_progress_interval_sec
        ):
            state["last_files"] = float(done)
            state["last_time"] = now
            await self.emit(
                job_id,
                "progress",
                {
                    "processed_files": snap["processed_files"],
                    "failed_files": snap["failed_files"],
                    "skipped_files": snap["skipped_files"],
                    "processed_bytes": snap["processed_bytes"],
                    "total_files": snap["total_files"],
                    "total_bytes": snap["total_bytes"],
                },
            )

    # ---------------------- Job loop ----------------------

    async def _run_job_safe(self, job_id: str) -> None:
        try:
            await self._run_job(job_id)
        except Exception as exc:  # noqa: BLE001
            db = get_database_service()
            try:
                await db.set_ingest_job_finished(
                    job_id, "failed", error_message=f"{type(exc).__name__}: {exc}"
                )
                await self.emit(job_id, "job_failed", {"error": str(exc)})
            except Exception:
                pass
            print(f"[ingest_worker] job {job_id} crashed: {type(exc).__name__}: {exc}")
        finally:
            self._tasks.pop(job_id, None)
            self._progress_state.pop(job_id, None)

    async def _run_job(self, job_id: str) -> None:
        db = get_database_service()
        job = await db.get_ingest_job(job_id)
        if not job:
            return
        if job["status"] == "cancelled":
            await self.emit(
                job_id,
                "job_cancelled",
                {
                    "job_id": job_id,
                    "processed_files": job["processed_files"],
                    "failed_files": job["failed_files"],
                },
            )
            return

        await db.set_ingest_job_running(job_id)
        started_mono = time.monotonic()
        await self.emit(
            job_id,
            "job_started",
            {
                "job_id": job_id,
                "total_files": job["total_files"],
                "total_bytes": job["total_bytes"],
                "root_label": job["root_label"],
            },
        )

        while True:
            if await db.is_ingest_job_cancelled(job_id):
                snap = await db.get_ingest_job(job_id)
                await self.emit(
                    job_id,
                    "job_cancelled",
                    {
                        "job_id": job_id,
                        "processed_files": snap["processed_files"] if snap else 0,
                        "failed_files": snap["failed_files"] if snap else 0,
                    },
                )
                return
            file_row = await db.next_pending_ingest_file(job_id)
            if file_row is None:
                # No more files in 'uploaded' state — but the client may still be
                # uploading. Bail; subsequent /upload calls will re-`ensure_worker_started`.
                break
            await self._process_one_file(job_id, file_row)
            await self._maybe_emit_progress(job_id)

        # Job is done — only when no pending or uploaded files remain.
        snap = await db.get_ingest_job(job_id)
        if snap is None:
            return

        # Re-check there are truly no `pending`/`uploaded` rows left. If the client
        # is still trickling uploads, exit silently — a later upload will respawn.
        any_unfinished = await self._has_unfinished_files(job_id)
        if any_unfinished:
            return

        duration_ms = int((time.monotonic() - started_mono) * 1000)
        await db.set_ingest_job_finished(job_id, "completed")
        await self.emit(
            job_id,
            "job_completed",
            {
                "processed_files": snap["processed_files"],
                "failed_files": snap["failed_files"],
                "skipped_files": snap["skipped_files"],
                "total_bytes": snap["total_bytes"],
                "duration_ms": duration_ms,
            },
        )

    async def _has_unfinished_files(self, job_id: str) -> bool:
        db = get_database_service()
        conn = await db._get_conn()
        rows = await conn.execute_fetchall(
            "SELECT 1 FROM ingest_job_files "
            "WHERE job_id = ? AND status IN ('pending','uploaded','processing') LIMIT 1",
            (job_id,),
        )
        return bool(rows)

    # ---------------------- Per-file ----------------------

    async def _process_one_file(
        self, job_id: str, file_row: dict[str, Any]
    ) -> None:
        db = get_database_service()
        storage = get_storage_service()
        processor = get_document_processor()
        settings = get_settings()

        file_id = file_row["id"]
        rel = file_row["relative_path"]
        size = int(file_row["file_size"] or 0)
        user_id = file_row["user_id"]
        storage_path = file_row.get("storage_path")
        content_type = file_row.get("content_type") or _infer_content_type(rel)

        await db.update_ingest_file_status(
            file_id, "processing", started_at=_now_iso()
        )
        await self.emit(
            job_id,
            "file_started",
            {"file_id": file_id, "relative_path": rel, "size": size},
        )

        if not storage_path:
            await db.update_ingest_file_status(
                file_id, "skipped",
                error_message="no_bytes",
                finished_at=_now_iso(),
            )
            await db.bump_ingest_job_counts(job_id, skipped=1)
            await self.emit(
                job_id,
                "file_skipped",
                {"file_id": file_id, "relative_path": rel, "reason": "no_bytes"},
            )
            return

        try:
            data = await asyncio.to_thread(storage.download_document, storage_path)

            doc_row = await db.insert_document(
                user_id=user_id,
                filename=_basename(rel),
                storage_path=storage_path,
                file_type=content_type,
                file_size=size,
                doc_status="processing",
                relative_path=rel,
                file_hash=file_row.get("file_hash"),
                ingest_job_id=job_id,
            )
            doc_id = doc_row["id"]

            if settings.rag_contextual_retrieval_enabled:
                triples = await processor.process_async(data, content_type)
            else:
                triples = await asyncio.to_thread(processor.process, data, content_type)

            md_meta: dict[str, list[str]] | None = None
            if _is_markdown(content_type, rel):
                try:
                    md_meta = extract_md_metadata(data.decode("utf-8", errors="replace"))
                except Exception:  # noqa: BLE001
                    md_meta = None

            chunk_rows: list[dict[str, Any]] = []
            for text, embedding, meta in triples:
                chunk_meta: dict[str, Any] = {
                    "heading": meta.get("heading"),
                    "relative_path": rel,
                }
                if md_meta:
                    if md_meta["tags"]:
                        chunk_meta["tags"] = md_meta["tags"]
                    if md_meta["wikilinks"]:
                        chunk_meta["wikilinks"] = md_meta["wikilinks"]
                row = {
                    "document_id": doc_id,
                    "user_id": user_id,
                    "content": text,
                    "embedding": embedding,
                    "chunk_index": meta.get("chunk_index", 0),
                    "metadata": chunk_meta,
                }
                if meta.get("content_contextualized"):
                    row["content_contextualized"] = meta["content_contextualized"]
                chunk_rows.append(row)

            for i in range(0, len(chunk_rows), 20):
                await db.insert_chunks(chunk_rows[i:i + 20])

            await db.update_document_status(doc_id, "completed", processed_at=_now_iso())
            await db.update_ingest_file_status(
                file_id, "completed",
                document_id=doc_id,
                finished_at=_now_iso(),
            )
            await db.bump_ingest_job_counts(job_id, processed=1, processed_bytes=size)
            await self.emit(
                job_id,
                "file_completed",
                {
                    "file_id": file_id,
                    "relative_path": rel,
                    "chunks": len(chunk_rows),
                    "document_id": doc_id,
                    "size": size,
                },
            )
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            print(f"[ingest_worker] file {rel} failed: {err}")
            await db.update_ingest_file_status(
                file_id, "failed",
                error_message=err,
                finished_at=_now_iso(),
            )
            await db.bump_ingest_job_counts(job_id, failed=1)
            await self.emit(
                job_id,
                "file_failed",
                {"file_id": file_id, "relative_path": rel, "error": err},
            )


def _basename(rel: str) -> str:
    return rel.rsplit("/", 1)[-1] or rel


_EXT_TO_MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".css": "text/css",
}


def _infer_content_type(rel: str) -> str:
    lower = rel.lower()
    dot = lower.rfind(".")
    if dot >= 0:
        ext = lower[dot:]
        if ext in _EXT_TO_MIME:
            return _EXT_TO_MIME[ext]
        return ext.lstrip(".") or "text/plain"
    return "text/plain"


_worker: IngestWorker | None = None


def get_ingest_worker() -> IngestWorker:
    global _worker
    if _worker is None:
        _worker = IngestWorker()
    return _worker
