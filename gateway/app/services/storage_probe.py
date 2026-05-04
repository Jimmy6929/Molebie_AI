"""Storage probe — DB size + table row counts cached for the live monitor.

Mirrors `SystemProbe`'s lifecycle (background task, idempotent start/stop,
last-snapshot cache). Polls every 30s — much slower than the request-path
probes because the underlying values change on document upload / chat
turn cadence, not at request rate.

The `/metrics/live` route reads the cache; it never triggers a poll
itself, so request latency is unaffected by SQLite count queries.

Designed to fail soft: any I/O or DB error sets a `note` field on the
snapshot and leaves prior values intact, rather than raising. The monitor
serving chat traffic is more important than live storage stats.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import aiosqlite

# Slow poll — DB row counts and file size shift on upload / chat
# completion timescales, not on per-request latency. 30s keeps the SQL
# cost negligible while still feeling fresh.
_POLL_INTERVAL_SEC = 30.0


@dataclass(slots=True)
class StorageSnapshot:
    ts: float
    db_path: str | None = None
    db_size_bytes: int | None = None
    documents_count: int | None = None
    chunks_count: int | None = None
    sessions_count: int | None = None
    messages_count: int | None = None
    memories_count: int | None = None
    note: str | None = None


class StorageProbe:
    """Cache the latest storage snapshot for /metrics/live to read."""

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self._latest = StorageSnapshot(ts=time.time())
        self._task: asyncio.Task | None = None
        self._stopping = False

    def latest(self) -> StorageSnapshot:
        return self._latest

    async def start(self, interval: float = _POLL_INTERVAL_SEC) -> None:
        """Idempotent — kicks off the background poll loop."""
        if self._task is not None:
            return
        # Run one poll immediately so /metrics/live has data on first request.
        try:
            self._latest = await self._poll_once()
        except Exception as exc:
            self._latest = StorageSnapshot(ts=time.time(), note=f"probe error: {exc}")
        self._task = asyncio.create_task(self._run(interval))

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run(self, interval: float) -> None:
        # Heartbeat into the Subsystems panel — see system_probe for rationale.
        from app.services.metrics_registry import get_metrics_registry
        registry = get_metrics_registry()
        while not self._stopping:
            t0 = time.monotonic()
            ok = True
            try:
                self._latest = await self._poll_once()
            except Exception as exc:
                ok = False
                self._latest = StorageSnapshot(ts=time.time(), note=f"probe error: {exc}")
            try:
                size = self._latest.db_size_bytes
                note = f"db {size//1024} KB" if size else None
                await registry.record_subsystem(
                    "probe.storage",
                    (time.monotonic() - t0) * 1000.0,
                    ok=ok, note=note,
                )
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def _poll_once(self) -> StorageSnapshot:
        db_path = os.path.join(self._data_dir, "molebie.db")
        snap = StorageSnapshot(ts=time.time(), db_path=db_path)

        # File size — cheap; missing file → None, not error.
        try:
            snap.db_size_bytes = os.path.getsize(db_path)
        except OSError:
            snap.note = "db file not found"
            return snap

        # Row counts via dedicated short-lived connection. SQLite COUNT(*)
        # on indexed tables is fast; on FTS / vec virtual tables it's still
        # microseconds. We accept eventual consistency — the monitor doesn't
        # need transactional guarantees.
        try:
            async with aiosqlite.connect(db_path) as db:
                snap.documents_count = await _count(db, "documents")
                snap.chunks_count = await _count(db, "document_chunks")
                snap.sessions_count = await _count(db, "chat_sessions")
                snap.messages_count = await _count(db, "chat_messages")
                snap.memories_count = await _count(db, "user_memories")
        except Exception as exc:
            snap.note = f"db query error: {exc}"
        return snap


async def _count(db: aiosqlite.Connection, table: str) -> int | None:
    """COUNT(*) helper — returns None if the table doesn't exist."""
    try:
        async with db.execute(f"SELECT COUNT(*) FROM {table}") as cur:
            row = await cur.fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return None


_probe: StorageProbe | None = None


def get_storage_probe(data_dir: str | None = None) -> StorageProbe:
    """Process-wide probe singleton. First caller must pass data_dir."""
    global _probe
    if _probe is None:
        if data_dir is None:
            raise RuntimeError("StorageProbe not yet initialised; pass data_dir on first call")
        _probe = StorageProbe(data_dir)
    return _probe


def reset_storage_probe() -> None:
    """Test-only helper."""
    global _probe
    _probe = None
