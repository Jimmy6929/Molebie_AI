"""Background storage mover — migrates local documents to satellites.

This is the engine that makes the local-first design *visible*: writes
always land on the primary's local disk (TieredStorageService, slice
9.3), and this mover later relocates cold files to a storage satellite,
rewriting their ``documents.storage_path`` from ``local://…`` to
``satellite://<node>/<sha256>`` and reclaiming the local disk.

Slice 9.4a ships the engine + a manual trigger (``POST
/fleet/storage/migrate``). The automatic scheduler that decides *when*
and *which* files to move is slice 9.4b — it will call this same engine.

Failure-safe ordering is the heart of the design. For each document:

    read local → hash → pick satellite → PUT → HEAD-verify
        → record (satellite_blobs + documents + audit, one txn)
        → delete local

The local file is deleted ONLY after the satellite copy is verified
present. Any failure before that step leaves the document fully
readable on the primary — there is no window where a document exists
nowhere.

Sync by design (sync httpx + sync sqlite3), matching the
``TieredStorageService`` precedent — v0.2 scale doesn't justify
propagating async through the migration path.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from app.schema import _get_db_path
from app.services import storage_uri
from app.services.storage import LocalStorageService

_SATELLITE_STORAGE_PORT = 8090
_HTTP_TIMEOUT_SEC = 30.0  # generous — blobs can be large
_CAPACITY_TIMEOUT_SEC = 5.0


@dataclass
class MigrationResult:
    """Outcome of attempting to migrate one document."""

    doc_id: str
    migrated: bool
    satellite_node_id: str | None = None
    sha256: str | None = None
    reason: str | None = None  # why skipped / failed (for response + audit)


class StorageMover:
    """Migrates locally-stored documents to storage satellites."""

    def __init__(
        self,
        local: LocalStorageService,
        operator_identity: str,
        data_dir: str,
        http_client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self.local = local
        self.operator_identity = operator_identity
        self.data_dir = data_dir
        self._client_factory = http_client_factory or httpx.Client

    # ----- public API -----

    def migrate_documents(self, limit: int) -> list[MigrationResult]:
        """Migrate up to ``limit`` locally-stored documents, oldest first."""
        results: list[MigrationResult] = []
        for doc in self._eligible_local_documents(limit):
            results.append(self._migrate_one(doc))
        return results

    # ----- per-document migration (failure-safe sequence) -----

    def _migrate_one(self, doc: dict) -> MigrationResult:
        doc_id = doc["id"]
        storage_path = doc["storage_path"]

        if storage_uri.parse(storage_path).scheme != "local":
            return MigrationResult(doc_id, migrated=False, reason="already_remote")

        target = self._pick_target_satellite()
        if target is None:
            return MigrationResult(doc_id, migrated=False, reason="no_satellite")
        node_id, host = target

        data = self.local.download_document(storage_path)
        digest = hashlib.sha256(data).hexdigest()
        headers = {"Tailscale-User-Login": self.operator_identity}
        blob_url = f"http://{host}:{_SATELLITE_STORAGE_PORT}/v1/storage/blobs/{digest}"

        # PUT the bytes.
        try:
            with self._client_factory() as client:
                put_resp = client.put(
                    blob_url, content=data, headers=headers, timeout=_HTTP_TIMEOUT_SEC
                )
        except (httpx.TimeoutException, httpx.TransportError):
            return MigrationResult(doc_id, migrated=False, reason="upload_failed")
        if put_resp.status_code >= 400:
            return MigrationResult(doc_id, migrated=False, reason="upload_failed")

        # Verify the satellite actually holds it, at the right size.
        try:
            with self._client_factory() as client:
                head_resp = client.head(
                    blob_url, headers=headers, timeout=_HTTP_TIMEOUT_SEC
                )
        except (httpx.TimeoutException, httpx.TransportError):
            return MigrationResult(doc_id, migrated=False, reason="verify_failed")
        if head_resp.status_code != 200:
            return MigrationResult(doc_id, migrated=False, reason="verify_failed")
        content_length = head_resp.headers.get("content-length")
        if content_length is None or int(content_length) != len(data):
            return MigrationResult(doc_id, migrated=False, reason="verify_failed")

        # Record (one transaction): inventory + path rewrite + audit.
        now = datetime.now(timezone.utc).isoformat()
        new_path = storage_uri.build_satellite(node_id, digest)
        self._record_migration(
            doc_id=doc_id,
            digest=digest,
            node_id=node_id,
            size=len(data),
            new_path=new_path,
            now=now,
        )

        # Reclaim local disk — only now that the satellite copy is verified.
        self.local.delete_document(storage_path)

        return MigrationResult(
            doc_id, migrated=True, satellite_node_id=node_id, sha256=digest
        )

    # ----- helpers -----

    def _record_migration(
        self,
        *,
        doc_id: str,
        digest: str,
        node_id: str,
        size: int,
        new_path: str,
        now: str,
    ) -> None:
        """Insert the satellite_blobs row, rewrite storage_path, and log the
        audit event — atomically in one sync transaction. (Sync because the
        mover is sync; mirrors audit_events' columns from PR #46.)"""
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            # UPSERT preserves last_verified_at on re-migration of an already-
            # verified blob; INSERT OR REPLACE deletes-then-inserts and would
            # null the stamp. Slice 9.5 starts writing last_verified_at, so
            # the column needs to survive a same-digest, same-node re-run.
            conn.execute(
                "INSERT INTO satellite_blobs "
                "(sha256, satellite_node_id, blob_type, size_bytes, uploaded_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(sha256, satellite_node_id) DO UPDATE SET "
                "blob_type=excluded.blob_type, "
                "size_bytes=excluded.size_bytes, "
                "uploaded_at=excluded.uploaded_at",
                (digest, node_id, "document", size, now),
            )
            conn.execute(
                "UPDATE documents SET storage_path = ? WHERE id = ?",
                (new_path, doc_id),
            )
            conn.execute(
                "INSERT INTO audit_events "
                "(event_type, actor, target, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "storage.migrate",
                    "system",
                    doc_id,
                    _json_metadata(node_id, digest, size),
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _eligible_local_documents(self, limit: int) -> list[dict]:
        """Completed documents still stored locally, oldest first."""
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, storage_path, file_size FROM documents "
                "WHERE status = 'completed' ORDER BY created_at ASC"
            ).fetchall()
        finally:
            conn.close()
        eligible: list[dict] = []
        for row in rows:
            if storage_uri.parse(row["storage_path"]).scheme == "local":
                eligible.append(dict(row))
                if len(eligible) >= limit:
                    break
        return eligible

    def _active_storage_satellites(self) -> list[tuple[str, str]]:
        """(node_id, host) for active satellites that accept storage."""
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            rows = conn.execute(
                "SELECT id, host FROM fleet_satellites "
                "WHERE role IN ('storage', 'both') AND status = 'active'"
            ).fetchall()
        finally:
            conn.close()
        return [(r[0], r[1]) for r in rows]

    def _pick_target_satellite(self) -> tuple[str, str] | None:
        """Least-full active storage satellite as (node_id, host), or None."""
        candidates = self._active_storage_satellites()
        best: tuple[str, str] | None = None
        best_free = -1
        headers = {"Tailscale-User-Login": self.operator_identity}
        for node_id, host in candidates:
            url = f"http://{host}:{_SATELLITE_STORAGE_PORT}/v1/storage/capacity"
            try:
                with self._client_factory() as client:
                    resp = client.get(
                        url, headers=headers, timeout=_CAPACITY_TIMEOUT_SEC
                    )
            except (httpx.TimeoutException, httpx.TransportError):
                continue
            if resp.status_code != 200:
                continue
            free = resp.json().get("free_bytes", 0)
            if free > best_free:
                best_free = free
                best = (node_id, host)
        return best


def _json_metadata(node_id: str, digest: str, size: int) -> str:
    import json
    return json.dumps({"node": node_id, "sha256": digest, "size": size})
