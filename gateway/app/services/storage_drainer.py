"""Storage drain — pull blobs back from a satellite to the primary.

The inverse of ``StorageMover``: each blob the primary records on a
satellite (``satellite_blobs``) is fetched back, rewritten into local
storage (one local file per affected document — CAS dedup at the
satellite can fan a single blob out to N documents), then deleted from
the satellite. The fleet_satellites row is removed by the CLI's final
``DELETE`` once drain reports zero remaining. Closes Storage Phase 1.

Failure-safe ordering (symmetric with the 9.4a mover's "delete-local-
only-after-satellite-verify" invariant):

    GET bytes → hash-verify → write local(s) → COMMIT (UPDATE docs +
        DELETE satellite_blobs + audit) → DELETE on satellite

The satellite blob is deleted ONLY after the primary has the bytes AND
the DB rewrite is committed. Any failure before commit leaves the
document fully readable via the satellite — no data-loss window. The
post-commit satellite DELETE is best-effort: failures log and proceed
because the DB already reflects the truth (the blob becomes an orphan
on the satellite, which 9.5's reconciler will detect).

Two modes, à la Cassandra:

* **Graceful** (``drain``): the satellite is reachable; pull bytes,
  rewrite ``storage_path`` to ``local://...``, then DELETE on satellite.
* **Force** (``force_remove``): satellite assumed dead. No satellite
  contact; emit a single ``storage.force_remove`` audit recording the
  lost-blob count; delete ``satellite_blobs`` rows + the
  ``fleet_satellites`` row. Affected documents keep their dangling
  ``satellite://`` paths — ``TieredStorageService`` will surface a clear
  ``BlobUnreachableError`` on read, which is the loud failure the
  operator wants.

Sync by design (sync httpx + sync sqlite3) — matches the
``StorageMover`` and ``ManifestReconciler`` precedent. The route wraps
calls in ``asyncio.to_thread`` to keep the event loop responsive.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.schema import _get_db_path
from app.services.storage import LocalStorageService
from app.services.storage_uri import build_satellite

_SATELLITE_STORAGE_PORT = 8090
_HTTP_TIMEOUT_SEC = 30.0  # generous — blobs can be large
_REACHABILITY_TIMEOUT_SEC = 5.0
# Reserve a small headroom on the primary so a drain that just barely fits
# doesn't strand the operator at 100% disk after it lands.
_PRIMARY_FREE_HEADROOM_BYTES = 100 * 1024 * 1024  # 100 MB


@dataclass
class DrainResult:
    """Outcome of attempting to drain one blob."""

    sha256: str
    drained: bool
    docs_relocated: int = 0
    bytes_drained: int = 0
    reason: str | None = None  # why skipped / failed


@dataclass
class DrainBatchReport:
    node_id: str
    drained: int = 0           # blobs successfully moved
    skipped: int = 0           # blobs that errored / mismatched
    remaining: int = 0         # satellite_blobs rows still pointing at this node
    bytes_drained: int = 0
    results: list[DrainResult] = field(default_factory=list)
    fetch_error: str | None = None  # set if we couldn't even reach the satellite


@dataclass
class DrainPreview:
    node_id: str
    blob_count: int = 0
    total_bytes: int = 0
    primary_free_bytes: int = 0
    feasible: bool = False
    satellite_reachable: bool = False


@dataclass
class ForceRemoveResult:
    node_id: str
    satellite_existed: bool
    lost_blobs: int = 0
    lost_bytes: int = 0


class StorageDrainer:
    """Move blobs back from a satellite to the primary's local storage."""

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

    def preview(self, node_id: str) -> DrainPreview:
        host = self._resolve_satellite_host(node_id)
        count, total = self._satellite_blobs_aggregate(node_id)
        primary_free = _measure_primary_free(self.data_dir)
        if host is None:
            return DrainPreview(
                node_id=node_id,
                blob_count=count,
                total_bytes=total,
                primary_free_bytes=primary_free,
                satellite_reachable=False,
                feasible=False,
            )
        reachable = self._ping_satellite(host)
        feasible = primary_free - _PRIMARY_FREE_HEADROOM_BYTES >= total
        return DrainPreview(
            node_id=node_id,
            blob_count=count,
            total_bytes=total,
            primary_free_bytes=primary_free,
            feasible=feasible,
            satellite_reachable=reachable,
        )

    def drain(self, node_id: str, limit: int) -> DrainBatchReport:
        host = self._resolve_satellite_host(node_id)
        if host is None:
            return DrainBatchReport(node_id=node_id, fetch_error="unknown_satellite")

        rows = self._eligible_blob_rows(node_id, limit)
        report = DrainBatchReport(node_id=node_id)
        for row in rows:
            result = self._drain_one(row, host=host, node_id=node_id)
            report.results.append(result)
            if result.drained:
                report.drained += 1
                report.bytes_drained += result.bytes_drained
            else:
                report.skipped += 1
        report.remaining = self._remaining_count(node_id)
        return report

    def force_remove(self, node_id: str) -> ForceRemoveResult:
        """Skip drain entirely: delete satellite_blobs + fleet_satellites rows,
        emit one ``storage.force_remove`` audit. Used when the satellite is
        gone and the operator accepts data loss for documents that lived on it.
        """
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            sat_row = conn.execute(
                "SELECT 1 FROM fleet_satellites WHERE id = ?", (node_id,)
            ).fetchone()
            satellite_existed = sat_row is not None

            agg = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) "
                "FROM satellite_blobs WHERE satellite_node_id = ?",
                (node_id,),
            ).fetchone()
            lost_blobs, lost_bytes = int(agg[0]), int(agg[1])

            conn.execute(
                "DELETE FROM satellite_blobs WHERE satellite_node_id = ?",
                (node_id,),
            )
            if satellite_existed:
                conn.execute(
                    "DELETE FROM fleet_satellites WHERE id = ?", (node_id,)
                )
                conn.execute(
                    "INSERT INTO audit_events "
                    "(event_type, actor, target, metadata_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        "storage.force_remove",
                        "system",
                        node_id,
                        json.dumps({
                            "node": node_id,
                            "lost_blobs": lost_blobs,
                            "lost_bytes": lost_bytes,
                        }),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        return ForceRemoveResult(
            node_id=node_id,
            satellite_existed=satellite_existed,
            lost_blobs=lost_blobs,
            lost_bytes=lost_bytes,
        )

    # ----- per-blob drain (failure-safe sequence) -----

    def _drain_one(
        self, blob_row: dict, *, host: str, node_id: str
    ) -> DrainResult:
        sha = blob_row["sha256"]
        expected_size = int(blob_row["size_bytes"])
        affected = self._affected_documents(node_id, sha)

        # GET bytes from satellite.
        try:
            data = self._get_blob(host, sha)
        except _SatelliteFetchError:
            return DrainResult(sha, drained=False, reason="satellite_get_failed")

        # Hash-verify before we trust the bytes — drain is destructive.
        if hashlib.sha256(data).hexdigest() != sha:
            self._emit_drift(sha, node_id, expected=expected_size, actual=len(data))
            return DrainResult(sha, drained=False, reason="hash_mismatch")

        # Write a local file for each affected document. If write fails
        # mid-way (e.g. disk full), we abort before COMMIT — partial local
        # files become orphans on the primary, but no doc has its
        # storage_path mutated yet, so reads still resolve to the satellite.
        try:
            new_paths: list[tuple[str, str]] = []  # (doc_id, new_storage_path)
            for doc in affected:
                new_path = self.local.upload_document(
                    doc["user_id"], doc["filename"], data, doc["file_type"]
                )
                new_paths.append((doc["id"], new_path))
        except OSError:
            return DrainResult(sha, drained=False, reason="local_write_failed")

        # One transaction: UPDATE each doc + DELETE the satellite_blobs row
        # + INSERT the audit event.
        now = datetime.now(timezone.utc).isoformat()
        self._commit_drain(
            node_id=node_id, sha=sha, size=len(data),
            new_paths=new_paths, affected_doc_ids=[d["id"] for d in affected], now=now,
        )

        # Best-effort satellite-side DELETE: the DB already says "not on
        # satellite," so a failure here just leaves an orphan that 9.5 will
        # eventually surface. Don't fail the drain over it.
        self._delete_blob_on_satellite(host, sha)

        return DrainResult(
            sha, drained=True,
            docs_relocated=len(new_paths), bytes_drained=len(data),
        )

    # ----- helpers: DB queries -----

    def _resolve_satellite_host(self, node_id: str) -> str | None:
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            row = conn.execute(
                "SELECT host FROM fleet_satellites WHERE id = ?", (node_id,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def _eligible_blob_rows(self, node_id: str, limit: int) -> list[dict]:
        """Oldest-first (coldest-first heuristic) blob rows for this node."""
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT sha256, size_bytes FROM satellite_blobs "
                "WHERE satellite_node_id = ? "
                "ORDER BY uploaded_at ASC LIMIT ?",
                (node_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def _affected_documents(self, node_id: str, sha: str) -> list[dict]:
        """All documents whose storage_path points at this blob — could be
        multiple due to CAS dedup at the satellite."""
        target_uri = build_satellite(node_id, sha)
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, user_id, filename, file_type FROM documents "
                "WHERE storage_path = ?",
                (target_uri,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def _satellite_blobs_aggregate(self, node_id: str) -> tuple[int, int]:
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) "
                "FROM satellite_blobs WHERE satellite_node_id = ?",
                (node_id,),
            ).fetchone()
            return int(row[0]), int(row[1])
        finally:
            conn.close()

    def _remaining_count(self, node_id: str) -> int:
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM satellite_blobs WHERE satellite_node_id = ?",
                (node_id,),
            ).fetchone()
            return int(row[0])
        finally:
            conn.close()

    def _commit_drain(
        self, *, node_id: str, sha: str, size: int,
        new_paths: list[tuple[str, str]], affected_doc_ids: list[str], now: str,
    ) -> None:
        """One transaction: rewrite storage_paths, drop the satellite_blobs
        row, and emit the audit event."""
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            conn.executemany(
                "UPDATE documents SET storage_path = ? WHERE id = ?",
                [(path, doc_id) for doc_id, path in new_paths],
            )
            conn.execute(
                "DELETE FROM satellite_blobs "
                "WHERE sha256 = ? AND satellite_node_id = ?",
                (sha, node_id),
            )
            conn.execute(
                "INSERT INTO audit_events "
                "(event_type, actor, target, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "storage.drain",
                    "system",
                    sha,
                    json.dumps({
                        "node": node_id,
                        "sha256": sha,
                        "size": size,
                        "docs_relocated": len(new_paths),
                        "doc_ids": affected_doc_ids,
                    }),
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _emit_drift(
        self, sha: str, node_id: str, *, expected: int, actual: int,
    ) -> None:
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            conn.execute(
                "INSERT INTO audit_events "
                "(event_type, actor, target, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "storage.drift",
                    "system",
                    sha,
                    json.dumps({
                        "node": node_id,
                        "kind": "size_mismatch",
                        "expected_size": expected,
                        "actual_size": actual,
                    }),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ----- helpers: satellite HTTP -----

    def _get_blob(self, host: str, sha: str) -> bytes:
        url = f"http://{host}:{_SATELLITE_STORAGE_PORT}/v1/storage/blobs/{sha}"
        headers = {"Tailscale-User-Login": self.operator_identity}
        try:
            with self._client_factory() as client:
                resp = client.get(url, headers=headers, timeout=_HTTP_TIMEOUT_SEC)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise _SatelliteFetchError(str(exc)) from exc
        if resp.status_code != 200:
            raise _SatelliteFetchError(f"HTTP {resp.status_code}")
        return resp.content

    def _delete_blob_on_satellite(self, host: str, sha: str) -> None:
        """Best-effort DELETE; failures are logged via print and don't fail
        the drain because the DB already reflects the truth."""
        url = f"http://{host}:{_SATELLITE_STORAGE_PORT}/v1/storage/blobs/{sha}"
        headers = {"Tailscale-User-Login": self.operator_identity}
        try:
            with self._client_factory() as client:
                resp = client.request(
                    "DELETE", url, headers=headers, timeout=_HTTP_TIMEOUT_SEC,
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            print(f"[drain] satellite DELETE failed for {sha[:8]}: {exc}")
            return
        # 404 = already gone, treat as success.
        if resp.status_code not in (200, 204, 404):
            print(
                f"[drain] satellite DELETE returned HTTP {resp.status_code} "
                f"for {sha[:8]} — blob is now orphaned on the satellite "
                f"(9.5 reconcile will surface it)"
            )

    def _ping_satellite(self, host: str) -> bool:
        """Cheap reachability check used by ``preview``. Reuses 9.1's
        ``/v1/storage/capacity`` so we don't add a new satellite-side route."""
        url = f"http://{host}:{_SATELLITE_STORAGE_PORT}/v1/storage/capacity"
        headers = {"Tailscale-User-Login": self.operator_identity}
        try:
            with self._client_factory() as client:
                resp = client.get(
                    url, headers=headers, timeout=_REACHABILITY_TIMEOUT_SEC,
                )
        except (httpx.TimeoutException, httpx.TransportError):
            return False
        return resp.status_code == 200


class _SatelliteFetchError(Exception):
    """Raised by ``_get_blob`` on timeout / transport error / non-200."""


def _measure_primary_free(data_dir: str) -> int:
    target = Path(data_dir)
    if not target.exists():
        target = target.parent
    return shutil.disk_usage(target).free
