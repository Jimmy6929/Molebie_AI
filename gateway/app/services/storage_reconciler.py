"""Manifest reconciliation — light scrub for the primary's blob index.

Slice 9.4 ships migration: blobs move primary → satellite, and the
primary records what landed where in ``satellite_blobs``. Nothing has so
far verified that what the table claims is on a satellite is *still*
actually there. Drift sources: a satellite reformatted out-of-band, an
operator ``rm``, a crashed mid-PUT leaving only a ``.partial``, a failed
in-flight migration that wrote bytes but never finished its DB
transaction.

This module is the "light scrub" (Ceph terminology): the satellite
enumerates what it has via ``GET /v1/storage/manifest``, the primary
diffs against ``satellite_blobs WHERE node=?``, and each per-blob
outcome becomes either a refreshed ``last_verified_at`` (match) or a
``storage.drift`` audit event (missing / orphan / size_mismatch). Detect
only — no auto-healing in 9.5; healing requires policy decisions (grace
periods, replication) that v0.2 doesn't have the prerequisites for.

Sync by design (sync httpx + sync sqlite3) — matches the
``StorageMover``/``TieredStorageService`` precedent. The HTTP timeout is
shared with the rest of the storage path.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import httpx

from app.schema import _get_db_path

_SATELLITE_STORAGE_PORT = 8090
_HTTP_TIMEOUT_SEC = 10.0

DriftKind = Literal["missing_on_satellite", "orphan_on_satellite", "size_mismatch"]


@dataclass
class DriftEntry:
    sha256: str
    kind: DriftKind
    expected_size: int | None = None
    actual_size: int | None = None


@dataclass
class ReconciliationReport:
    node_id: str
    verified: int = 0
    drift: list[DriftEntry] = field(default_factory=list)
    fetch_error: str | None = None


class ManifestReconciler:
    """Compares the primary's ``satellite_blobs`` to a satellite's manifest."""

    def __init__(
        self,
        operator_identity: str,
        data_dir: str,
        http_client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self.operator_identity = operator_identity
        self.data_dir = data_dir
        self._client_factory = http_client_factory or httpx.Client

    # ----- public API -----

    def reconcile(self, node_id: str) -> ReconciliationReport:
        host = self._resolve_satellite_host(node_id)
        if host is None:
            return ReconciliationReport(node_id=node_id, fetch_error="unknown_satellite")

        manifest = self._fetch_manifest(host)
        if isinstance(manifest, str):
            return ReconciliationReport(node_id=node_id, fetch_error=manifest)

        actual = {b["sha256"]: int(b["size_bytes"]) for b in manifest.get("blobs", [])}
        expected = self._expected_blobs(node_id)
        matches, drifts = _diff(expected, actual)

        now = datetime.now(timezone.utc).isoformat()
        self._record_outcomes(node_id, matches, drifts, now)
        return ReconciliationReport(node_id=node_id, verified=len(matches), drift=drifts)

    def reconcile_all(self) -> list[ReconciliationReport]:
        return [self.reconcile(nid) for nid in self._active_storage_satellite_ids()]

    # ----- helpers -----

    def _fetch_manifest(self, host: str) -> dict | str:
        """Return parsed manifest dict, or an error-tag string on failure."""
        url = f"http://{host}:{_SATELLITE_STORAGE_PORT}/v1/storage/manifest"
        headers = {"Tailscale-User-Login": self.operator_identity}
        try:
            with self._client_factory() as client:
                resp = client.get(url, headers=headers, timeout=_HTTP_TIMEOUT_SEC)
        except (httpx.TimeoutException, httpx.TransportError):
            return "manifest_fetch_failed"
        if resp.status_code != 200:
            return "manifest_fetch_failed"
        return resp.json()

    def _expected_blobs(self, node_id: str) -> dict[str, int]:
        """Map ``sha256 → size_bytes`` for blobs the primary believes live on
        ``node_id``."""
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            rows = conn.execute(
                "SELECT sha256, size_bytes FROM satellite_blobs "
                "WHERE satellite_node_id = ?",
                (node_id,),
            ).fetchall()
        finally:
            conn.close()
        return {r[0]: int(r[1]) for r in rows}

    def _resolve_satellite_host(self, node_id: str) -> str | None:
        """Look up a satellite's host by id from this reconciler's data_dir.

        Local to the reconciler rather than reusing
        ``storage._resolve_satellite_host_from_db`` because that helper
        reads ``data_dir`` from global settings, which doesn't compose
        cleanly with the reconciler's per-instance data_dir."""
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            row = conn.execute(
                "SELECT host FROM fleet_satellites WHERE id = ?", (node_id,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def _active_storage_satellite_ids(self) -> list[str]:
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            rows = conn.execute(
                "SELECT id FROM fleet_satellites "
                "WHERE role IN ('storage', 'both') AND status = 'active'"
            ).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]

    def _record_outcomes(
        self,
        node_id: str,
        matches: list[str],
        drifts: list[DriftEntry],
        now: str,
    ) -> None:
        """One sync transaction: bulk-stamp matches, INSERT one audit row per
        drift entry. Mirrors ``storage_mover._record_migration``'s pattern."""
        conn = sqlite3.connect(_get_db_path(self.data_dir))
        try:
            if matches:
                conn.executemany(
                    "UPDATE satellite_blobs SET last_verified_at = ? "
                    "WHERE sha256 = ? AND satellite_node_id = ?",
                    [(now, sha, node_id) for sha in matches],
                )
            for d in drifts:
                conn.execute(
                    "INSERT INTO audit_events "
                    "(event_type, actor, target, metadata_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        "storage.drift",
                        "system",
                        d.sha256,
                        _drift_metadata(node_id, d),
                        now,
                    ),
                )
            conn.commit()
        finally:
            conn.close()


def _diff(
    expected: dict[str, int],
    actual: dict[str, int],
) -> tuple[list[str], list[DriftEntry]]:
    """Compare primary's expected vs satellite's actual blob inventory.

    Returns ``(matches, drifts)``. ``matches`` is the list of sha256s
    present on both sides with equal sizes (eligible for last_verified_at
    stamping). ``drifts`` covers everything else.
    """
    matches: list[str] = []
    drifts: list[DriftEntry] = []
    for sha, exp_size in expected.items():
        if sha not in actual:
            drifts.append(DriftEntry(sha, "missing_on_satellite", expected_size=exp_size))
            continue
        act_size = actual[sha]
        if act_size != exp_size:
            drifts.append(
                DriftEntry(sha, "size_mismatch", expected_size=exp_size, actual_size=act_size)
            )
        else:
            matches.append(sha)
    for sha, act_size in actual.items():
        if sha not in expected:
            drifts.append(DriftEntry(sha, "orphan_on_satellite", actual_size=act_size))
    return matches, drifts


def _drift_metadata(node_id: str, entry: DriftEntry) -> str:
    payload: dict = {"node": node_id, "kind": entry.kind}
    if entry.expected_size is not None:
        payload["expected_size"] = entry.expected_size
    if entry.actual_size is not None:
        payload["actual_size"] = entry.actual_size
    return json.dumps(payload)
