"""Tests for ManifestReconciler — slice 9.5.

We exercise ``reconcile()`` directly against a real SQLite tempdir +
seeded ``satellite_blobs`` rows; the satellite manifest is mocked via an
injected httpx client.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schema import init_database_sync
from app.services.storage_reconciler import (
    DriftEntry,
    ManifestReconciler,
    ReconciliationReport,
    _diff,
)
from tests.test_storage_mover import _register_satellite  # reuse 9.4a helper


@pytest.fixture
def data_dir():
    with tempfile.TemporaryDirectory() as td:
        init_database_sync(td, embedding_dim=1024, auth_mode="single")
        yield td


# ─────────────────────── seeding helpers ───────────────────────


def _seed_blob_row(
    data_dir: str, sha256: str, node_id: str, size: int,
    last_verified_at: str | None = None,
) -> None:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        conn.execute(
            "INSERT INTO satellite_blobs "
            "(sha256, satellite_node_id, blob_type, size_bytes, uploaded_at, last_verified_at) "
            "VALUES (?, ?, 'document', ?, '2026-05-01T00:00:00+00:00', ?)",
            (sha256, node_id, size, last_verified_at),
        )
        conn.commit()
    finally:
        conn.close()


def _blob_row(data_dir: str, sha256: str, node_id: str) -> dict:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM satellite_blobs WHERE sha256 = ? AND satellite_node_id = ?",
            (sha256, node_id),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _audit_events(data_dir: str) -> list[dict]:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM audit_events WHERE event_type = 'storage.drift' "
            "ORDER BY id ASC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─────────────────────── fake httpx ───────────────────────


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None):
        self.status_code = status_code
        self._json = json_body or {}

    def json(self):
        return self._json


class _ManifestClient:
    """Serves a programmable manifest per host or raises an injected error."""

    def __init__(self, manifest_by_host: dict, raise_for: set | None = None):
        self._manifest = manifest_by_host
        self._raise = raise_for or set()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, headers=None, timeout=None):
        host = url.split("//")[1].split(":")[0]
        if host in self._raise:
            raise httpx.ConnectError("unreachable")
        manifest = self._manifest.get(host)
        if manifest is None:
            return _FakeResponse(404)
        return _FakeResponse(200, json_body=manifest)


def _reconciler(data_dir: str, client) -> ManifestReconciler:
    return ManifestReconciler(
        operator_identity="op@x", data_dir=data_dir,
        http_client_factory=lambda: client,
    )


# ─────────────────────── pure diff ───────────────────────


class TestDiff:
    def test_all_match(self):
        matches, drifts = _diff({"a" * 64: 10, "b" * 64: 20}, {"a" * 64: 10, "b" * 64: 20})
        assert sorted(matches) == sorted(["a" * 64, "b" * 64])
        assert drifts == []

    def test_missing_orphan_mismatch_in_one_call(self):
        matches, drifts = _diff(
            expected={"aa" * 32: 10, "bb" * 32: 20, "cc" * 32: 30},
            actual={"aa" * 32: 10, "bb" * 32: 99, "dd" * 32: 40},
        )
        assert matches == ["aa" * 32]
        kinds = sorted((d.kind, d.sha256) for d in drifts)
        assert kinds == [
            ("missing_on_satellite", "cc" * 32),
            ("orphan_on_satellite", "dd" * 32),
            ("size_mismatch", "bb" * 32),
        ]


# ─────────────────────── reconcile() outcomes ───────────────────────


class TestReconcile:
    def test_all_match_stamps_last_verified(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        _seed_blob_row(data_dir, "a" * 64, "node-a", 10)
        _seed_blob_row(data_dir, "b" * 64, "node-a", 20)
        client = _ManifestClient({
            "100.64.0.9": {"generated_at": "2026-06-01T...", "blobs": [
                {"sha256": "a" * 64, "size_bytes": 10},
                {"sha256": "b" * 64, "size_bytes": 20},
            ]}
        })

        report = _reconciler(data_dir, client).reconcile("node-a")

        assert report.verified == 2
        assert report.drift == []
        assert report.fetch_error is None
        # Both rows have last_verified_at populated
        for sha in ["a" * 64, "b" * 64]:
            row = _blob_row(data_dir, sha, "node-a")
            assert row["last_verified_at"] is not None

    def test_missing_on_satellite_emits_drift(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        _seed_blob_row(data_dir, "a" * 64, "node-a", 10)
        client = _ManifestClient({"100.64.0.9": {"blobs": []}})

        report = _reconciler(data_dir, client).reconcile("node-a")

        assert report.verified == 0
        assert len(report.drift) == 1
        assert report.drift[0].kind == "missing_on_satellite"
        assert report.drift[0].sha256 == "a" * 64

        events = _audit_events(data_dir)
        assert len(events) == 1
        meta = json.loads(events[0]["metadata_json"])
        assert meta == {"node": "node-a", "kind": "missing_on_satellite", "expected_size": 10}

    def test_orphan_emits_drift(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        # No row in satellite_blobs.
        client = _ManifestClient({"100.64.0.9": {"blobs": [
            {"sha256": "c" * 64, "size_bytes": 7},
        ]}})

        report = _reconciler(data_dir, client).reconcile("node-a")

        assert report.verified == 0
        assert len(report.drift) == 1
        assert report.drift[0].kind == "orphan_on_satellite"
        meta = json.loads(_audit_events(data_dir)[0]["metadata_json"])
        assert meta == {"node": "node-a", "kind": "orphan_on_satellite", "actual_size": 7}

    def test_size_mismatch_emits_drift(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        _seed_blob_row(data_dir, "a" * 64, "node-a", 100)
        client = _ManifestClient({"100.64.0.9": {"blobs": [
            {"sha256": "a" * 64, "size_bytes": 50},  # wrong size
        ]}})

        report = _reconciler(data_dir, client).reconcile("node-a")

        assert report.verified == 0
        assert len(report.drift) == 1
        d = report.drift[0]
        assert d.kind == "size_mismatch"
        assert d.expected_size == 100 and d.actual_size == 50
        # row's last_verified_at is NOT stamped
        assert _blob_row(data_dir, "a" * 64, "node-a")["last_verified_at"] is None

    def test_mixed_outcomes_single_report(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        _seed_blob_row(data_dir, "a" * 64, "node-a", 10)  # match
        _seed_blob_row(data_dir, "b" * 64, "node-a", 20)  # missing
        _seed_blob_row(data_dir, "c" * 64, "node-a", 30)  # size mismatch
        client = _ManifestClient({"100.64.0.9": {"blobs": [
            {"sha256": "a" * 64, "size_bytes": 10},
            {"sha256": "c" * 64, "size_bytes": 99},
            {"sha256": "d" * 64, "size_bytes": 40},  # orphan
        ]}})

        report = _reconciler(data_dir, client).reconcile("node-a")

        assert report.verified == 1
        kinds = sorted(d.kind for d in report.drift)
        assert kinds == ["missing_on_satellite", "orphan_on_satellite", "size_mismatch"]
        assert _blob_row(data_dir, "a" * 64, "node-a")["last_verified_at"] is not None
        assert _blob_row(data_dir, "c" * 64, "node-a")["last_verified_at"] is None
        assert len(_audit_events(data_dir)) == 3

    def test_unknown_satellite_returns_fetch_error(self, data_dir):
        # No fleet_satellites row for "ghost"
        client = _ManifestClient({})
        report = _reconciler(data_dir, client).reconcile("ghost")
        assert report.fetch_error == "unknown_satellite"
        assert report.verified == 0
        assert report.drift == []
        assert _audit_events(data_dir) == []

    def test_manifest_fetch_failure_makes_no_db_changes(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        _seed_blob_row(data_dir, "a" * 64, "node-a", 10, last_verified_at=None)
        client = _ManifestClient({}, raise_for={"100.64.0.9"})

        report = _reconciler(data_dir, client).reconcile("node-a")

        assert report.fetch_error == "manifest_fetch_failed"
        assert report.verified == 0
        # last_verified_at unchanged; no audit events written
        assert _blob_row(data_dir, "a" * 64, "node-a")["last_verified_at"] is None
        assert _audit_events(data_dir) == []


# ─────────────────────── reconcile_all ───────────────────────


class TestReconcileAll:
    def test_iterates_active_storage_satellites(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.1")
        _register_satellite(data_dir, "node-b", "100.64.0.2", role="both")
        _register_satellite(data_dir, "node-c", "100.64.0.3", role="compute")  # not storage
        _seed_blob_row(data_dir, "a" * 64, "node-a", 10)
        client = _ManifestClient({
            "100.64.0.1": {"blobs": [{"sha256": "a" * 64, "size_bytes": 10}]},
            "100.64.0.2": {"blobs": []},
        })

        reports = _reconciler(data_dir, client).reconcile_all()

        ids = sorted(r.node_id for r in reports)
        assert ids == ["node-a", "node-b"]  # compute-only excluded
        report_a = next(r for r in reports if r.node_id == "node-a")
        assert report_a.verified == 1
        report_b = next(r for r in reports if r.node_id == "node-b")
        assert report_b.verified == 0 and report_b.drift == []

    def test_no_active_storage_satellites_returns_empty(self, data_dir):
        # Only a compute satellite registered.
        _register_satellite(data_dir, "node-c", "100.64.0.3", role="compute")
        client = _ManifestClient({})
        assert _reconciler(data_dir, client).reconcile_all() == []


# ─────────────────────── return-type sanity ───────────────────────


class TestReturnTypes:
    def test_report_is_reconciliation_report(self, data_dir):
        client = _ManifestClient({})
        report = _reconciler(data_dir, client).reconcile("ghost")
        assert isinstance(report, ReconciliationReport)
        # DriftEntry import is reachable
        assert DriftEntry("x" * 64, "missing_on_satellite").kind == "missing_on_satellite"
