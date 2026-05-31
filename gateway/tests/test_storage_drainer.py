"""Tests for StorageDrainer — slice 9.6.

Real LocalStorageService + real SQLite over a tempdir; satellite HTTP is
mocked via an injected client factory. The "delete-satellite-only-after-
DB-commit" invariant is the key behavior under test.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schema import init_database_sync
from app.services.storage import LocalStorageService
from app.services.storage_drainer import (
    DrainBatchReport,
    StorageDrainer,
)
from tests.test_storage_mover import _register_satellite


@pytest.fixture
def data_dir():
    with tempfile.TemporaryDirectory() as td:
        init_database_sync(td, embedding_dim=1024, auth_mode="single")
        yield td


# ─────────────────────── seeding helpers ───────────────────────


def _seed_satellite_doc(
    data_dir: str, node_id: str, sha256: str, size: int,
    *, filename: str = "report.pdf", file_type: str = "pdf",
    user_id: str = "user-abc",
) -> str:
    """Plant a documents row whose storage_path is satellite://<node>/<sha>
    plus the corresponding satellite_blobs row. Returns the doc_id."""
    doc_id = uuid.uuid4().hex
    storage_path = f"satellite://{node_id}/{sha256}"
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        conn.execute(
            "INSERT INTO documents (id, user_id, filename, storage_path, file_type, "
            "file_size, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'completed', ?)",
            (doc_id, user_id, filename, storage_path, file_type, size,
             datetime.now(timezone.utc).isoformat()),
        )
        # Idempotent insert for satellite_blobs in case multiple docs share a digest.
        conn.execute(
            "INSERT INTO satellite_blobs "
            "(sha256, satellite_node_id, blob_type, size_bytes, uploaded_at) "
            "VALUES (?, ?, 'document', ?, ?) "
            "ON CONFLICT(sha256, satellite_node_id) DO NOTHING",
            (sha256, node_id, size, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return doc_id


def _doc_path(data_dir: str, doc_id: str) -> str:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        return conn.execute(
            "SELECT storage_path FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()[0]
    finally:
        conn.close()


def _satellite_blobs_count(data_dir: str, node_id: str) -> int:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM satellite_blobs WHERE satellite_node_id = ?",
            (node_id,),
        ).fetchone()[0]
    finally:
        conn.close()


def _audit_events(data_dir: str, event_type: str) -> list[dict]:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM audit_events WHERE event_type = ? ORDER BY id ASC",
            (event_type,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─────────────────────── fake httpx ───────────────────────


class _Resp:
    def __init__(self, status: int, content: bytes = b"", json_body: dict | None = None):
        self.status_code = status
        self.content = content
        self._json = json_body or {}

    def json(self):
        return self._json


class _DrainerClient:
    """Serves blob bytes on GET, returns status on DELETE/GET-capacity."""

    def __init__(
        self,
        *,
        blobs: dict[str, bytes] | None = None,
        delete_status: int = 204,
        capacity_reachable: bool = True,
        get_raises: set | None = None,
    ):
        self._blobs = blobs or {}
        self._delete_status = delete_status
        self._capacity_reachable = capacity_reachable
        self._get_raises = get_raises or set()
        self.deletes: list[str] = []  # sha256s we were asked to delete

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, headers=None, timeout=None):
        if "/capacity" in url:
            if not self._capacity_reachable:
                raise httpx.ConnectError("refused")
            return _Resp(200, json_body={"free_bytes": 10**12, "total_bytes": 10**12})
        # /v1/storage/blobs/<sha>
        sha = url.rsplit("/", 1)[-1]
        if sha in self._get_raises:
            raise httpx.TimeoutException("timeout")
        if sha not in self._blobs:
            return _Resp(404)
        return _Resp(200, content=self._blobs[sha])

    def request(self, method, url, headers=None, timeout=None):
        if method == "DELETE":
            sha = url.rsplit("/", 1)[-1]
            self.deletes.append(sha)
            return _Resp(self._delete_status)
        raise AssertionError(f"unexpected method {method}")


def _drainer(data_dir, client) -> StorageDrainer:
    return StorageDrainer(
        local=LocalStorageService(data_dir),
        operator_identity="op@x",
        data_dir=data_dir,
        http_client_factory=lambda: client,
    )


# ─────────────────────── happy path ───────────────────────


class TestDrainHappy:
    def test_single_blob_drained_doc_relocated(self, data_dir):
        content = b"hello drain world"
        sha = hashlib.sha256(content).hexdigest()
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        doc_id = _seed_satellite_doc(data_dir, "node-a", sha, len(content))

        client = _DrainerClient(blobs={sha: content})
        report = _drainer(data_dir, client).drain("node-a", limit=10)

        assert report.fetch_error is None
        assert report.drained == 1 and report.skipped == 0
        assert report.bytes_drained == len(content)
        assert report.remaining == 0
        assert report.results[0].docs_relocated == 1
        # documents.storage_path now points at a local path (no scheme prefix
        # = legacy/bare-path; LocalStorageService.upload_document returns
        # "<user>/<uuid>_<filename>").
        new_path = _doc_path(data_dir, doc_id)
        assert "satellite://" not in new_path
        assert new_path.startswith("user-abc/")
        # The local file actually exists.
        assert (Path(data_dir) / "documents" / new_path).exists()
        # The satellite_blobs row is gone.
        assert _satellite_blobs_count(data_dir, "node-a") == 0
        # Satellite DELETE was issued.
        assert client.deletes == [sha]
        # Audit row recorded.
        audits = _audit_events(data_dir, "storage.drain")
        assert len(audits) == 1
        meta = json.loads(audits[0]["metadata_json"])
        assert meta["node"] == "node-a" and meta["sha256"] == sha
        assert meta["docs_relocated"] == 1 and meta["doc_ids"] == [doc_id]

    def test_multi_doc_dedup_one_blob_three_docs(self, data_dir):
        """CAS dedup at the satellite: one blob can back many documents."""
        content = b"shared bytes"
        sha = hashlib.sha256(content).hexdigest()
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        ids = [_seed_satellite_doc(data_dir, "node-a", sha, len(content),
                                   filename=f"f{i}.pdf")
               for i in range(3)]

        client = _DrainerClient(blobs={sha: content})
        report = _drainer(data_dir, client).drain("node-a", limit=10)

        assert report.drained == 1   # one blob
        assert report.results[0].docs_relocated == 3
        # All three documents now have local paths, each unique.
        new_paths = {_doc_path(data_dir, i) for i in ids}
        assert len(new_paths) == 3   # distinct files per doc
        for p in new_paths:
            assert not p.startswith("satellite://")
        # Exactly one satellite_blobs row removed, one DELETE issued.
        assert _satellite_blobs_count(data_dir, "node-a") == 0
        assert client.deletes == [sha]
        # One audit row, doc_ids enumerates all three.
        audits = _audit_events(data_dir, "storage.drain")
        assert len(audits) == 1
        meta = json.loads(audits[0]["metadata_json"])
        assert sorted(meta["doc_ids"]) == sorted(ids)


# ─────────────────────── failure-safe paths ───────────────────────


class TestDrainFailureSafe:
    def test_hash_mismatch_leaves_intact_emits_drift(self, data_dir):
        sha = "a" * 64
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        doc_id = _seed_satellite_doc(data_dir, "node-a", sha, 99)

        # Satellite returns bytes that do NOT hash to `sha`.
        client = _DrainerClient(blobs={sha: b"corrupted"})
        report = _drainer(data_dir, client).drain("node-a", limit=10)

        assert report.drained == 0 and report.skipped == 1
        assert report.results[0].reason == "hash_mismatch"
        # Doc untouched.
        assert _doc_path(data_dir, doc_id).startswith("satellite://")
        # satellite_blobs row still present.
        assert _satellite_blobs_count(data_dir, "node-a") == 1
        # No DELETE issued.
        assert client.deletes == []
        # Drift event emitted.
        drift = _audit_events(data_dir, "storage.drift")
        assert len(drift) == 1
        assert json.loads(drift[0]["metadata_json"])["kind"] == "size_mismatch"

    def test_satellite_get_timeout_leaves_intact(self, data_dir):
        content = b"x"
        sha = hashlib.sha256(content).hexdigest()
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        doc_id = _seed_satellite_doc(data_dir, "node-a", sha, len(content))

        client = _DrainerClient(blobs={sha: content}, get_raises={sha})
        report = _drainer(data_dir, client).drain("node-a", limit=10)

        assert report.skipped == 1
        assert report.results[0].reason == "satellite_get_failed"
        assert _doc_path(data_dir, doc_id).startswith("satellite://")
        assert _satellite_blobs_count(data_dir, "node-a") == 1
        assert client.deletes == []

    def test_satellite_delete_404_counts_as_drained(self, data_dir):
        content = b"already-gone"
        sha = hashlib.sha256(content).hexdigest()
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        _seed_satellite_doc(data_dir, "node-a", sha, len(content))

        client = _DrainerClient(blobs={sha: content}, delete_status=404)
        report = _drainer(data_dir, client).drain("node-a", limit=10)

        assert report.drained == 1   # 404 on DELETE means "already gone" — fine
        assert client.deletes == [sha]

    def test_satellite_delete_500_still_progresses_db(self, data_dir, capsys):
        """If the satellite DELETE fails after we've committed the DB rewrite,
        the result is still ``drained`` — the DB is the source of truth and
        the orphan-on-satellite is a 9.5-reconcile problem."""
        content = b"orphan-on-fail"
        sha = hashlib.sha256(content).hexdigest()
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        _seed_satellite_doc(data_dir, "node-a", sha, len(content))

        client = _DrainerClient(blobs={sha: content}, delete_status=500)
        report = _drainer(data_dir, client).drain("node-a", limit=10)

        assert report.drained == 1
        assert _satellite_blobs_count(data_dir, "node-a") == 0
        # We log a warning to stdout for the operator.
        out = capsys.readouterr().out
        assert "satellite DELETE returned HTTP 500" in out


# ─────────────────────── batching + selection ───────────────────────


class TestBatching:
    def test_batch_limit_respected_and_remaining_counted(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        blobs: dict[str, bytes] = {}
        for i in range(3):
            content = b"x" * (10 + i)
            sha = hashlib.sha256(content).hexdigest()
            _seed_satellite_doc(data_dir, "node-a", sha, len(content),
                                filename=f"f{i}.pdf")
            blobs[sha] = content

        client = _DrainerClient(blobs=blobs)
        report = _drainer(data_dir, client).drain("node-a", limit=2)

        assert report.drained == 2
        assert report.remaining == 1
        assert _satellite_blobs_count(data_dir, "node-a") == 1

    def test_unknown_satellite_returns_fetch_error(self, data_dir):
        client = _DrainerClient(blobs={})
        report = _drainer(data_dir, client).drain("ghost", limit=10)
        assert isinstance(report, DrainBatchReport)
        assert report.fetch_error == "unknown_satellite"
        assert report.drained == 0


# ─────────────────────── preview ───────────────────────


class TestPreview:
    def test_preview_returns_counts_and_feasibility(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        for i in range(2):
            content = b"x" * 100
            sha = hashlib.sha256(content + bytes([i])).hexdigest()
            _seed_satellite_doc(data_dir, "node-a", sha, len(content),
                                filename=f"f{i}.pdf")

        client = _DrainerClient(blobs={})  # reachable via capacity ping
        preview = _drainer(data_dir, client).preview("node-a")

        assert preview.blob_count == 2
        assert preview.total_bytes == 200
        assert preview.satellite_reachable is True
        assert preview.primary_free_bytes > 0
        assert preview.feasible is True

    def test_preview_unreachable_satellite(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        _seed_satellite_doc(data_dir, "node-a", "a" * 64, 10)

        client = _DrainerClient(blobs={}, capacity_reachable=False)
        preview = _drainer(data_dir, client).preview("node-a")

        assert preview.satellite_reachable is False
        # Counts still surface even if satellite is down.
        assert preview.blob_count == 1
