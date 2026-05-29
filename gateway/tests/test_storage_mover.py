"""Tests for StorageMover — the local→satellite migration engine.

Real LocalStorageService + real SQLite over tempdirs; satellite HTTP is
mocked via an injected client factory. The failure-safe ordering (delete
local only after verify) is the key behavior under test.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schema import init_database_sync
from app.services.storage import LocalStorageService
from app.services.storage_mover import StorageMover

# ─────────────────────────── fixtures + fakes ───────────────────────────


@pytest.fixture
def data_dir():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        init_database_sync(td, embedding_dim=1024, auth_mode="single")
        yield td


def _seed_document(data_dir: str, local: LocalStorageService, content: bytes) -> str:
    """Plant a completed local document + its file. Returns doc_id."""
    storage_path = local.upload_document("user-abc", "report.pdf", content, "application/pdf")
    doc_id = uuid.uuid4().hex
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        conn.execute(
            "INSERT INTO documents (id, user_id, filename, storage_path, file_type, "
            "file_size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'completed', ?)",
            (doc_id, "user-abc", "report.pdf", storage_path, "pdf", len(content),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return doc_id


def _register_satellite(data_dir: str, node_id: str, host: str, role: str = "storage",
                        status: str = "active") -> None:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO fleet_satellites (id, host, role, status, label, "
            "capabilities_json, tailscale_user, joined_at, updated_at) "
            "VALUES (?, ?, ?, ?, NULL, NULL, 'op@x', ?, ?)",
            (node_id, host, role, status, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def _doc_storage_path(data_dir: str, doc_id: str) -> str:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        row = conn.execute("SELECT storage_path FROM documents WHERE id=?", (doc_id,)).fetchone()
        return row[0]
    finally:
        conn.close()


def _satellite_blobs(data_dir: str) -> list[dict]:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM satellite_blobs")]
    finally:
        conn.close()


def _audit_events(data_dir: str) -> list[dict]:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM audit_events")]
    finally:
        conn.close()


class _FakeResponse:
    def __init__(self, status_code: int, headers: dict | None = None, json_body: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self._json = json_body or {}

    def json(self):
        return self._json


class _FakeClient:
    """Programmable per-method responses; records calls. Routes by HTTP method
    and a substring of the URL ('/blobs/' vs '/capacity')."""

    def __init__(self, *, put=None, head=None, capacity=None):
        self._put = put
        self._head = head
        self._capacity = capacity
        self.calls: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def put(self, url, content=None, headers=None, timeout=None):
        self.calls.append(("PUT", url))
        return self._dispatch(self._put)

    def head(self, url, headers=None, timeout=None):
        self.calls.append(("HEAD", url))
        return self._dispatch(self._head)

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url))
        return self._dispatch(self._capacity)

    @staticmethod
    def _dispatch(resp):
        if isinstance(resp, Exception):
            raise resp
        return resp


def _mover(data_dir, client: _FakeClient) -> StorageMover:
    local = LocalStorageService(data_dir)
    return StorageMover(
        local=local,
        operator_identity="op@example.com",
        data_dir=data_dir,
        http_client_factory=lambda: client,
    )


_CONTENT = b"hello molebie storage mover"
_DIGEST = hashlib.sha256(_CONTENT).hexdigest()


# ─────────────────────────── _migrate_one happy + skip/fail ───────────────────────────


class TestMigrateOne:
    def test_happy_path(self, data_dir):
        local = LocalStorageService(data_dir)
        doc_id = _seed_document(data_dir, local, _CONTENT)
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        client = _FakeClient(
            put=_FakeResponse(201),
            head=_FakeResponse(200, headers={"content-length": str(len(_CONTENT))}),
            capacity=_FakeResponse(200, json_body={"free_bytes": 999}),
        )
        results = _mover(data_dir, client).migrate_documents(limit=10)

        assert len(results) == 1 and results[0].migrated is True
        assert results[0].sha256 == _DIGEST
        # storage_path rewritten
        assert _doc_storage_path(data_dir, doc_id) == f"satellite://node-a/{_DIGEST}"
        # satellite_blobs row recorded
        blobs = _satellite_blobs(data_dir)
        assert len(blobs) == 1
        assert blobs[0]["sha256"] == _DIGEST
        assert blobs[0]["satellite_node_id"] == "node-a"
        assert blobs[0]["blob_type"] == "document"
        # audit event emitted
        audit = _audit_events(data_dir)
        assert any(e["event_type"] == "storage.migrate" and e["target"] == doc_id for e in audit)

    def test_local_file_deleted_after_migration(self, data_dir):
        local = LocalStorageService(data_dir)
        doc_id = _seed_document(data_dir, local, _CONTENT)
        old_path = _doc_storage_path(data_dir, doc_id)
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        client = _FakeClient(
            put=_FakeResponse(201),
            head=_FakeResponse(200, headers={"content-length": str(len(_CONTENT))}),
            capacity=_FakeResponse(200, json_body={"free_bytes": 999}),
        )
        _mover(data_dir, client).migrate_documents(limit=10)
        # The old local file must be gone.
        full = Path(data_dir) / "documents" / old_path
        assert not full.exists()

    def test_no_satellite_leaves_local_intact(self, data_dir):
        local = LocalStorageService(data_dir)
        doc_id = _seed_document(data_dir, local, _CONTENT)
        # No satellite registered.
        client = _FakeClient(capacity=_FakeResponse(200, json_body={"free_bytes": 1}))
        results = _mover(data_dir, client).migrate_documents(limit=10)
        assert results[0].migrated is False
        assert results[0].reason == "no_satellite"
        # storage_path unchanged, file still present
        assert _doc_storage_path(data_dir, doc_id).startswith("user-abc/")
        assert _satellite_blobs(data_dir) == []

    def test_upload_failure_leaves_local_intact(self, data_dir):
        local = LocalStorageService(data_dir)
        doc_id = _seed_document(data_dir, local, _CONTENT)
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        client = _FakeClient(
            put=_FakeResponse(500),
            capacity=_FakeResponse(200, json_body={"free_bytes": 999}),
        )
        results = _mover(data_dir, client).migrate_documents(limit=10)
        assert results[0].migrated is False
        assert results[0].reason == "upload_failed"
        assert _doc_storage_path(data_dir, doc_id).startswith("user-abc/")
        assert _satellite_blobs(data_dir) == []

    def test_verify_size_mismatch_leaves_local_intact(self, data_dir):
        local = LocalStorageService(data_dir)
        doc_id = _seed_document(data_dir, local, _CONTENT)
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        client = _FakeClient(
            put=_FakeResponse(201),
            head=_FakeResponse(200, headers={"content-length": "999999"}),  # wrong size
            capacity=_FakeResponse(200, json_body={"free_bytes": 999}),
        )
        results = _mover(data_dir, client).migrate_documents(limit=10)
        assert results[0].migrated is False
        assert results[0].reason == "verify_failed"
        assert _doc_storage_path(data_dir, doc_id).startswith("user-abc/")
        assert _satellite_blobs(data_dir) == []

    def test_already_remote_skipped(self, data_dir):
        local = LocalStorageService(data_dir)
        doc_id = _seed_document(data_dir, local, _CONTENT)
        # Pre-rewrite the row to a satellite URI.
        conn = sqlite3.connect(Path(data_dir) / "molebie.db")
        conn.execute(
            "UPDATE documents SET storage_path = ? WHERE id = ?",
            (f"satellite://node-a/{_DIGEST}", doc_id),
        )
        conn.commit()
        conn.close()
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        client = _FakeClient(capacity=_FakeResponse(200, json_body={"free_bytes": 999}))
        results = _mover(data_dir, client).migrate_documents(limit=10)
        # No eligible local docs → no results at all (filtered out before _migrate_one).
        assert results == []


# ─────────────────────────── satellite selection ───────────────────────────


class TestPickTargetSatellite:
    def test_picks_most_free(self, data_dir):
        _register_satellite(data_dir, "node-small", "100.64.0.1")
        _register_satellite(data_dir, "node-big", "100.64.0.2")

        def _client():
            # capacity depends on URL host
            return _CapacityClient({"100.64.0.1": 100, "100.64.0.2": 5000})

        mover = StorageMover(LocalStorageService(data_dir), "op@x", data_dir, _client)
        picked = mover._pick_target_satellite()
        assert picked == ("node-big", "100.64.0.2")

    def test_skips_unreachable(self, data_dir):
        _register_satellite(data_dir, "node-dead", "100.64.0.1")
        _register_satellite(data_dir, "node-live", "100.64.0.2")

        def _client():
            return _CapacityClient({"100.64.0.2": 500}, unreachable={"100.64.0.1"})

        mover = StorageMover(LocalStorageService(data_dir), "op@x", data_dir, _client)
        assert mover._pick_target_satellite() == ("node-live", "100.64.0.2")

    def test_none_when_no_storage_satellites(self, data_dir):
        # Register a compute-only satellite — not eligible for storage.
        _register_satellite(data_dir, "node-c", "100.64.0.3", role="compute")
        client = _FakeClient(capacity=_FakeResponse(200, json_body={"free_bytes": 1}))
        mover = StorageMover(LocalStorageService(data_dir), "op@x", data_dir, lambda: client)
        assert mover._pick_target_satellite() is None


class _CapacityClient:
    """Capacity-only fake that returns free_bytes keyed by URL host."""

    def __init__(self, free_by_host: dict, unreachable: set | None = None):
        self._free = free_by_host
        self._unreachable = unreachable or set()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, headers=None, timeout=None):
        host = url.split("//")[1].split(":")[0]
        if host in self._unreachable:
            raise httpx.ConnectError("refused")
        return _FakeResponse(200, json_body={"free_bytes": self._free.get(host, 0)})


# ─────────────────────────── batch + idempotency ───────────────────────────


class TestBatchAndIdempotency:
    def test_migrate_limit_oldest_first(self, data_dir):
        local = LocalStorageService(data_dir)
        # Three docs; migrate only 2.
        ids = [_seed_document(data_dir, local, _CONTENT + bytes([i])) for i in range(3)]
        _register_satellite(data_dir, "node-a", "100.64.0.9")

        # One shared client so PUT's recorded size carries to the HEAD verify
        # (the mover opens PUT and HEAD in separate `with` blocks).
        shared = _AlwaysOkClient()
        results = StorageMover(local, "op@x", data_dir, lambda: shared).migrate_documents(limit=2)
        assert sum(1 for r in results if r.migrated) == 2
        # Exactly two docs now remote.
        remote = [i for i in ids if _doc_storage_path(data_dir, i).startswith("satellite://")]
        assert len(remote) == 2

    def test_idempotent_second_run(self, data_dir):
        local = LocalStorageService(data_dir)
        _seed_document(data_dir, local, _CONTENT)
        _register_satellite(data_dir, "node-a", "100.64.0.9")

        shared = _AlwaysOkClient()
        mover = StorageMover(local, "op@x", data_dir, lambda: shared)
        first = mover.migrate_documents(limit=10)
        second = mover.migrate_documents(limit=10)
        assert sum(1 for r in first if r.migrated) == 1
        # Second run finds nothing local-eligible.
        assert second == []
        # Still exactly one satellite_blobs row.
        assert len(_satellite_blobs(data_dir)) == 1


class _AlwaysOkClient:
    """PUT 201, HEAD 200 with matching size, capacity ample. Size taken from
    the most recent PUT content length."""

    def __init__(self):
        self._last_size = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def put(self, url, content=None, headers=None, timeout=None):
        self._last_size = len(content)
        return _FakeResponse(201)

    def head(self, url, headers=None, timeout=None):
        return _FakeResponse(200, headers={"content-length": str(self._last_size)})

    def get(self, url, headers=None, timeout=None):
        return _FakeResponse(200, json_body={"free_bytes": 10**12})
