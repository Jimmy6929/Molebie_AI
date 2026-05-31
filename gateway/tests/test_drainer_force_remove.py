"""Tests for StorageDrainer.force_remove — slice 9.6.

The force path is the "satellite is gone, accept the data loss" mode
(à la Cassandra `nodetool removenode`). No satellite contact; primary-
side cleanup + a single ``storage.force_remove`` audit.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schema import init_database_sync
from app.services.storage import LocalStorageService
from app.services.storage_drainer import StorageDrainer
from tests.test_storage_drainer import _seed_satellite_doc
from tests.test_storage_mover import _register_satellite


@pytest.fixture
def data_dir():
    with tempfile.TemporaryDirectory() as td:
        init_database_sync(td, embedding_dim=1024, auth_mode="single")
        yield td


def _drainer(data_dir) -> StorageDrainer:
    return StorageDrainer(
        local=LocalStorageService(data_dir),
        operator_identity="op@x",
        data_dir=data_dir,
        http_client_factory=lambda: None,  # not used by force_remove
    )


def _row_counts(data_dir: str, node_id: str) -> tuple[int, int]:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        sat = conn.execute(
            "SELECT COUNT(*) FROM fleet_satellites WHERE id = ?", (node_id,)
        ).fetchone()[0]
        blobs = conn.execute(
            "SELECT COUNT(*) FROM satellite_blobs WHERE satellite_node_id = ?",
            (node_id,),
        ).fetchone()[0]
        return sat, blobs
    finally:
        conn.close()


def _force_audits(data_dir: str) -> list[dict]:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM audit_events WHERE event_type = 'storage.force_remove'"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


class TestForceRemove:
    def test_counts_lost_blobs_deletes_rows_emits_audit(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        _seed_satellite_doc(data_dir, "node-a", "a" * 64, 100, filename="f0.pdf")
        _seed_satellite_doc(data_dir, "node-a", "b" * 64, 250, filename="f1.pdf")

        result = _drainer(data_dir).force_remove("node-a")

        assert result.satellite_existed is True
        assert result.lost_blobs == 2
        assert result.lost_bytes == 350
        # fleet_satellites + satellite_blobs rows gone.
        sat, blobs = _row_counts(data_dir, "node-a")
        assert sat == 0 and blobs == 0
        # Single audit row with lost counts.
        audits = _force_audits(data_dir)
        assert len(audits) == 1
        meta = json.loads(audits[0]["metadata_json"])
        assert meta == {"node": "node-a", "lost_blobs": 2, "lost_bytes": 350}

    def test_idempotent_on_unknown_node(self, data_dir):
        result = _drainer(data_dir).force_remove("ghost")
        assert result.satellite_existed is False
        assert result.lost_blobs == 0
        # No audit row emitted (nothing happened).
        assert _force_audits(data_dir) == []

    def test_preserves_other_satellites(self, data_dir):
        _register_satellite(data_dir, "node-a", "100.64.0.9")
        _register_satellite(data_dir, "node-b", "100.64.0.10")
        _seed_satellite_doc(data_dir, "node-a", "a" * 64, 10, filename="a.pdf")
        _seed_satellite_doc(data_dir, "node-b", "b" * 64, 20, filename="b.pdf")

        _drainer(data_dir).force_remove("node-a")

        # node-a gone, node-b intact.
        a_sat, a_blobs = _row_counts(data_dir, "node-a")
        b_sat, b_blobs = _row_counts(data_dir, "node-b")
        assert a_sat == 0 and a_blobs == 0
        assert b_sat == 1 and b_blobs == 1
