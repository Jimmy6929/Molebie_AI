"""Schema-level tests for the ``satellite_blobs`` table.

Mirrors the shape of ``test_fleet_satellites_schema.py`` (PR #47):
idempotent additive migration; re-running ``init_database_sync`` is safe
and preserves data; upgrades from a DB missing the table work.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schema import init_database_sync


@pytest.fixture
def fresh_data_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


def _table_exists(db_path: str, table: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone() is not None
    finally:
        conn.close()


def _columns(db_path: str, table: str) -> dict[str, str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1]: row[2] for row in rows}
    finally:
        conn.close()


def _indexes(db_path: str, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (table,),
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


class TestSatelliteBlobsSchema:
    def test_fresh_init_creates_table(self, fresh_data_dir):
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        assert _table_exists(db_path, "satellite_blobs")
        cols = _columns(db_path, "satellite_blobs")
        for expected in (
            "sha256", "satellite_node_id", "blob_type",
            "size_bytes", "uploaded_at", "last_verified_at",
        ):
            assert expected in cols, f"column {expected!r} missing"

    def test_indexes_created(self, fresh_data_dir):
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        idx = _indexes(db_path, "satellite_blobs")
        assert "idx_satellite_blobs_node" in idx
        assert "idx_satellite_blobs_type" in idx

    def test_reinit_is_idempotent(self, fresh_data_dir):
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        # Plant a sentinel row to confirm re-init doesn't drop data.
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO satellite_blobs "
                "(sha256, satellite_node_id, blob_type, size_bytes, "
                " uploaded_at, last_verified_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "a" * 64, "test-node", "document", 1024,
                    "2026-05-22T00:00:00Z", None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        init_database_sync(fresh_data_dir, embedding_dim=1024)

        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT satellite_node_id FROM satellite_blobs WHERE sha256=?",
                ("a" * 64,),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == "test-node"

    def test_upgrade_from_db_without_table(self, fresh_data_dir):
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DROP TABLE satellite_blobs")
            conn.commit()
        finally:
            conn.close()
        assert not _table_exists(db_path, "satellite_blobs")

        init_database_sync(fresh_data_dir, embedding_dim=1024)
        assert _table_exists(db_path, "satellite_blobs")
