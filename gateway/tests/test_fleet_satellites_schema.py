"""Schema-level tests for the ``fleet_satellites`` table.

Mirrors the audit_events schema-test shape: idempotent additive
migration; re-running ``init_database_sync`` is safe and preserves data.
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


class TestFleetSatellitesSchema:
    def test_fresh_init_creates_table(self, fresh_data_dir):
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        assert _table_exists(db_path, "fleet_satellites")
        cols = _columns(db_path, "fleet_satellites")
        for expected in (
            "id", "host", "role", "status", "label",
            "capabilities_json", "tailscale_user", "joined_at", "updated_at",
        ):
            assert expected in cols, f"column {expected!r} missing"

    def test_indexes_created(self, fresh_data_dir):
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        idx = _indexes(db_path, "fleet_satellites")
        assert "idx_fleet_satellites_role" in idx
        assert "idx_fleet_satellites_status" in idx

    def test_reinit_is_idempotent(self, fresh_data_dir):
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        # Plant a sentinel row.
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO fleet_satellites "
                "(id, host, role, status, label, capabilities_json, "
                " tailscale_user, joined_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "sentinel-id", "host-x", "storage", "active", None, None,
                    "user@example.com", "2026-05-21T00:00:00Z",
                    "2026-05-21T00:00:00Z",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        init_database_sync(fresh_data_dir, embedding_dim=1024)

        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT host FROM fleet_satellites WHERE id='sentinel-id'"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == "host-x"

    def test_upgrade_from_db_without_table(self, fresh_data_dir):
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DROP TABLE fleet_satellites")
            conn.commit()
        finally:
            conn.close()
        assert not _table_exists(db_path, "fleet_satellites")

        init_database_sync(fresh_data_dir, embedding_dim=1024)
        assert _table_exists(db_path, "fleet_satellites")
