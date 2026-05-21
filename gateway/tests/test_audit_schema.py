"""Schema-level tests for the audit_events table.

The migration follows the project's idempotent-additive pattern: run
``init_database_sync`` against any data_dir and the table appears (or
stays) without dropping existing data.
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
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None
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


class TestAuditEventsSchema:
    def test_fresh_init_creates_audit_events_table(self, fresh_data_dir):
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        assert _table_exists(db_path, "audit_events")
        cols = _columns(db_path, "audit_events")
        assert cols["id"] == "INTEGER"
        assert cols["event_type"] == "TEXT"
        assert "actor" in cols
        assert "target" in cols
        assert "metadata_json" in cols
        assert cols["created_at"] == "TEXT"

    def test_fresh_init_creates_indexes(self, fresh_data_dir):
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        idx = _indexes(db_path, "audit_events")
        assert "idx_audit_events_created" in idx
        assert "idx_audit_events_type" in idx

    def test_reinit_is_idempotent(self, fresh_data_dir):
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        # Insert a sentinel row so we can confirm re-init doesn't drop data.
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO audit_events "
                "(event_type, actor, target, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("test.sentinel", "user-x", "node-y", None, "2026-05-21T00:00:00Z"),
            )
            conn.commit()
        finally:
            conn.close()

        # Re-run init — must not raise, must not drop the row.
        init_database_sync(fresh_data_dir, embedding_dim=1024)
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT event_type FROM audit_events WHERE event_type='test.sentinel'"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == "test.sentinel"

    def test_upgrade_from_db_without_audit_events_table(self, fresh_data_dir):
        # Simulate an older install: init the DB, then drop the audit_events
        # table so the next init looks like an upgrade.
        db_path = init_database_sync(fresh_data_dir, embedding_dim=1024)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DROP TABLE audit_events")
            conn.commit()
        finally:
            conn.close()
        assert not _table_exists(db_path, "audit_events")

        # Re-init: migration should add the table back.
        init_database_sync(fresh_data_dir, embedding_dim=1024)
        assert _table_exists(db_path, "audit_events")
