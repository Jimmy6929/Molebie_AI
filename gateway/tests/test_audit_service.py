"""Tests for the ``audit.record`` helper.

The helper writes to the same SQLite DB the DatabaseService uses; tests
spin up an isolated data_dir and round-trip a few records, checking the
shape of what landed and the never-raises contract on failure.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync


@pytest.fixture
def isolated_data_dir(monkeypatch):
    """Fresh data_dir + initialised DB per test. Clears the DatabaseService
    singleton so each test gets a clean aiosqlite connection."""
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("DATA_DIR", td)
        get_settings.cache_clear()
        init_database_sync(td, embedding_dim=1024, auth_mode="single")

        from app.services import database
        database._db_service = None  # type: ignore[attr-defined]
        database.get_database_service.cache_clear()

        yield td

        async def _close():
            from app.services.database import get_database_service
            await get_database_service().close()
        asyncio.run(_close())


def _audit_rows(data_dir: str) -> list[dict]:
    """Read audit_events directly via sqlite3 — bypasses the async layer."""
    db = sqlite3.connect(Path(data_dir) / "molebie.db")
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute(
        "SELECT * FROM audit_events ORDER BY id ASC"
    )]
    db.close()
    return rows


class TestAuditRecord:
    def test_record_inserts_row_with_event_type_and_timestamp(self, isolated_data_dir):
        from app.services.audit import record

        asyncio.run(record("storage.upload"))

        rows = _audit_rows(isolated_data_dir)
        assert len(rows) == 1
        row = rows[0]
        assert row["event_type"] == "storage.upload"
        assert row["created_at"]  # non-empty ISO timestamp
        assert row["actor"] is None
        assert row["target"] is None
        assert row["metadata_json"] is None

    def test_record_with_all_fields(self, isolated_data_dir):
        from app.services.audit import record

        asyncio.run(record(
            "routing.fallback",
            actor="local-thinking",
            target="local-instant",
            metadata={"reason": "circuit_open", "consecutive_failures": 3},
        ))

        rows = _audit_rows(isolated_data_dir)
        assert len(rows) == 1
        row = rows[0]
        assert row["event_type"] == "routing.fallback"
        assert row["actor"] == "local-thinking"
        assert row["target"] == "local-instant"
        parsed = json.loads(row["metadata_json"])
        assert parsed == {"reason": "circuit_open", "consecutive_failures": 3}

    def test_metadata_none_stays_null(self, isolated_data_dir):
        from app.services.audit import record

        asyncio.run(record("test.event", metadata=None))
        rows = _audit_rows(isolated_data_dir)
        assert rows[0]["metadata_json"] is None

    def test_record_never_raises_on_db_failure(self, isolated_data_dir, capsys):
        from app.services.audit import record

        # Patch _get_conn to blow up — the helper must swallow it.
        async def _broken_conn(self):  # noqa: ANN001
            raise RuntimeError("simulated DB outage")

        with patch(
            "app.services.database.DatabaseService._get_conn",
            new=_broken_conn,
        ):
            # Must not raise.
            asyncio.run(record("test.event", metadata={"x": 1}))

        # The failure should be visible on stderr / stdout for ops.
        captured = capsys.readouterr()
        assert "[audit] failed to record" in captured.out or "[audit] failed to record" in captured.err

    def test_many_records_all_persist(self, isolated_data_dir):
        from app.services.audit import record

        async def _hammer():
            for i in range(100):
                await record("burst.event", metadata={"i": i})

        asyncio.run(_hammer())
        rows = _audit_rows(isolated_data_dir)
        assert len(rows) == 100
        # IDs are auto-increment and unique.
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == 100
