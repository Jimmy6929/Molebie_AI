"""Tests for deep_checker's sqlite-vec verdict.

Regression guard: a Python built without loadable SQLite extension support must
be reported as a FAILURE by `doctor --deep`, not green-lit. The gateway runs the
same venv, so if the extension can't load here it can't load there either —
the previous "OK, gateway handles it" verdict masked a startup crash.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cli.services import deep_checker
from cli.services.deep_checker import _extension_fix_hint, check_sqlite_vec


def _make_db_with_vec_tables(tmp_path: Path) -> Path:
    """Create a DB whose sqlite_master.sql mentions vec0 for the expected tables.

    check_sqlite_vec only inspects sqlite_master for the vec0 substring before the
    extension-load test, so plain tables with 'vec0' in their DDL are sufficient to
    reach that branch without needing the extension itself.
    """
    db_path = tmp_path / "molebie.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE document_chunks_vec (vec0col BLOB)")
    conn.execute("CREATE TABLE user_memories_vec (vec0col BLOB)")
    conn.commit()
    conn.close()
    return db_path


def test_incapable_python_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """can_load is None (no loadable-extension support) → hard FAIL with a fix hint."""
    db_path = _make_db_with_vec_tables(tmp_path)
    monkeypatch.setattr(deep_checker, "_try_load_sqlite_vec", lambda _p: None)

    result = check_sqlite_vec(db_path)

    assert result.passed is False, "an extension-less Python must FAIL the sqlite-vec check"
    assert "extension" in result.message.lower()
    assert result.fix_hint, "a failure must carry actionable remediation"
    assert "install.sh" in result.fix_hint


def test_capable_python_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """can_load is True → pass (sanity that the change didn't break the happy path)."""
    db_path = _make_db_with_vec_tables(tmp_path)
    monkeypatch.setattr(deep_checker, "_try_load_sqlite_vec", lambda _p: True)

    result = check_sqlite_vec(db_path)

    assert result.passed is True


def test_extension_fix_hint_is_actionable() -> None:
    """The hint always tells the user how to rebuild, regardless of package manager."""
    hint = _extension_fix_hint()
    assert "install.sh" in hint
    assert hint.strip()
