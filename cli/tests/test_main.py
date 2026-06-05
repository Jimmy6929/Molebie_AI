"""Tests for the top-level `molebie-ai` CLI app wiring."""

from __future__ import annotations

from cli.main import app


def test_join_command_not_registered() -> None:
    """`molebie-ai join` was satellite-side logic stranded in the primary's CLI.

    It was removed in favour of the standalone `molebie-satellite` package. This
    guards against an accidental re-add (e.g. a bad merge resurrecting the import
    + registration in `cli/main.py`).
    """
    names = [c.name for c in app.registered_commands]
    assert "join" not in names


def test_core_commands_still_registered() -> None:
    """Sanity check that removing `join` didn't disturb the other top-level commands."""
    names = {c.name for c in app.registered_commands}
    assert {"install", "run", "doctor", "status", "monitor"} <= names
