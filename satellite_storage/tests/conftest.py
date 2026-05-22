"""Shared fixtures for satellite_storage tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Make the repo root importable so `from satellite_storage...` works
# when pytest is run with `pytest tests/satellite_storage/` from root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from satellite_storage.config import get_settings


@pytest.fixture
def tempdir_data_dir(monkeypatch):
    """Point Settings.data_dir at a fresh tempdir for the test, then clean up.

    Clears the cached Settings instance so each test sees its own dir.
    """
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("MOLEBIE_STORAGE_DATA_DIR", td)
        get_settings.cache_clear()
        yield Path(td)
        get_settings.cache_clear()
