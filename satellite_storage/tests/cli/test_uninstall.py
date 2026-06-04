"""Tests for ``molebie-satellite uninstall`` — service teardown + --purge."""

from __future__ import annotations

import pytest

from satellite_storage.cli import uninstall as uninstall_mod


@pytest.fixture
def fake_data_dir(tmp_path, monkeypatch):
    """Replace ``default_data_dir`` so --purge tests target a temp tree."""
    monkeypatch.setattr(uninstall_mod, "default_data_dir", lambda: tmp_path / "data")
    return tmp_path / "data"


@pytest.fixture
def mock_service(monkeypatch):
    state = {"installed": False, "uninstall_called": False}

    def _is_installed(label=None):
        return state["installed"]

    def _uninstall(label=None):
        state["uninstall_called"] = True
        state["installed"] = False

    monkeypatch.setattr(uninstall_mod, "is_service_installed", _is_installed)
    monkeypatch.setattr(uninstall_mod, "uninstall_service", _uninstall)
    return state


class TestUninstall:
    def test_default_uninstalls_service_only(self, fake_data_dir, mock_service):
        # Service exists, data dir exists, but no --purge.
        mock_service["installed"] = True
        fake_data_dir.mkdir(parents=True, exist_ok=True)
        (fake_data_dir / "blob.bin").write_bytes(b"x")

        uninstall_mod.uninstall_command(purge=False, data_dir=None, yes=True)

        assert mock_service["uninstall_called"] is True
        assert fake_data_dir.exists()  # untouched
        assert (fake_data_dir / "blob.bin").exists()

    def test_purge_with_yes_deletes_data(self, fake_data_dir, mock_service):
        mock_service["installed"] = True
        fake_data_dir.mkdir(parents=True, exist_ok=True)
        (fake_data_dir / "blob.bin").write_bytes(b"x")

        uninstall_mod.uninstall_command(purge=True, data_dir=None, yes=True)

        assert mock_service["uninstall_called"] is True
        assert not fake_data_dir.exists()

    def test_purge_without_yes_aborted_keeps_data(
        self, fake_data_dir, mock_service, monkeypatch
    ):
        mock_service["installed"] = True
        fake_data_dir.mkdir(parents=True, exist_ok=True)
        (fake_data_dir / "blob.bin").write_bytes(b"x")

        # ask_confirm returns False → purge aborted.
        monkeypatch.setattr(uninstall_mod, "ask_confirm", lambda *_a, **_k: False)

        uninstall_mod.uninstall_command(purge=True, data_dir=None, yes=False)

        assert mock_service["uninstall_called"] is True
        assert fake_data_dir.exists()
        assert (fake_data_dir / "blob.bin").exists()

    def test_no_service_installed_is_quiet(self, fake_data_dir, mock_service, capsys):
        # Service NOT installed.
        mock_service["installed"] = False

        uninstall_mod.uninstall_command(purge=False, data_dir=None, yes=True)

        assert mock_service["uninstall_called"] is False
        out = capsys.readouterr().out
        assert "nothing to remove" in out.lower()
