"""Tests for project root discovery in config_manager."""

from __future__ import annotations

from pathlib import Path

from cli.services.config_manager import get_project_root

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cwd_project_takes_precedence(tmp_path, monkeypatch):
    (tmp_path / ".env.example").touch()
    (tmp_path / "gateway").mkdir()
    nested = tmp_path / "webapp" / "src"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert get_project_root() == tmp_path.resolve()


def test_falls_back_to_install_location_outside_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert get_project_root() == REPO_ROOT
