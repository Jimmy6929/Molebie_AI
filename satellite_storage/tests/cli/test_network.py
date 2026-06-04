"""Tests for ``satellite_storage.cli.network`` — Tailscale CLI discovery
and identity parsing. Mirrors the gateway's ``test_tailscale_outbound.py``
shape but adds the cross-platform fallback-path cases.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from satellite_storage.cli import network


@pytest.fixture(autouse=True)
def _clear_cache():
    """Tailscale lookups are ``@cache``d — wipe between tests."""
    network.get_tailscale_whoami.cache_clear()
    network.get_tailscale_ip.cache_clear()
    yield
    network.get_tailscale_whoami.cache_clear()
    network.get_tailscale_ip.cache_clear()


def _fake_run(stdout: str, returncode: int = 0):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [], returncode=returncode, stdout=stdout, stderr=""
        )
    return _run


def _status_payload(
    *, user_id: int = 42, login: str | None = "ops@example.com",
    display: str | None = "Ops", backend_state: str = "Running",
) -> str:
    payload: dict = {
        "BackendState": backend_state,
        "Self": {"UserID": user_id, "HostName": "test"},
    }
    if login is not None:
        profile: dict = {"ID": user_id, "LoginName": login}
        if display is not None:
            profile["DisplayName"] = display
        payload["User"] = {str(user_id): profile}
    else:
        payload["User"] = {str(user_id): {"ID": user_id}}
    return json.dumps(payload)


# ─────────────────────── _find_tailscale_cli ───────────────────────


class TestFindCli:
    def test_path_wins_first(self, monkeypatch):
        monkeypatch.setattr(network.shutil, "which", lambda _name: "/usr/bin/tailscale")
        assert network._find_tailscale_cli() == "/usr/bin/tailscale"

    def test_macos_fallback_used_when_not_on_path(self, monkeypatch):
        monkeypatch.setattr(network.sys, "platform", "darwin")
        monkeypatch.setattr(network.shutil, "which", lambda _name: None)
        monkeypatch.setattr(network.Path, "exists", lambda self: True)
        result = network._find_tailscale_cli()
        assert result == "/Applications/Tailscale.app/Contents/MacOS/Tailscale"

    def test_windows_fallback_used_when_not_on_path(self, monkeypatch):
        monkeypatch.setattr(network.sys, "platform", "win32")
        monkeypatch.setattr(network.shutil, "which", lambda _name: None)
        # The first Windows fallback path is `C:\Program Files\Tailscale\tailscale.exe`.
        monkeypatch.setattr(
            network.Path, "exists",
            lambda self: str(self).endswith("Tailscale\\tailscale.exe"),
        )
        result = network._find_tailscale_cli()
        assert result and result.endswith("Tailscale\\tailscale.exe")

    def test_returns_none_when_not_found_anywhere(self, monkeypatch):
        monkeypatch.setattr(network.shutil, "which", lambda _name: None)
        monkeypatch.setattr(network.Path, "exists", lambda self: False)
        assert network._find_tailscale_cli() is None


# ─────────────────────── get_tailscale_whoami ───────────────────────


class TestWhoami:
    def test_happy_path(self, monkeypatch):
        monkeypatch.setattr(network, "_find_tailscale_cli", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(subprocess, "run", _fake_run(_status_payload()))
        info = network.get_tailscale_whoami()
        assert info is not None
        assert info.user_login == "ops@example.com"
        assert info.display_name == "Ops"

    def test_backend_not_running_returns_none(self, monkeypatch):
        monkeypatch.setattr(network, "_find_tailscale_cli", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(
            subprocess, "run",
            _fake_run(_status_payload(backend_state="Stopped")),
        )
        assert network.get_tailscale_whoami() is None

    def test_missing_login_returns_none(self, monkeypatch):
        monkeypatch.setattr(network, "_find_tailscale_cli", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(subprocess, "run", _fake_run(_status_payload(login=None)))
        assert network.get_tailscale_whoami() is None

    def test_binary_missing_returns_none(self, monkeypatch):
        monkeypatch.setattr(network, "_find_tailscale_cli", lambda: None)
        assert network.get_tailscale_whoami() is None


# ─────────────────────── get_tailscale_ip ───────────────────────


class TestIp:
    def test_returns_first_non_empty_line(self, monkeypatch):
        monkeypatch.setattr(network, "_find_tailscale_cli", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(subprocess, "run", _fake_run("100.64.0.5\nfd7a::5\n"))
        assert network.get_tailscale_ip() == "100.64.0.5"

    def test_returns_none_on_non_zero_exit(self, monkeypatch):
        monkeypatch.setattr(network, "_find_tailscale_cli", lambda: "/usr/bin/tailscale")
        monkeypatch.setattr(subprocess, "run", _fake_run("", returncode=1))
        assert network.get_tailscale_ip() is None
