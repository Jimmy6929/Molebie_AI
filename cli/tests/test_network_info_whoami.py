"""Tests for ``get_tailscale_whoami()`` — the local Tailscale identity reader.

Exercises the parser via monkeypatching ``subprocess.run`` so we can return
synthetic outputs for each edge case. Never invokes a real `tailscale`
binary. Also patches ``_find_tailscale_cli`` to a sentinel value so the
helper proceeds to the subprocess step in every test.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from cli.services import network_info
from cli.services.network_info import TailscaleWhoamiInfo, get_tailscale_whoami


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Pretend the tailscale CLI exists; clear the lru_cache between tests."""
    monkeypatch.setattr(network_info, "_find_tailscale_cli", lambda: "/usr/bin/tailscale")
    get_tailscale_whoami.cache_clear()
    yield
    get_tailscale_whoami.cache_clear()


def _fake_run(stdout: str, returncode: int = 0):
    """Return a callable that mimics ``subprocess.run``."""
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [], returncode=returncode, stdout=stdout, stderr=""
        )
    return _run


class TestGetTailscaleWhoami:
    def test_valid_payload_returns_identity(self, monkeypatch):
        payload = json.dumps({
            "UserProfile": {
                "LoginName": "jimmy@github",
                "DisplayName": "Jimmy Z",
            },
        })
        monkeypatch.setattr(subprocess, "run", _fake_run(payload))
        info = get_tailscale_whoami()
        assert info == TailscaleWhoamiInfo(user_login="jimmy@github", display_name="Jimmy Z")

    def test_display_name_absent_is_none(self, monkeypatch):
        payload = json.dumps({"UserProfile": {"LoginName": "alice@example.com"}})
        monkeypatch.setattr(subprocess, "run", _fake_run(payload))
        info = get_tailscale_whoami()
        assert info is not None
        assert info.user_login == "alice@example.com"
        assert info.display_name is None

    def test_missing_user_profile_returns_none(self, monkeypatch):
        payload = json.dumps({"Other": {"x": 1}})
        monkeypatch.setattr(subprocess, "run", _fake_run(payload))
        assert get_tailscale_whoami() is None

    def test_missing_login_name_returns_none(self, monkeypatch):
        payload = json.dumps({"UserProfile": {"DisplayName": "no login"}})
        monkeypatch.setattr(subprocess, "run", _fake_run(payload))
        assert get_tailscale_whoami() is None

    def test_non_json_output_returns_none(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_run("not json at all"))
        assert get_tailscale_whoami() is None

    def test_non_zero_exit_returns_none(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_run("{}", returncode=1))
        assert get_tailscale_whoami() is None

    def test_subprocess_timeout_returns_none(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=2.0)
        monkeypatch.setattr(subprocess, "run", _raise)
        assert get_tailscale_whoami() is None

    def test_binary_missing_returns_none(self, monkeypatch):
        monkeypatch.setattr(network_info, "_find_tailscale_cli", lambda: None)
        assert get_tailscale_whoami() is None
