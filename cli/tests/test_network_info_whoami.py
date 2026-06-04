"""Tests for ``get_tailscale_whoami()`` — the local Tailscale identity reader.

Exercises the parser via monkeypatching ``subprocess.run`` so we can return
synthetic outputs for each edge case. Never invokes a real ``tailscale``
binary. Also patches ``_find_tailscale_cli`` to a sentinel value so the
helper proceeds to the subprocess step in every test.

Parses ``tailscale status --json`` (the ``whoami`` subcommand does not
exist in the production Tailscale CLI; this regressed silently because the
old tests mocked the subprocess. Caught by a real-hardware smoke test).
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


def _status_payload(
    *,
    user_id: int = 249207961229939,
    login: str | None = "jimmy@github",
    display: str | None = "Jimmy Z",
    backend_state: str = "Running",
    include_self: bool = True,
    include_user: bool = True,
) -> str:
    """Construct a ``tailscale status --json`` payload matching the
    real-CLI shape we saw in the smoke test."""
    payload: dict = {"BackendState": backend_state}
    if include_self:
        payload["Self"] = {"UserID": user_id, "HostName": "test-host"}
    if include_user and login is not None:
        user_profile: dict = {"ID": user_id, "LoginName": login}
        if display is not None:
            user_profile["DisplayName"] = display
        payload["User"] = {str(user_id): user_profile}
    elif include_user:
        payload["User"] = {str(user_id): {"ID": user_id}}
    return json.dumps(payload)


class TestGetTailscaleWhoami:
    def test_valid_payload_returns_identity(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_run(_status_payload()))
        info = get_tailscale_whoami()
        assert info == TailscaleWhoamiInfo(user_login="jimmy@github", display_name="Jimmy Z")

    def test_display_name_absent_is_none(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            _fake_run(_status_payload(login="alice@example.com", display=None)),
        )
        info = get_tailscale_whoami()
        assert info is not None
        assert info.user_login == "alice@example.com"
        assert info.display_name is None

    def test_backend_not_running_returns_none(self, monkeypatch):
        """Daemon stopped or user not authenticated — should fail to return
        an identity rather than hand the caller a stale login."""
        monkeypatch.setattr(
            subprocess, "run", _fake_run(_status_payload(backend_state="Stopped")),
        )
        assert get_tailscale_whoami() is None

    def test_missing_self_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", _fake_run(_status_payload(include_self=False)),
        )
        assert get_tailscale_whoami() is None

    def test_missing_user_dict_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", _fake_run(_status_payload(include_user=False)),
        )
        assert get_tailscale_whoami() is None

    def test_missing_login_name_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", _fake_run(_status_payload(login=None)),
        )
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
