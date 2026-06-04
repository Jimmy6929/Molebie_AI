"""Tests for the outbound Tailscale identity helper.

Parses ``tailscale status --json``; the ``whoami`` subcommand does not
exist in the production Tailscale CLI (the old tests mocked the
subprocess so the regression went undetected until a real-hardware smoke
test). The shape under test mirrors what ``tailscale status --json``
actually returns: ``Self.UserID`` (int) keys into a top-level ``User``
dict whose entries carry ``LoginName``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services import tailscale_outbound
from app.services.tailscale_outbound import get_operator_identity


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(tailscale_outbound, "_find_tailscale_cli", lambda: "/usr/bin/tailscale")
    get_operator_identity.cache_clear()
    yield
    get_operator_identity.cache_clear()


def _fake_run(stdout: str, returncode: int = 0):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [], returncode=returncode, stdout=stdout, stderr=""
        )
    return _run


def _status_payload(
    *, user_id: int = 249207961229939, login: str | None = "jimmy@github",
    backend_state: str = "Running",
) -> str:
    payload: dict = {
        "BackendState": backend_state,
        "Self": {"UserID": user_id, "HostName": "test-host"},
    }
    if login is not None:
        payload["User"] = {str(user_id): {"ID": user_id, "LoginName": login}}
    else:
        payload["User"] = {str(user_id): {"ID": user_id}}
    return json.dumps(payload)


class TestGetOperatorIdentity:
    def test_parses_login_name(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_run(_status_payload()))
        assert get_operator_identity() == "jimmy@github"

    def test_backend_not_running_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", _fake_run(_status_payload(backend_state="NoState")),
        )
        assert get_operator_identity() is None

    def test_returns_none_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(tailscale_outbound, "_find_tailscale_cli", lambda: None)
        assert get_operator_identity() is None

    def test_returns_none_on_non_json(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_run("not json"))
        assert get_operator_identity() is None

    def test_returns_none_when_login_missing(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", _fake_run(_status_payload(login=None)),
        )
        assert get_operator_identity() is None
