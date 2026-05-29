"""Tests for the outbound Tailscale identity helper."""

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


class TestGetOperatorIdentity:
    def test_parses_login_name(self, monkeypatch):
        payload = json.dumps({"UserProfile": {"LoginName": "jimmy@github"}})
        monkeypatch.setattr(subprocess, "run", _fake_run(payload))
        assert get_operator_identity() == "jimmy@github"

    def test_returns_none_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(tailscale_outbound, "_find_tailscale_cli", lambda: None)
        assert get_operator_identity() is None

    def test_returns_none_on_non_json(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_run("not json"))
        assert get_operator_identity() is None

    def test_returns_none_when_login_missing(self, monkeypatch):
        payload = json.dumps({"UserProfile": {"DisplayName": "x"}})
        monkeypatch.setattr(subprocess, "run", _fake_run(payload))
        assert get_operator_identity() is None
