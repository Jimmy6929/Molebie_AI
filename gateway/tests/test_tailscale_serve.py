"""Unit tests for the three subprocess helpers in tailscale_serve.

All tests monkeypatch ``subprocess.run`` to return synthetic outputs so
the test never invokes a real ``tailscale`` binary. ``_find_tailscale_cli``
is also stubbed so the helper functions proceed past the binary-discovery
step (one test inverts this to verify the missing-binary path).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services import tailscale_serve as ts_serve


@pytest.fixture(autouse=True)
def _fake_cli(monkeypatch):
    """Pretend the tailscale CLI is installed at a known path."""
    monkeypatch.setattr(ts_serve, "_find_tailscale_cli", lambda: "/usr/bin/tailscale")


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )
    return _run


# ─────────────────────────── is_serve_configured ───────────────────────────


class TestIsServeConfigured:
    def test_returns_true_when_route_matches(self, monkeypatch):
        payload = json.dumps({
            "Web": {
                "host.tailnet.ts.net:443": {
                    "Handlers": {
                        "/": {"Proxy": "http://127.0.0.1:8000"},
                    },
                },
            },
        })
        monkeypatch.setattr(subprocess, "run", _completed(stdout=payload))
        assert ts_serve.is_serve_configured(port=8000) is True

    def test_returns_false_when_no_matching_proxy(self, monkeypatch):
        payload = json.dumps({
            "Web": {
                "host.tailnet.ts.net:443": {
                    "Handlers": {"/": {"Proxy": "http://127.0.0.1:9999"}},
                },
            },
        })
        monkeypatch.setattr(subprocess, "run", _completed(stdout=payload))
        assert ts_serve.is_serve_configured(port=8000) is False

    def test_returns_false_when_web_empty(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _completed(stdout=json.dumps({"Web": {}})))
        assert ts_serve.is_serve_configured() is False

    def test_returns_false_on_non_zero_exit(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _completed(stdout="{}", returncode=1))
        assert ts_serve.is_serve_configured() is False

    def test_returns_false_on_malformed_json(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _completed(stdout="not-json"))
        assert ts_serve.is_serve_configured() is False

    def test_returns_false_on_timeout(self, monkeypatch):
        def _raise(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=a[0], timeout=2.0)
        monkeypatch.setattr(subprocess, "run", _raise)
        assert ts_serve.is_serve_configured() is False

    def test_returns_false_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(ts_serve, "_find_tailscale_cli", lambda: None)
        assert ts_serve.is_serve_configured() is False


# ─────────────────────────── enable_serve ───────────────────────────


class TestEnableServe:
    def test_runs_expected_argv_and_returns_true(self, monkeypatch):
        captured: dict = {}

        def _run(cmd, **kwargs):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _run)
        assert ts_serve.enable_serve(port=8000) is True
        assert captured["cmd"][1:] == [
            "serve", "--bg", "--https=443", "http://127.0.0.1:8000",
        ]

    def test_returns_false_on_non_zero_exit(self, monkeypatch, capsys):
        monkeypatch.setattr(
            subprocess, "run",
            _completed(stderr="HTTPS not enabled for this tailnet", returncode=1),
        )
        assert ts_serve.enable_serve() is False
        captured = capsys.readouterr().out
        assert "HTTPS not enabled" in captured

    def test_returns_false_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(ts_serve, "_find_tailscale_cli", lambda: None)
        assert ts_serve.enable_serve() is False

    def test_returns_false_on_timeout(self, monkeypatch):
        def _raise(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=a[0], timeout=10.0)
        monkeypatch.setattr(subprocess, "run", _raise)
        assert ts_serve.enable_serve() is False


# ─────────────────────────── get_https_url ───────────────────────────


class TestGetHttpsUrl:
    def test_parses_self_dnsname(self, monkeypatch):
        payload = json.dumps({"Self": {"DNSName": "myhost.tailnet-abc.ts.net."}})
        monkeypatch.setattr(subprocess, "run", _completed(stdout=payload))
        assert ts_serve.get_https_url() == "https://myhost.tailnet-abc.ts.net/"

    def test_strips_trailing_dot(self, monkeypatch):
        payload = json.dumps({"Self": {"DNSName": "h.t.ts.net"}})  # no trailing dot
        monkeypatch.setattr(subprocess, "run", _completed(stdout=payload))
        assert ts_serve.get_https_url() == "https://h.t.ts.net/"

    def test_returns_none_when_self_missing(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _completed(stdout=json.dumps({"Peer": {}})))
        assert ts_serve.get_https_url() is None

    def test_returns_none_on_non_zero_exit(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _completed(stdout="{}", returncode=1))
        assert ts_serve.get_https_url() is None

    def test_returns_none_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(ts_serve, "_find_tailscale_cli", lambda: None)
        assert ts_serve.get_https_url() is None

    def test_returns_none_when_dnsname_empty(self, monkeypatch):
        payload = json.dumps({"Self": {"DNSName": ""}})
        monkeypatch.setattr(subprocess, "run", _completed(stdout=payload))
        assert ts_serve.get_https_url() is None
