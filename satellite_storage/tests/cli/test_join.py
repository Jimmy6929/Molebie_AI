"""Tests for ``molebie-satellite join`` — one-shot register."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest
import typer

from satellite_storage.cli import join as join_mod
from satellite_storage.cli.network import TailscaleWhoamiInfo


@dataclass
class _FakeResp:
    status_code: int
    text: str = ""
    _json: dict | None = None

    def json(self):
        if self._json is None:
            raise ValueError("no JSON")
        return self._json


@pytest.fixture
def good_identity(monkeypatch):
    monkeypatch.setattr(
        join_mod, "get_tailscale_whoami",
        lambda: TailscaleWhoamiInfo("ops@example.com", "Ops"),
    )
    monkeypatch.setattr(join_mod, "get_tailscale_ip", lambda: "100.64.0.5")


def _route(get_resp=None, post_resp=None):
    """Returns a (get_fn, post_fn) pair for monkeypatching httpx."""
    def _get(url, **kwargs):
        if isinstance(get_resp, Exception):
            raise get_resp
        return get_resp or _FakeResp(404, text="not configured")

    def _post(url, **kwargs):
        if isinstance(post_resp, Exception):
            raise post_resp
        return post_resp or _FakeResp(404, text="not configured")

    return _get, _post


class TestJoin:
    def test_happy_path(self, good_identity, monkeypatch, capsys):
        get_fn, post_fn = _route(
            get_resp=_FakeResp(200),
            post_resp=_FakeResp(
                200, _json={"id": "abc1234567890", "host": "100.64.0.5", "role": "storage"},
            ),
        )
        monkeypatch.setattr(httpx, "get", get_fn)
        monkeypatch.setattr(httpx, "post", post_fn)

        join_mod.join_command(primary="100.64.0.1", port=8000, role="storage", label=None)

        out = capsys.readouterr().out
        assert "Registered as" in out

    def test_no_identity_exits(self, monkeypatch):
        monkeypatch.setattr(join_mod, "get_tailscale_whoami", lambda: None)
        with pytest.raises(typer.Exit) as exc:
            join_mod.join_command(
                primary="100.64.0.1", port=8000, role="storage", label=None,
            )
        assert exc.value.exit_code == 1

    def test_primary_unreachable_exits(self, good_identity, monkeypatch):
        get_fn, _ = _route(get_resp=httpx.ConnectError("nope"))
        monkeypatch.setattr(httpx, "get", get_fn)
        with pytest.raises(typer.Exit) as exc:
            join_mod.join_command(
                primary="100.64.0.1", port=8000, role="storage", label=None,
            )
        assert exc.value.exit_code == 1

    def test_register_401_exits_with_hint(self, good_identity, monkeypatch, capsys):
        get_fn, post_fn = _route(
            get_resp=_FakeResp(200),
            post_resp=_FakeResp(401, text="missing header"),
        )
        monkeypatch.setattr(httpx, "get", get_fn)
        monkeypatch.setattr(httpx, "post", post_fn)
        with pytest.raises(typer.Exit) as exc:
            join_mod.join_command(
                primary="100.64.0.1", port=8000, role="storage", label=None,
            )
        assert exc.value.exit_code == 1
        out = capsys.readouterr().out
        assert "401" in out

    def test_invalid_role_exits(self, good_identity):
        with pytest.raises(typer.Exit) as exc:
            join_mod.join_command(
                primary="100.64.0.1", port=8000, role="not-real", label=None,
            )
        assert exc.value.exit_code == 1
