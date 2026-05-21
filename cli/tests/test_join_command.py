"""Tests for ``molebie-ai join`` — the satellite registration command.

Mocks the four external surfaces:
* ``get_tailscale_whoami()`` — local Tailscale identity
* ``get_tailscale_ip()`` — local Tailscale IPv4
* ``httpx.get`` / ``httpx.post`` — the calls to the primary
* ``config_manager.save_config`` — disk write

Mirrors the monkeypatch-only style of ``test_install_wizard.py``: no
``CliRunner``, no real network, no filesystem mutation. The command
function is invoked directly with kwargs that match its Typer parameter
list.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest
import typer

from cli.commands import join as join_module
from cli.commands.join import join
from cli.models.config import (
    MolebieConfig,
    SatelliteNode,
    SatelliteRole,
    SatelliteStatus,
)
from cli.services.network_info import TailscaleWhoamiInfo


@dataclass
class _FakeResponse:
    """Minimal stand-in for httpx.Response in tests."""
    status_code: int
    _json: dict | None = None
    text: str = ""

    def json(self) -> dict:
        if self._json is None:
            raise ValueError("no JSON body")
        return self._json


def _ok_register_response(host: str = "100.64.0.5") -> _FakeResponse:
    return _FakeResponse(
        status_code=200,
        _json={
            "id": "00000000-aaaa-bbbb-cccc-000000000001",
            "host": host,
            "role": "storage",
            "status": "active",
            "label": None,
            "capabilities": {},
            "joined_at": "2026-05-21T16:00:00+00:00",
            "updated_at": "2026-05-21T16:00:00+00:00",
        },
    )


@pytest.fixture
def patched(monkeypatch):
    """Replace the four external surfaces with controllable fakes.

    Tests further override individual fakes as needed. Default state: all
    happy-path. Returns a dict of name → list-of-calls so tests can assert
    invocation behavior.
    """
    calls: dict[str, list] = {
        "get": [], "post": [], "save_config": [],
        "ask_choice": [], "ask_text": [],
    }

    # Local identity + IP
    monkeypatch.setattr(
        join_module, "get_tailscale_whoami",
        lambda: TailscaleWhoamiInfo(user_login="jimmy@github", display_name="Jimmy"),
    )
    monkeypatch.setattr(join_module, "get_tailscale_ip", lambda: "100.64.0.5")

    # httpx
    def _fake_get(url, *args, **kwargs):
        calls["get"].append((url, args, kwargs))
        return _FakeResponse(status_code=200)
    def _fake_post(url, *args, **kwargs):
        calls["post"].append((url, args, kwargs))
        return _ok_register_response()
    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", _fake_post)

    # Config — start with an empty config, capture save_config calls
    saved_config = MolebieConfig()
    monkeypatch.setattr(join_module.config_manager, "load_config", lambda: saved_config)
    def _fake_save(cfg):
        calls["save_config"].append(cfg)
    monkeypatch.setattr(join_module.config_manager, "save_config", _fake_save)

    # Prompts — default behavior: should not be called when flags are
    # passed. Tests that DO want prompts override these.
    def _ask_choice(*args, **kwargs):
        calls["ask_choice"].append((args, kwargs))
        return "storage"
    def _ask_text(*args, **kwargs):
        calls["ask_text"].append((args, kwargs))
        return ""
    monkeypatch.setattr(join_module, "ask_choice", _ask_choice)
    monkeypatch.setattr(join_module, "ask_text", _ask_text)

    return {"calls": calls, "saved_config": saved_config}


class TestJoinCommand:
    def test_successful_join_writes_config(self, patched):
        join(primary_host="100.103.5.7", port=8000, role="storage", label="NAS")

        cfg = patched["saved_config"]
        assert len(cfg.satellites) == 1
        sat = cfg.satellites[0]
        assert isinstance(sat, SatelliteNode)
        assert sat.host == "100.64.0.5"
        assert sat.role == SatelliteRole.STORAGE
        assert sat.status == SatelliteStatus.ACTIVE
        assert len(patched["calls"]["save_config"]) == 1

    def test_flags_bypass_prompts(self, patched):
        join(primary_host="100.103.5.7", port=8000, role="compute", label="GPU box")
        # Neither prompt was called because flags were supplied.
        assert patched["calls"]["ask_choice"] == []
        assert patched["calls"]["ask_text"] == []

    def test_post_sends_identity_header_and_payload(self, patched):
        join(primary_host="100.103.5.7", port=8000, role="storage", label="NAS")
        # Exactly one POST was made.
        assert len(patched["calls"]["post"]) == 1
        url, _args, kwargs = patched["calls"]["post"][0]
        assert url == "http://100.103.5.7:8000/fleet/satellites/register"
        assert kwargs["headers"] == {"Tailscale-User-Login": "jimmy@github"}
        assert kwargs["json"]["host"] == "100.64.0.5"
        assert kwargs["json"]["role"] == "storage"
        assert kwargs["json"]["label"] == "NAS"

    def test_missing_identity_exits_without_calls(self, patched, monkeypatch):
        monkeypatch.setattr(join_module, "get_tailscale_whoami", lambda: None)
        with pytest.raises(typer.Exit) as exc:
            join(primary_host="100.103.5.7", port=8000, role="storage", label=None)
        assert exc.value.exit_code == 1
        assert patched["calls"]["get"] == []
        assert patched["calls"]["post"] == []
        assert patched["calls"]["save_config"] == []

    def test_reachability_failure_exits_without_posting(self, patched, monkeypatch):
        def _bad_get(*args, **kwargs):
            raise httpx.ConnectError("nope")
        monkeypatch.setattr(httpx, "get", _bad_get)
        with pytest.raises(typer.Exit) as exc:
            join(primary_host="100.103.5.7", port=8000, role="storage", label=None)
        assert exc.value.exit_code == 1
        assert patched["calls"]["post"] == []
        assert patched["calls"]["save_config"] == []

    def test_unauthorized_response_exits(self, patched, monkeypatch):
        monkeypatch.setattr(
            httpx, "post",
            lambda *a, **kw: _FakeResponse(status_code=401, text="unauthorized"),
        )
        with pytest.raises(typer.Exit) as exc:
            join(primary_host="100.103.5.7", port=8000, role="storage", label=None)
        assert exc.value.exit_code == 1
        assert patched["calls"]["save_config"] == []

    def test_validation_error_response_exits(self, patched, monkeypatch):
        def _bad_post(*a, **kw):
            return _FakeResponse(
                status_code=422,
                _json={"detail": [{"loc": ["body", "role"], "msg": "Invalid role"}]},
            )
        monkeypatch.setattr(httpx, "post", _bad_post)
        with pytest.raises(typer.Exit) as exc:
            join(primary_host="100.103.5.7", port=8000, role="storage", label=None)
        assert exc.value.exit_code == 1
        assert patched["calls"]["save_config"] == []

    def test_rejoin_replaces_in_place(self, patched, monkeypatch):
        # Seed config with an existing satellite for the same host the
        # primary will return. The join should REPLACE in place.
        existing = SatelliteNode(
            host="100.64.0.5",
            role=SatelliteRole.COMPUTE,
            joined_at="2026-05-20T00:00:00+00:00",
            label="old label",
        )
        seeded = MolebieConfig(satellites=[existing])
        monkeypatch.setattr(join_module.config_manager, "load_config", lambda: seeded)

        # Capture the final saved config separately.
        saved: list[MolebieConfig] = []
        monkeypatch.setattr(join_module.config_manager, "save_config", saved.append)

        join(primary_host="100.103.5.7", port=8000, role="storage", label="NAS")

        assert len(saved) == 1
        cfg = saved[0]
        assert len(cfg.satellites) == 1  # not duplicated
        assert cfg.satellites[0].role == SatelliteRole.STORAGE  # replaced

    def test_local_tailscale_ip_missing_exits(self, patched, monkeypatch):
        monkeypatch.setattr(join_module, "get_tailscale_ip", lambda: None)
        with pytest.raises(typer.Exit) as exc:
            join(primary_host="100.103.5.7", port=8000, role="storage", label=None)
        assert exc.value.exit_code == 1
        assert patched["calls"]["post"] == []

    def test_invalid_role_flag_exits(self, patched):
        with pytest.raises(typer.Exit) as exc:
            join(primary_host="100.103.5.7", port=8000, role="invalid-role", label=None)
        assert exc.value.exit_code == 1
        assert patched["calls"]["post"] == []
