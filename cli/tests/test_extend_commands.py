"""Tests for ``molebie-ai extend list/audit/status``.

All three subcommands are read-only HTTP wrappers over the primary's
loopback fleet routes. Tests monkeypatch ``httpx.get`` to return
synthetic responses; assertions cover the table-building logic, empty
states, parameter forwarding, and friendly error paths. No real network,
no filesystem mutation. Style mirrors ``test_join_command.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import typer

from cli.commands import extend as extend_module


@dataclass
class _FakeResponse:
    status_code: int
    _json: dict[str, Any] | None = None
    text: str = ""

    def json(self) -> dict[str, Any]:
        if self._json is None:
            raise ValueError("no JSON body")
        return self._json


def _inventory_response(satellites: list[dict[str, Any]]) -> _FakeResponse:
    return _FakeResponse(
        status_code=200,
        _json={"satellites": satellites, "count": len(satellites)},
    )


def _audit_response(events: list[dict[str, Any]]) -> _FakeResponse:
    return _FakeResponse(
        status_code=200,
        _json={"events": events, "count": len(events)},
    )


def _make_satellite(
    *,
    sat_id: str = "11111111-aaaa-bbbb-cccc-222222222222",
    host: str = "100.64.0.5",
    role: str = "storage",
    status: str = "active",
    label: str | None = None,
    joined_at: str = "2026-05-21T16:00:00+00:00",
) -> dict[str, Any]:
    return {
        "id": sat_id,
        "host": host,
        "role": role,
        "status": status,
        "label": label,
        "capabilities": None,
        "joined_at": joined_at,
        "updated_at": joined_at,
    }


def _make_event(
    *,
    event_type: str = "satellite.join",
    actor: str = "jimmy@github",
    target: str = "100.64.0.5",
    metadata: dict[str, Any] | None = None,
    created_at: str = "2026-05-21T16:30:00+00:00",
    event_id: int = 1,
) -> dict[str, Any]:
    return {
        "id": event_id,
        "event_type": event_type,
        "actor": actor,
        "target": target,
        "metadata": metadata,
        "created_at": created_at,
    }


@pytest.fixture
def gateway(monkeypatch):
    """Programmable httpx.get fake.

    Returns a dict the test can mutate to control the response per URL:
    `gateway["routes"][path] = _FakeResponse(...)`. Captures every call
    in `gateway["calls"]` as (url, params).
    """
    state: dict[str, Any] = {
        "routes": {},  # path → _FakeResponse OR Exception
        "calls": [],   # list[(url, params)]
    }

    def _fake_get(url, params=None, timeout=None, **_kwargs):
        state["calls"].append((url, params))
        # Find the matching route by path-suffix.
        for path, resp in state["routes"].items():
            if url.endswith(path):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        # Default: pretend the path doesn't exist.
        return _FakeResponse(status_code=404, text="not found")

    monkeypatch.setattr(httpx, "get", _fake_get)
    return state


# ─────────────────────────── extend list ───────────────────────────


class TestExtendList:
    def test_lists_satellites_in_table(self, gateway, capsys):
        gateway["routes"]["/fleet/inventory"] = _inventory_response([
            _make_satellite(host="home-server", role="storage", label="NAS"),
            _make_satellite(
                sat_id="33333333-dddd-eeee-ffff-444444444444",
                host="gpu-box", role="compute", status="degraded",
            ),
        ])
        extend_module.list_satellites()
        captured = capsys.readouterr().out
        assert "home-server" in captured
        assert "gpu-box" in captured
        assert "storage" in captured
        assert "compute" in captured
        assert "NAS" in captured
        assert "2 satellite(s) registered" in captured

    def test_empty_fleet_friendly_message(self, gateway, capsys):
        gateway["routes"]["/fleet/inventory"] = _inventory_response([])
        extend_module.list_satellites()
        captured = capsys.readouterr().out
        assert "No satellites registered" in captured
        # No table header rendered.
        assert "Host" not in captured.split("No satellites")[0] or "Host" not in captured

    def test_gateway_unreachable_exits(self, gateway):
        gateway["routes"]["/fleet/inventory"] = httpx.ConnectError("nope")
        with pytest.raises(typer.Exit) as exc:
            extend_module.list_satellites()
        assert exc.value.exit_code == 1

    def test_non_2xx_response_exits(self, gateway):
        gateway["routes"]["/fleet/inventory"] = _FakeResponse(
            status_code=500, text="boom"
        )
        with pytest.raises(typer.Exit) as exc:
            extend_module.list_satellites()
        assert exc.value.exit_code == 1


# ─────────────────────────── extend audit ───────────────────────────


class TestExtendAudit:
    def test_lists_events_in_table(self, gateway, capsys):
        gateway["routes"]["/fleet/audit"] = _audit_response([
            _make_event(event_type="satellite.join", target="home-server"),
            _make_event(
                event_type="satellite.update", target="home-server",
                event_id=2, created_at="2026-05-21T17:00:00+00:00",
                metadata={"role": "both"},
            ),
        ])
        extend_module.audit_log(limit=50, event_type=None, since=None)
        captured = capsys.readouterr().out
        assert "satellite.join" in captured
        assert "satellite.update" in captured
        assert "2 event(s) shown" in captured

    def test_event_type_and_since_forwarded(self, gateway):
        gateway["routes"]["/fleet/audit"] = _audit_response([])
        extend_module.audit_log(
            limit=10, event_type="satellite.join", since="2026-05-01T00:00:00Z",
        )
        # Inspect the GET call.
        assert len(gateway["calls"]) == 1
        _url, params = gateway["calls"][0]
        assert params["limit"] == 10
        assert params["event_type"] == "satellite.join"
        assert params["since"] == "2026-05-01T00:00:00Z"

    def test_no_filter_means_no_extra_params(self, gateway):
        gateway["routes"]["/fleet/audit"] = _audit_response([])
        extend_module.audit_log(limit=25, event_type=None, since=None)
        _url, params = gateway["calls"][0]
        assert params == {"limit": 25}

    def test_empty_friendly_message(self, gateway, capsys):
        gateway["routes"]["/fleet/audit"] = _audit_response([])
        extend_module.audit_log(limit=50, event_type=None, since=None)
        captured = capsys.readouterr().out
        assert "No audit events" in captured

    def test_limit_out_of_range_exits(self, gateway):
        with pytest.raises(typer.Exit) as exc:
            extend_module.audit_log(limit=1000, event_type=None, since=None)
        assert exc.value.exit_code == 1
        with pytest.raises(typer.Exit):
            extend_module.audit_log(limit=0, event_type=None, since=None)


# ─────────────────────────── extend status ───────────────────────────


class TestExtendStatus:
    def test_aggregates_by_role_and_status(self, gateway, capsys):
        sats = [
            _make_satellite(host="a", role="storage", status="active"),
            _make_satellite(host="b", role="storage", status="active"),
            _make_satellite(host="c", role="compute", status="degraded"),
        ]
        gateway["routes"]["/fleet/inventory"] = _inventory_response(sats)
        gateway["routes"]["/fleet/audit"] = _audit_response([
            _make_event(event_type="satellite.join", target="a"),
        ])
        extend_module.fleet_status()
        captured = capsys.readouterr().out
        # The summary line includes "3" total + the per-role + per-status breakdowns.
        assert "3" in captured
        assert "2 storage" in captured
        assert "1 compute" in captured
        assert "active" in captured
        assert "degraded" in captured

    def test_shows_recent_events(self, gateway, capsys):
        gateway["routes"]["/fleet/inventory"] = _inventory_response([
            _make_satellite(host="a"),
        ])
        gateway["routes"]["/fleet/audit"] = _audit_response([
            _make_event(event_type="satellite.update", target="a", event_id=2),
            _make_event(event_type="satellite.join", target="a", event_id=1),
        ])
        extend_module.fleet_status()
        captured = capsys.readouterr().out
        assert "Recent events" in captured
        assert "satellite.update" in captured
        assert "satellite.join" in captured

    def test_empty_fleet_friendly(self, gateway, capsys):
        gateway["routes"]["/fleet/inventory"] = _inventory_response([])
        gateway["routes"]["/fleet/audit"] = _audit_response([])
        extend_module.fleet_status()
        captured = capsys.readouterr().out
        assert "Fleet is empty" in captured

    def test_gateway_unreachable_exits(self, gateway):
        gateway["routes"]["/fleet/inventory"] = httpx.ConnectError("nope")
        with pytest.raises(typer.Exit) as exc:
            extend_module.fleet_status()
        assert exc.value.exit_code == 1
