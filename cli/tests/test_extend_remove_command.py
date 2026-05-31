"""Tests for ``molebie-ai extend remove`` — slice 9.6.

Mirrors the test_extend_commands.py pattern: monkeypatch httpx
verbs to return programmable per-route responses; assert the CLI
flow (preview → drain loop → final DELETE) calls the right endpoints
in the right order with the right params, and surfaces friendly errors
on the refusal paths (unreachable, infeasible, no-such-host).
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


def _ok(body: dict[str, Any]) -> _FakeResponse:
    return _FakeResponse(status_code=200, _json=body)


@pytest.fixture
def gateway(monkeypatch):
    """Programmable httpx.get / .post / .delete fake.

    `state["routes"][(method, path)] = _FakeResponse | Exception | callable`.
    For drain (which is called in a loop), the callable form receives the
    call index and returns a response — lets tests script "first call
    returns remaining=3, second returns remaining=0."
    """
    state: dict[str, Any] = {
        "routes": {},   # (method, path) → response | list[response] | Exception
        "calls": [],    # list[(method, url, params)]
    }

    def _dispatch(method: str):
        def _fake(url, params=None, timeout=None, **_kwargs):
            state["calls"].append((method, url, params))
            for (m, path), resp in state["routes"].items():
                if m == method and path in url:
                    if isinstance(resp, list):
                        # Pop the next scripted response.
                        if not resp:
                            return _FakeResponse(status_code=500, text="script exhausted")
                        return resp.pop(0)
                    if isinstance(resp, Exception):
                        raise resp
                    return resp
            return _FakeResponse(status_code=404, text="not found")
        return _fake

    monkeypatch.setattr(httpx, "get", _dispatch("GET"))
    monkeypatch.setattr(httpx, "post", _dispatch("POST"))
    monkeypatch.setattr(httpx, "delete", _dispatch("DELETE"))
    return state


_NODE_ID = "11111111-aaaa-bbbb-cccc-222222222222"
_HOST = "100.64.0.5"


def _inventory_with_host(host: str = _HOST) -> _FakeResponse:
    return _ok({
        "satellites": [{
            "id": _NODE_ID, "host": host, "role": "storage", "status": "active",
            "label": None, "capabilities": None,
            "joined_at": "2026-05-21T16:00:00+00:00",
            "updated_at": "2026-05-21T16:00:00+00:00",
        }],
        "count": 1,
    })


# ─────────────────────── graceful happy path ───────────────────────


class TestGracefulPath:
    def test_drains_then_deletes(self, gateway, capsys, monkeypatch):
        """Two drain batches (5 then 0) followed by the final DELETE."""
        gateway["routes"][("GET", "/fleet/inventory")] = _inventory_with_host()
        gateway["routes"][("GET", "/fleet/extend/drain-preview")] = _ok({
            "node_id": _NODE_ID, "blob_count": 5, "total_bytes": 500,
            "primary_free_bytes": 10**9, "feasible": True,
            "satellite_reachable": True,
        })
        gateway["routes"][("POST", "/fleet/storage/drain")] = [
            _ok({"node_id": _NODE_ID, "drained": 3, "skipped": 0,
                 "remaining": 2, "bytes_drained": 300, "fetch_error": None,
                 "results": []}),
            _ok({"node_id": _NODE_ID, "drained": 2, "skipped": 0,
                 "remaining": 0, "bytes_drained": 200, "fetch_error": None,
                 "results": []}),
        ]
        gateway["routes"][("DELETE", f"/fleet/satellites/{_NODE_ID}")] = _ok({
            "node_id": _NODE_ID, "removed": True, "forced": False,
            "lost_blobs": 0, "lost_bytes": 0,
        })

        extend_module.remove_satellite(host=_HOST, force=False, yes=True)

        captured = capsys.readouterr().out
        assert "drained=3" in captured
        assert "drained=2" in captured
        assert "Removed 100.64.0.5: 5 blob(s) drained" in captured
        # Two POST drains + one DELETE issued.
        methods = [m for m, _, _ in gateway["calls"]]
        assert methods.count("POST") == 2
        assert methods.count("DELETE") == 1

    def test_aborts_when_not_feasible(self, gateway, capsys):
        gateway["routes"][("GET", "/fleet/inventory")] = _inventory_with_host()
        gateway["routes"][("GET", "/fleet/extend/drain-preview")] = _ok({
            "node_id": _NODE_ID, "blob_count": 5, "total_bytes": 10**12,
            "primary_free_bytes": 100,
            "feasible": False, "satellite_reachable": True,
        })

        with pytest.raises(typer.Exit) as exc:
            extend_module.remove_satellite(host=_HOST, force=False, yes=True)
        assert exc.value.exit_code == 1
        captured = capsys.readouterr().out
        assert "needs at least" in captured
        # No drain or delete should have been issued.
        methods = [m for m, _, _ in gateway["calls"]]
        assert "POST" not in methods
        assert "DELETE" not in methods

    def test_aborts_when_satellite_unreachable(self, gateway, capsys):
        gateway["routes"][("GET", "/fleet/inventory")] = _inventory_with_host()
        gateway["routes"][("GET", "/fleet/extend/drain-preview")] = _ok({
            "node_id": _NODE_ID, "blob_count": 5, "total_bytes": 500,
            "primary_free_bytes": 10**9,
            "feasible": True, "satellite_reachable": False,
        })

        with pytest.raises(typer.Exit) as exc:
            extend_module.remove_satellite(host=_HOST, force=False, yes=True)
        assert exc.value.exit_code == 1
        captured = capsys.readouterr().out
        assert "unreachable" in captured
        assert "--force" in captured


# ─────────────────────── force path ───────────────────────


class TestForcePath:
    def test_force_when_unreachable_calls_delete_force(self, gateway, capsys):
        gateway["routes"][("GET", "/fleet/inventory")] = _inventory_with_host()
        gateway["routes"][("GET", "/fleet/extend/drain-preview")] = _ok({
            "node_id": _NODE_ID, "blob_count": 5, "total_bytes": 500,
            "primary_free_bytes": 10**9,
            "feasible": True, "satellite_reachable": False,
        })
        gateway["routes"][("DELETE", f"/fleet/satellites/{_NODE_ID}")] = _ok({
            "node_id": _NODE_ID, "removed": True, "forced": True,
            "lost_blobs": 5, "lost_bytes": 500,
        })

        extend_module.remove_satellite(host=_HOST, force=True, yes=True)

        captured = capsys.readouterr().out
        assert "Removed 100.64.0.5 (force)" in captured
        assert "5 blob(s) lost" in captured
        # DELETE was called with force=true as a query param.
        delete_calls = [c for c in gateway["calls"] if c[0] == "DELETE"]
        assert len(delete_calls) == 1
        assert delete_calls[0][2] == {"force": "true"}

    def test_force_on_reachable_satellite_warns(self, gateway, capsys):
        """When --force is used on a satellite that's actually responding,
        the CLI warns but still proceeds with --yes."""
        gateway["routes"][("GET", "/fleet/inventory")] = _inventory_with_host()
        gateway["routes"][("GET", "/fleet/extend/drain-preview")] = _ok({
            "node_id": _NODE_ID, "blob_count": 5, "total_bytes": 500,
            "primary_free_bytes": 10**9,
            "feasible": True, "satellite_reachable": True,  # reachable!
        })
        gateway["routes"][("DELETE", f"/fleet/satellites/{_NODE_ID}")] = _ok({
            "node_id": _NODE_ID, "removed": True, "forced": True,
            "lost_blobs": 5, "lost_bytes": 500,
        })

        extend_module.remove_satellite(host=_HOST, force=True, yes=True)

        captured = capsys.readouterr().out
        assert "satellite responds" in captured
        assert "will lose data" in captured


# ─────────────────────── refusal paths ───────────────────────


class TestRefusalPaths:
    def test_unknown_host_friendly_error(self, gateway, capsys):
        # inventory has a different host.
        gateway["routes"][("GET", "/fleet/inventory")] = _inventory_with_host(
            host="some-other-host"
        )

        with pytest.raises(typer.Exit) as exc:
            extend_module.remove_satellite(host=_HOST, force=False, yes=True)
        assert exc.value.exit_code == 1
        captured = capsys.readouterr().out
        assert "No satellite registered with host" in captured
        assert "molebie-ai extend list" in captured

    def test_zero_blobs_skips_drain_loop(self, gateway, capsys):
        """If the satellite has already been drained, the loop is skipped and
        the inventory delete fires directly."""
        gateway["routes"][("GET", "/fleet/inventory")] = _inventory_with_host()
        gateway["routes"][("GET", "/fleet/extend/drain-preview")] = _ok({
            "node_id": _NODE_ID, "blob_count": 0, "total_bytes": 0,
            "primary_free_bytes": 10**9,
            "feasible": True, "satellite_reachable": True,
        })
        gateway["routes"][("DELETE", f"/fleet/satellites/{_NODE_ID}")] = _ok({
            "node_id": _NODE_ID, "removed": True, "forced": False,
            "lost_blobs": 0, "lost_bytes": 0,
        })

        extend_module.remove_satellite(host=_HOST, force=False, yes=True)

        captured = capsys.readouterr().out
        assert "no blobs to drain" in captured
        # No POST drain calls.
        methods = [m for m, _, _ in gateway["calls"]]
        assert "POST" not in methods
