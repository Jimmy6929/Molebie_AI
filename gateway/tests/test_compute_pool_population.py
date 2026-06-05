"""Tests for compute-satellite pool population (Plan B compute extension, D1).

Covers the consumer side of the compute-capabilities contract:
  * ``parse_compute_capabilities`` against the golden fixture + garbage.
  * ``InferenceService`` populating per-tier pools from ``fleet_satellites``.
  * ``reload_backends`` adding/removing satellites and preserving the drift
    pin + health of surviving backends across a build-and-swap.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.schema import init_database_sync
from app.services.inference import (
    InferenceService,
    parse_compute_capabilities,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "compute_capabilities_v1.json"


@pytest.fixture
def isolated_data_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("DATA_DIR", td)
        get_settings.cache_clear()
        init_database_sync(td, embedding_dim=1024, auth_mode="single")
        yield td
        get_settings.cache_clear()


def _seed_satellite(
    data_dir: str, *, host: str, role: str, capabilities: dict | None,
    status: str = "active",
) -> None:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        conn.execute(
            "INSERT INTO fleet_satellites "
            "(id, host, role, status, label, capabilities_json, "
            " tailscale_user, joined_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"id-{host}", host, role, status, None,
                json.dumps(capabilities) if capabilities is not None else None,
                "ops@example.com", "2026-06-05T00:00:00+00:00",
                "2026-06-05T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _delete_satellite(data_dir: str, host: str) -> None:
    conn = sqlite3.connect(Path(data_dir) / "molebie.db")
    try:
        conn.execute("DELETE FROM fleet_satellites WHERE host = ?", (host,))
        conn.commit()
    finally:
        conn.close()


def _backends(svc: InferenceService, tier: str):
    return {b.node_id: b for b in svc.selector.pools[tier].backends}


def _sat_backends(svc: InferenceService, tier: str):
    """Only the satellite backends (local backends may exist from default config)."""
    return {nid: b for nid, b in _backends(svc, tier).items()
            if nid.startswith("satellite-")}


# ─────────────────────────── contract: the parser ───────────────────────────


def test_parse_golden_fixture():
    """The committed golden fixture parses to the normalized shape. This is the
    cross-package contract pin — the satellite producer asserts the same file."""
    raw = json.loads(_FIXTURE.read_text())
    caps = parse_compute_capabilities(raw)
    assert caps == {
        "port": 11434,
        "api_prefix": "/v1",
        "tiers": {"instant": "llama3.2:3b", "thinking": "qwen2.5:14b"},
    }


@pytest.mark.parametrize("raw", [
    None,
    {},
    {"compute": None},
    {"compute": {}},
    {"compute": {"port": 11434}},                       # no tiers
    {"compute": {"port": "11434", "tiers": {"instant": {"model": "m"}}}},  # str port
    {"compute": {"port": 0, "tiers": {"instant": {"model": "m"}}}},        # bad port
    {"compute": {"port": 11434, "tiers": {"instant": {}}}},               # no model
    {"compute": {"port": 11434, "tiers": {"bogus": {"model": "m"}}}},     # bad tier
    {"compute": {"port": True, "tiers": {"instant": {"model": "m"}}}},    # bool port
])
def test_parse_rejects_garbage(raw):
    assert parse_compute_capabilities(raw) is None


# ─────────────────────────── pool population ───────────────────────────


def test_pool_populated_from_db(isolated_data_dir):
    _seed_satellite(
        isolated_data_dir, host="100.99.0.7", role="compute",
        capabilities=json.loads(_FIXTURE.read_text()),
    )
    svc = InferenceService(get_settings())

    thinking = _backends(svc, "thinking")
    instant = _backends(svc, "instant")
    assert "satellite-100.99.0.7-thinking" in thinking
    assert "satellite-100.99.0.7-instant" in instant

    b = thinking["satellite-100.99.0.7-thinking"]
    assert b.url == "http://100.99.0.7:11434"
    assert b.api_prefix == "/v1"
    assert b.model == "qwen2.5:14b"


def test_role_both_is_included(isolated_data_dir):
    _seed_satellite(
        isolated_data_dir, host="100.99.0.8", role="both",
        capabilities={"compute": {"port": 11434, "api_prefix": "/v1",
                                  "tiers": {"thinking": {"model": "qwen2.5:14b"}}}},
    )
    svc = InferenceService(get_settings())
    assert "satellite-100.99.0.8-thinking" in _backends(svc, "thinking")


def test_storage_role_contributes_no_compute_backend(isolated_data_dir):
    _seed_satellite(
        isolated_data_dir, host="100.99.0.9", role="storage",
        capabilities=json.loads(_FIXTURE.read_text()),
    )
    svc = InferenceService(get_settings())
    assert _sat_backends(svc, "thinking") == {}
    assert _sat_backends(svc, "instant") == {}


def test_inactive_satellite_excluded(isolated_data_dir):
    _seed_satellite(
        isolated_data_dir, host="100.99.0.10", role="compute", status="draining",
        capabilities=json.loads(_FIXTURE.read_text()),
    )
    svc = InferenceService(get_settings())
    assert _sat_backends(svc, "thinking") == {}


def test_malformed_capabilities_skipped_not_crash(isolated_data_dir):
    _seed_satellite(
        isolated_data_dir, host="100.99.0.11", role="compute",
        capabilities={"compute": "not-a-dict"},
    )
    # Must not raise — a bad row contributes nothing.
    svc = InferenceService(get_settings())
    assert _sat_backends(svc, "thinking") == {}


# ─────────────────────────── reload (build-and-swap) ───────────────────────────


def test_reload_adds_then_removes(isolated_data_dir):
    svc = InferenceService(get_settings())
    assert _sat_backends(svc, "thinking") == {}

    _seed_satellite(
        isolated_data_dir, host="100.99.0.12", role="compute",
        capabilities=json.loads(_FIXTURE.read_text()),
    )
    svc.reload_backends()
    assert "satellite-100.99.0.12-thinking" in _backends(svc, "thinking")

    _delete_satellite(isolated_data_dir, "100.99.0.12")
    svc.reload_backends()
    assert _sat_backends(svc, "thinking") == {}


def test_reload_preserves_drift_pin_and_health(isolated_data_dir):
    """A reload is not a restart: surviving backends keep their drift pin and
    health so an unrelated join doesn't un-trip a breaker or unpin drift."""
    _seed_satellite(
        isolated_data_dir, host="100.99.0.13", role="compute",
        capabilities={"compute": {"port": 11434, "api_prefix": "/v1",
                                  "tiers": {"thinking": {"model": "qwen2.5:14b"}}}},
    )
    svc = InferenceService(get_settings())
    # Pin the tier's expected fingerprint and the surviving backend's health.
    svc.selector.pools["thinking"].expected_fingerprint = "sha-abc"
    surviving = _backends(svc, "thinking")["satellite-100.99.0.13-thinking"]
    surviving_health = surviving.health

    # A second, unrelated compute satellite joins.
    _seed_satellite(
        isolated_data_dir, host="100.99.0.14", role="compute",
        capabilities={"compute": {"port": 11434, "api_prefix": "/v1",
                                  "tiers": {"thinking": {"model": "qwen2.5:14b"}}}},
    )
    svc.reload_backends()

    assert svc.selector.pools["thinking"].expected_fingerprint == "sha-abc"
    # Surviving backend keeps the SAME health object (not reset to a fresh one).
    assert _backends(svc, "thinking")["satellite-100.99.0.13-thinking"].health is surviving_health
    assert "satellite-100.99.0.14-thinking" in _backends(svc, "thinking")


# ───────────────── in-flight selector capture (both request paths) ─────────────────


class _OKStream:
    status_code = 200

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for ln in ['data: {"choices":[{"delta":{"content":"hi"}}]}', "data: [DONE]"]:
            yield ln


class _StreamCtx:
    """Stream ctx that fails for the thinking primary, succeeds for instant."""

    def __init__(self, url: str):
        self.url = url

    async def __aenter__(self):
        if "test-thinking" in self.url:
            raise httpx.ConnectError("thinking down")
        return _OKStream()

    async def __aexit__(self, *a):
        return False


@pytest.mark.asyncio
async def test_stream_fallback_threads_captured_selector(monkeypatch):
    """Regression (review finding): the streaming post-call fallback must reuse
    the per-request captured selector, not re-read self.selector — otherwise a
    mid-request reload_backends() swap splits fb selection from fb accounting.
    Guards the inference.py:1117 fix that the bulk replace_all missed."""
    get_settings.cache_clear()
    settings = get_settings()
    settings.inference_instant_url = "http://test-instant:8081"
    settings.inference_thinking_url = "http://test-thinking:8080"
    settings.routing_thinking_fallback_to_instant = True
    svc = InferenceService(settings)

    seen_selectors = []
    real_select = svc._select_backend

    def spy(mode, session_id=None, voice_mode=False, selector=None):
        seen_selectors.append(selector)
        return real_select(mode, session_id, voice_mode, selector=selector)

    monkeypatch.setattr(svc, "_select_backend", spy)
    monkeypatch.setattr(
        httpx.AsyncClient, "stream", lambda self, method, url, **kw: _StreamCtx(url)
    )

    async for _ in svc.generate_response_stream(
        messages=[{"role": "user", "content": "hi"}], mode="thinking",
    ):
        pass

    # Primary (thinking) select + post-failure fallback (instant) select both ran.
    assert len(seen_selectors) >= 2
    # BOTH must use the same captured selector — never None, never a re-read.
    assert all(s is svc.selector for s in seen_selectors)
    get_settings.cache_clear()
