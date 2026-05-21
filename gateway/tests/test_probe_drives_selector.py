"""Integration tests for the full probe → BackendHealth → BackendSelector chain.

Unit tests cover each link in isolation (test_inference_pool.py for the
breaker, test_backend_probe.py for the probe). These tests pin the end-to-end
contract: probe outcomes must drive selector decisions, with no
intermediate translation layer.

The shape mirrors test_inference_cross_tier_fallback.py — single-file
integration tests that protect a wiring contract against future refactors.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.services.backend_probe import BackendProbe
from app.services.inference_pool import (
    BackendPool,
    BackendSelector,
    CircuitState,
    InferenceBackend,
    NoHealthyBackendError,
)


def _make_backend(node_id: str, tier: str) -> InferenceBackend:
    return InferenceBackend(
        url=f"http://127.0.0.1:8080/{node_id}",
        api_prefix="/v1",
        model="dummy",
        node_id=node_id,
        tier=tier,
    )


def _make_selector_and_pools() -> tuple[BackendSelector, dict[str, BackendPool]]:
    thinking = _make_backend("local-thinking", "thinking")
    instant = _make_backend("local-instant", "instant")
    pools: dict[str, BackendPool] = {
        "instant": BackendPool("instant", backends=[instant]),
        "thinking": BackendPool("thinking", backends=[thinking]),
    }
    selector = BackendSelector(pools, fallback_to_instant=True)
    return selector, pools


def _models_response(model_id: str) -> httpx.Response:
    return httpx.Response(
        200, json={"object": "list", "data": [{"id": model_id, "object": "model"}]}
    )


async def _tick_probe_for(probe: BackendProbe, backend: InferenceBackend, response: httpx.Response | None = None, raise_exc: Exception | None = None) -> None:
    probe._client = httpx.AsyncClient(timeout=2.0)
    if raise_exc is not None:
        mocked = AsyncMock(side_effect=raise_exc)
    else:
        mocked = AsyncMock(return_value=response)
    with patch.object(httpx.AsyncClient, "get", new=mocked):
        await probe._probe_one(backend)
    await probe._client.aclose()


class TestProbeDrivesSelectorRouting:
    async def test_probe_failures_route_thinking_to_instant_via_fallback(self):
        selector, pools = _make_selector_and_pools()
        probe = BackendProbe(lambda: pools)
        thinking = pools["thinking"].backends[0]

        # Three consecutive probe failures on the thinking backend.
        for _ in range(3):
            await _tick_probe_for(probe, thinking, raise_exc=httpx.ConnectError("refused"))

        assert thinking.health.state == CircuitState.OPEN
        # Selector now sees no healthy thinking backend → falls back to instant.
        chosen = selector.select("thinking")
        assert chosen.node_id == "local-instant"

    async def test_drift_routes_thinking_to_instant_immediately(self):
        selector, pools = _make_selector_and_pools()
        probe = BackendProbe(lambda: pools)
        thinking = pools["thinking"].backends[0]

        # First probe pins the fingerprint.
        await _tick_probe_for(probe, thinking, response=_models_response("qwen3-thinking"))
        assert pools["thinking"].expected_fingerprint is not None

        # Second probe sees a different model → drift → sticky OPEN.
        await _tick_probe_for(probe, thinking, response=_models_response("imposter"))
        assert thinking.health.drift_open is True
        assert thinking.health.state == CircuitState.OPEN

        # Selector must fall back without waiting for cooldown.
        chosen = selector.select("thinking")
        assert chosen.node_id == "local-instant"

    async def test_routes_again_after_drift_clears(self):
        selector, pools = _make_selector_and_pools()
        probe = BackendProbe(lambda: pools)
        thinking = pools["thinking"].backends[0]

        await _tick_probe_for(probe, thinking, response=_models_response("qwen3-thinking"))
        await _tick_probe_for(probe, thinking, response=_models_response("imposter"))
        assert thinking.health.drift_open is True

        # Operator restores the right model — next probe sees the matching
        # fingerprint and clears drift.
        await _tick_probe_for(probe, thinking, response=_models_response("qwen3-thinking"))
        assert thinking.health.drift_open is False
        assert thinking.health.state == CircuitState.CLOSED

        chosen = selector.select("thinking")
        assert chosen.node_id == "local-thinking"

    async def test_no_healthy_backend_when_both_tiers_dead(self):
        selector, pools = _make_selector_and_pools()
        probe = BackendProbe(lambda: pools)
        thinking = pools["thinking"].backends[0]
        instant = pools["instant"].backends[0]

        for _ in range(3):
            await _tick_probe_for(probe, thinking, raise_exc=httpx.ConnectError("nope"))
            await _tick_probe_for(probe, instant, raise_exc=httpx.ConnectError("nope"))

        assert thinking.health.state == CircuitState.OPEN
        assert instant.health.state == CircuitState.OPEN

        with pytest.raises(NoHealthyBackendError) as exc:
            selector.select("thinking")
        # Both tiers' states must be in the error report.
        node_ids = {n for n, _ in exc.value.attempted}
        assert "local-thinking" in node_ids
        assert "local-instant" in node_ids
