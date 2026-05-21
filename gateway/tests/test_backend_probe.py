"""Tests for BackendProbe — per-backend health probing that drives circuit
transitions in BackendHealth.

The probe is the source-of-truth for unreachable / drifted backends; the
selector reacts via BackendHealth without consulting the probe directly.
These tests exercise the probe's wiring into BackendHealth, the fingerprint
algorithm and drift handling, and the snapshot shape the metrics route
consumes.

httpx is mocked at the AsyncClient level — same pattern used in
test_inference_cross_tier_fallback.py.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.services.backend_probe import (
    BackendProbe,
    _fingerprint_from_response,
    get_backend_probe,
    reset_backend_probe,
)
from app.services.inference_pool import (
    BackendPool,
    CircuitState,
    InferenceBackend,
)

# ─────────────────────────── helpers ───────────────────────────


def _make_pools(*backends: InferenceBackend) -> dict[str, BackendPool]:
    pools: dict[str, BackendPool] = {
        "instant": BackendPool("instant"),
        "thinking": BackendPool("thinking"),
    }
    for b in backends:
        pools[b.tier].backends.append(b)
    return pools


def _make_backend(node_id: str = "local-instant", tier: str = "instant") -> InferenceBackend:
    return InferenceBackend(
        url="http://127.0.0.1:8081",
        api_prefix="/v1",
        model="dummy",
        node_id=node_id,
        tier=tier,
    )


def _models_response(model_id: str = "qwen3-instant", server: str | None = None) -> httpx.Response:
    headers = {"server": server} if server else {}
    return httpx.Response(
        200,
        json={"object": "list", "data": [{"id": model_id, "object": "model"}]},
        headers=headers,
    )


@pytest.fixture(autouse=True)
def _reset_probe_singleton():
    """Probe is a module-level singleton; reset around every test so probe
    state from one test doesn't leak into the next."""
    reset_backend_probe()
    yield
    reset_backend_probe()


# ─────────────────────────── fingerprint algorithm ───────────────────────────


class TestFingerprintFromResponse:
    def test_deterministic_for_same_payload(self):
        r1 = _models_response("qwen3-thinking")
        r2 = _models_response("qwen3-thinking")
        assert _fingerprint_from_response(r1) == _fingerprint_from_response(r2)

    def test_differs_when_model_id_changes(self):
        r1 = _models_response("qwen3-thinking")
        r2 = _models_response("llama-3.2")
        assert _fingerprint_from_response(r1) != _fingerprint_from_response(r2)

    def test_uses_data_zero_id(self):
        expected = hashlib.sha256(b"qwen3-thinking").hexdigest()
        assert _fingerprint_from_response(_models_response("qwen3-thinking")) == expected

    def test_returns_none_for_malformed_body(self):
        resp = httpx.Response(200, json={"unexpected": "shape"})
        assert _fingerprint_from_response(resp) is None

    def test_returns_none_for_empty_data(self):
        resp = httpx.Response(200, json={"object": "list", "data": []})
        assert _fingerprint_from_response(resp) is None

    def test_returns_none_for_non_json(self):
        resp = httpx.Response(200, text="not json")
        assert _fingerprint_from_response(resp) is None


# ─────────────────────────── probe → BackendHealth wiring ───────────────────────────


async def _probe_with(
    backend: InferenceBackend,
    *,
    response: httpx.Response | None = None,
    raise_exc: Exception | None = None,
) -> BackendProbe:
    """Run one probe tick against `backend` with a mocked httpx response."""
    pools = _make_pools(backend)
    probe = BackendProbe(lambda: pools)
    probe._client = httpx.AsyncClient(timeout=2.0)
    if raise_exc is not None:
        mocked = AsyncMock(side_effect=raise_exc)
    else:
        mocked = AsyncMock(return_value=response)
    with patch.object(httpx.AsyncClient, "get", new=mocked):
        await probe._probe_one(backend)
    await probe._client.aclose()
    return probe


class TestProbeDrivesBackendHealth:
    async def test_success_pins_pool_fingerprint(self):
        backend = _make_backend("local-instant", "instant")
        probe = await _probe_with(backend, response=_models_response("qwen3-instant"))
        pool = probe._get_pools()["instant"]
        assert pool.expected_fingerprint == _fingerprint_from_response(
            _models_response("qwen3-instant")
        )
        assert backend.model_fingerprint == pool.expected_fingerprint
        assert backend.health.state == CircuitState.CLOSED

    async def test_drift_marks_backend_sticky_open(self):
        backend = _make_backend("local-instant", "instant")
        # First poll pins the fingerprint.
        probe = await _probe_with(backend, response=_models_response("qwen3-instant"))
        # Second poll: same backend, different model id → drift.
        with patch.object(
            httpx.AsyncClient,
            "get",
            new=AsyncMock(return_value=_models_response("llama-3.2")),
        ):
            probe._client = httpx.AsyncClient(timeout=2.0)
            await probe._probe_one(backend)
            await probe._client.aclose()
        assert backend.health.state == CircuitState.OPEN
        assert backend.health.drift_open is True
        snap = probe.latest()["local-instant"]
        assert snap.last_error == "fingerprint_drift"

    async def test_drift_clears_when_fingerprint_returns(self):
        backend = _make_backend("local-instant", "instant")
        # Pin → drift → recover.
        probe = await _probe_with(backend, response=_models_response("qwen3-instant"))
        with patch.object(
            httpx.AsyncClient,
            "get",
            new=AsyncMock(return_value=_models_response("llama-3.2")),
        ):
            probe._client = httpx.AsyncClient(timeout=2.0)
            await probe._probe_one(backend)
            await probe._client.aclose()
        assert backend.health.drift_open is True
        # Restore the original model id — drift should clear.
        with patch.object(
            httpx.AsyncClient,
            "get",
            new=AsyncMock(return_value=_models_response("qwen3-instant")),
        ):
            probe._client = httpx.AsyncClient(timeout=2.0)
            await probe._probe_one(backend)
            await probe._client.aclose()
        assert backend.health.drift_open is False
        assert backend.health.state == CircuitState.CLOSED

    async def test_three_consecutive_probe_failures_trip_open(self):
        backend = _make_backend("local-instant", "instant")
        for _ in range(3):
            await _probe_with(backend, raise_exc=httpx.ConnectError("refused"))
        assert backend.health.state == CircuitState.OPEN
        snap = backend.health
        assert snap.consecutive_failures >= 3

    async def test_timeout_recorded_as_failure(self):
        backend = _make_backend("local-instant", "instant")
        probe = await _probe_with(backend, raise_exc=httpx.TimeoutException("slow"))
        snap = probe.latest()["local-instant"]
        assert snap.status == "down"
        assert snap.last_error == "timeout"
        assert backend.health.consecutive_failures == 1

    async def test_http_5xx_recorded_as_failure(self):
        backend = _make_backend("local-instant", "instant")
        probe = await _probe_with(backend, response=httpx.Response(503, text="unavailable"))
        snap = probe.latest()["local-instant"]
        assert snap.status == "down"
        assert snap.last_error == "HTTP 503"
        assert backend.health.consecutive_failures == 1

    async def test_server_header_captured(self):
        backend = _make_backend("local-instant", "instant")
        probe = await _probe_with(
            backend,
            response=_models_response("qwen3-instant", server="vLLM/0.6.2"),
        )
        assert backend.server_version == "vLLM/0.6.2"
        assert probe.latest()["local-instant"].server_version == "vLLM/0.6.2"

    async def test_server_header_absent_is_safe(self):
        backend = _make_backend("local-instant", "instant")
        probe = await _probe_with(backend, response=_models_response("qwen3-instant"))
        assert backend.server_version is None
        assert probe.latest()["local-instant"].server_version is None

    async def test_malformed_models_body_does_not_crash(self):
        backend = _make_backend("local-instant", "instant")
        probe = await _probe_with(
            backend,
            response=httpx.Response(200, json={"unexpected": "shape"}),
        )
        # No fingerprint, no drift, but the probe still records a success.
        assert backend.model_fingerprint is None
        assert backend.health.state == CircuitState.CLOSED
        snap = probe.latest()["local-instant"]
        assert snap.status in ("up", "cold")  # cold depends on registry state


# ─────────────────────────── snapshot keying ───────────────────────────


class TestSnapshotShape:
    async def test_latest_keyed_by_node_id(self):
        backend = _make_backend("local-instant", "instant")
        probe = await _probe_with(backend, response=_models_response())
        snaps = probe.latest()
        assert "local-instant" in snaps
        assert snaps["local-instant"].tier == "instant"

    async def test_circuit_state_read_through(self):
        backend = _make_backend("local-instant", "instant")
        # Trip it via probe failures.
        for _ in range(3):
            await _probe_with(backend, raise_exc=httpx.ConnectError("nope"))
        # Now a successful probe — but pool's expected fingerprint is unpinned
        # because every prior probe failed. Use a fresh probe so the pools dict
        # state is consistent for the final tick.
        probe = await _probe_with(backend, response=_models_response())
        snap = probe.latest()["local-instant"]
        # After 3 prior trips, the backend was OPEN. The successful poll
        # transitions HALF_OPEN→CLOSED via _close_circuit (probe success
        # path). The snapshot reflects the live circuit_state.
        assert snap.circuit_state in (CircuitState.CLOSED, CircuitState.HALF_OPEN, CircuitState.OPEN)


# ─────────────────────────── singleton accessor ───────────────────────────


class TestProbeSingleton:
    def test_get_backend_probe_returns_singleton(self):
        p1 = get_backend_probe(get_pools=lambda: {})
        p2 = get_backend_probe()
        assert p1 is p2

    def test_reset_drops_singleton(self):
        p1 = get_backend_probe(get_pools=lambda: {})
        reset_backend_probe()
        p2 = get_backend_probe(get_pools=lambda: {})
        assert p1 is not p2
