"""Background probe that pings each backend in the selector's pools.

Polls `/v1/models` on every backend across every pool every N seconds and
feeds the result into each backend's ``BackendHealth``. The selector reads
the resulting circuit state directly when picking a backend per request, so
unhealthy local or satellite backends are skipped without waiting for the
next request to time out.

Status semantics (the probe's *snapshot* status, distinct from the
``BackendHealth.state`` circuit on the same backend):
  * up    — HTTP 2xx within the timeout
  * down  — HTTP error, timeout, or connection refused
  * cold  — up but no successful chat request served in the last 5 min
            (combined here from the metrics registry, informational only)

Wiring: the probe takes a ``get_pools`` callable rather than the selector
itself — minimal coupling, lazy resolution so the inference service can be
constructed after the probe is wired up (matches existing lazy-singleton
patterns elsewhere).
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from app.services.inference_pool import (
    BackendPool,
    CircuitState,
    InferenceBackend,
)

_COLD_THRESHOLD_SEC = 5 * 60
_PROBE_INTERVAL_SEC = 5.0
_PROBE_TIMEOUT_SEC = 2.0


@dataclass(slots=True)
class BackendSnapshot:
    node_id: str
    tier: str
    url: str | None
    model: str | None
    status: str           # 'up' | 'down' | 'cold' | 'not_configured'
    last_latency_ms: float | None = None
    last_error: str | None = None
    last_checked_at: float = 0.0
    model_fingerprint: str | None = None
    server_version: str | None = None
    # Read-through from the matching InferenceBackend.health.state at the
    # moment the snapshot was taken. Lets the monitor render circuit state
    # without reaching into pool internals.
    circuit_state: CircuitState = CircuitState.CLOSED


PoolsResolver = Callable[[], dict[str, BackendPool]]


class BackendProbe:
    def __init__(self, get_pools: PoolsResolver) -> None:
        self._get_pools = get_pools
        self._snapshots: dict[str, BackendSnapshot] = {}
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._client: httpx.AsyncClient | None = None

    def latest(self) -> dict[str, BackendSnapshot]:
        """All snapshots keyed by ``node_id``."""
        return dict(self._snapshots)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._client = httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SEC)
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _backends(self) -> list[InferenceBackend]:
        """Flatten pools into a single backend list for this tick."""
        try:
            pools = self._get_pools()
        except Exception:
            return []
        out: list[InferenceBackend] = []
        for pool in pools.values():
            out.extend(pool.backends)
        return out

    async def _run(self) -> None:
        # Heartbeat into the Subsystems panel — see system_probe for rationale.
        from app.services.metrics_registry import get_metrics_registry
        registry = get_metrics_registry()

        while not self._stopping:
            backends = self._backends()
            t0 = time.perf_counter()
            ok = True
            if backends:
                try:
                    await asyncio.gather(*[self._probe_one(b) for b in backends])
                except Exception:
                    ok = False
            try:
                up_count = sum(1 for s in self._snapshots.values() if s.status == "up")
                await registry.record_subsystem(
                    "probe.backend",
                    (time.perf_counter() - t0) * 1000.0,
                    ok=ok,
                    note=f"{up_count}/{len(self._snapshots)} up",
                )
            except Exception:
                pass
            await asyncio.sleep(_PROBE_INTERVAL_SEC)

    async def _probe_one(self, backend: InferenceBackend) -> None:
        assert self._client is not None
        probe_url = f"{backend.url.rstrip('/')}{backend.api_prefix}/models"
        started = time.perf_counter()
        now = time.time()
        try:
            resp = await self._client.get(probe_url)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if resp.status_code < 400:
                await self._handle_success(backend, resp, latency_ms, now)
            else:
                self._handle_failure(
                    backend, f"HTTP {resp.status_code}", latency_ms, now
                )
        except httpx.TimeoutException:
            self._handle_failure(backend, "timeout", None, now)
        except httpx.TransportError as exc:
            self._handle_failure(backend, type(exc).__name__, None, now)

    async def _handle_success(
        self,
        backend: InferenceBackend,
        resp: httpx.Response,
        latency_ms: float,
        now: float,
    ) -> None:
        fingerprint = _fingerprint_from_response(resp)
        server_version = resp.headers.get("server")

        backend.server_version = server_version
        backend.model_fingerprint = fingerprint

        # Drift check: if the pool has pinned a fingerprint and this backend
        # reports a different one, the operator swapped the model behind the
        # backend's back. Mark drift; do not record a probe-success.
        if fingerprint is not None:
            pool = self._pool_for(backend)
            if pool is not None:
                if pool.expected_fingerprint is None:
                    pool.expected_fingerprint = fingerprint
                elif pool.expected_fingerprint != fingerprint:
                    backend.health.mark_fingerprint_drift()
                    self._record_snapshot(
                        backend,
                        status="down",
                        last_latency_ms=latency_ms,
                        last_error="fingerprint_drift",
                        now=now,
                        fingerprint=fingerprint,
                        server_version=server_version,
                    )
                    return
                elif backend.health.drift_open:
                    # Fingerprint matches pool expectation again — recovery.
                    backend.health.clear_drift()

        backend.health.record_probe_success()

        # Cold status is a probe-snapshot concern only — informational for
        # operators. Compute from the registry's last-completed timestamp.
        status = "up"
        try:
            from app.services.metrics_registry import get_metrics_registry
            reg_snap = await get_metrics_registry().snapshot()
            tier_stats = reg_snap.get("tiers", {}).get(backend.tier, {})
            last = tier_stats.get("last_completed_at")
            if last is None or (now - last) > _COLD_THRESHOLD_SEC:
                status = "cold"
        except Exception:
            pass  # registry hiccups must never break probing

        self._record_snapshot(
            backend,
            status=status,
            last_latency_ms=latency_ms,
            last_error=None,
            now=now,
            fingerprint=fingerprint,
            server_version=server_version,
        )

    def _handle_failure(
        self,
        backend: InferenceBackend,
        error: str,
        latency_ms: float | None,
        now: float,
    ) -> None:
        backend.health.record_probe_failure()
        self._record_snapshot(
            backend,
            status="down",
            last_latency_ms=latency_ms,
            last_error=error,
            now=now,
            fingerprint=backend.model_fingerprint,
            server_version=backend.server_version,
        )

    def _record_snapshot(
        self,
        backend: InferenceBackend,
        *,
        status: str,
        last_latency_ms: float | None,
        last_error: str | None,
        now: float,
        fingerprint: str | None,
        server_version: str | None,
    ) -> None:
        self._snapshots[backend.node_id] = BackendSnapshot(
            node_id=backend.node_id,
            tier=backend.tier,
            url=backend.url,
            model=backend.model,
            status=status,
            last_latency_ms=last_latency_ms,
            last_error=last_error,
            last_checked_at=now,
            model_fingerprint=fingerprint,
            server_version=server_version,
            circuit_state=backend.health.state,
        )

    def _pool_for(self, backend: InferenceBackend) -> BackendPool | None:
        try:
            return self._get_pools().get(backend.tier)
        except Exception:
            return None


def _fingerprint_from_response(resp: httpx.Response) -> str | None:
    """Compute a stable hash from an OpenAI-shape /v1/models response.

    Minimal: hashes ``data[0]["id"]`` only. Ephemeral fields (``created``,
    request counters, uptime) are deliberately excluded so a server restart
    doesn't trigger spurious drift. This catches the "operator pointed at a
    different model entirely" case; it does NOT catch "same id, different
    weights underneath" — no OpenAI-compatible server reliably exposes a
    weight checksum, so that case is unsolvable here. Defer to the future
    ``/molebie/satellite/info`` endpoint once satellites land.
    """
    try:
        body = resp.json()
    except Exception:
        return None
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    model_id = first.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    return hashlib.sha256(model_id.encode("utf-8")).hexdigest()


_probe: BackendProbe | None = None


def get_backend_probe(
    get_pools: PoolsResolver | None = None,
) -> BackendProbe:
    global _probe
    if _probe is None:
        if get_pools is None:
            # Default: resolve through the inference service's selector lazily,
            # so the inference service can be constructed on demand without
            # forcing the probe to know about it at import time.
            def _default_get_pools() -> dict[str, BackendPool]:
                from app.services.inference import get_inference_service
                return get_inference_service().selector.pools
            get_pools = _default_get_pools
        _probe = BackendProbe(get_pools)
    return _probe


def reset_backend_probe() -> None:
    global _probe
    _probe = None
