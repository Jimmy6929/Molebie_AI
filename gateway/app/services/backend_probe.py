"""Background probe that pings each configured inference tier.

Polls `/v1/models` (or equivalent health path) on each backend URL every
N seconds and caches the result. Cheap: 2 s timeout, async httpx, and
only the tiers actually configured via settings are polled.

Status semantics:
  * up    — HTTP 2xx within the timeout
  * down  — HTTP error, timeout, or connection refused
  * cold  — up but no successful chat request served in the last 5 min
            (set by the chat route via MetricsRegistry, combined here)

v1 scope: instant + thinking only. Ollama/other backends can be layered
later by extending the configured-tiers list.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from app.config import Settings

_COLD_THRESHOLD_SEC = 5 * 60
_PROBE_INTERVAL_SEC = 5.0
_PROBE_TIMEOUT_SEC = 2.0


@dataclass(slots=True)
class BackendSnapshot:
    tier: str
    url: str | None
    model: str | None
    status: str           # 'up' | 'down' | 'cold' | 'not_configured'
    last_latency_ms: float | None = None
    last_error: str | None = None
    last_checked_at: float = 0.0


class BackendProbe:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._snapshots: dict[str, BackendSnapshot] = {}
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._client: httpx.AsyncClient | None = None

    def latest(self) -> dict[str, BackendSnapshot]:
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

    def _tiers(self) -> list[tuple[str, str, str]]:
        """Return [(tier_name, base_url, model_name)] for configured tiers."""
        s = self._settings
        out: list[tuple[str, str, str]] = []
        if s.inference_instant_url:
            out.append(("instant", s.inference_instant_url, s.get_model_for_mode("instant")))
        if s.inference_thinking_url:
            out.append(("thinking", s.inference_thinking_url, s.get_model_for_mode("thinking")))
        return out

    async def _run(self) -> None:
        # Seed unconfigured entries so the UI can show "(not configured)".
        for tier in ("instant", "thinking"):
            self._snapshots.setdefault(
                tier,
                BackendSnapshot(tier=tier, url=None, model=None, status="not_configured"),
            )

        # Heartbeat into the Subsystems panel — see system_probe for rationale.
        from app.services.metrics_registry import get_metrics_registry
        registry = get_metrics_registry()

        while not self._stopping:
            tiers = self._tiers()
            t0 = time.perf_counter()
            ok = True
            if tiers:
                try:
                    await asyncio.gather(*[self._probe_one(t, url, model) for t, url, model in tiers])
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

    async def _probe_one(self, tier: str, url: str, model: str) -> None:
        assert self._client is not None
        prefix = self._settings.get_api_prefix_for_mode(tier)
        # `/v1/models` is the OpenAI-compatible health path; mlx_vlm uses `/v1/models` too.
        probe_url = f"{url.rstrip('/')}{prefix}/models"
        started = time.perf_counter()
        now = time.time()
        try:
            resp = await self._client.get(probe_url)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if resp.status_code < 400:
                # Combine with registry data to flag "cold" — if healthy but not
                # used recently, surface that so operators know a first request
                # will pay cold-start cost.
                status = "up"
                try:
                    from app.services.metrics_registry import get_metrics_registry
                    reg_snap = await get_metrics_registry().snapshot()
                    tier_stats = reg_snap.get("tiers", {}).get(tier, {})
                    last = tier_stats.get("last_completed_at")
                    if last is None or (now - last) > _COLD_THRESHOLD_SEC:
                        status = "cold"
                except Exception:
                    pass  # never let registry issues break probing
                self._snapshots[tier] = BackendSnapshot(
                    tier=tier,
                    url=url,
                    model=model,
                    status=status,
                    last_latency_ms=latency_ms,
                    last_checked_at=now,
                )
            else:
                self._snapshots[tier] = BackendSnapshot(
                    tier=tier,
                    url=url,
                    model=model,
                    status="down",
                    last_error=f"HTTP {resp.status_code}",
                    last_checked_at=now,
                )
        except httpx.TimeoutException:
            self._snapshots[tier] = BackendSnapshot(
                tier=tier, url=url, model=model, status="down",
                last_error="timeout", last_checked_at=now,
            )
        except httpx.TransportError as exc:
            self._snapshots[tier] = BackendSnapshot(
                tier=tier, url=url, model=model, status="down",
                last_error=f"{type(exc).__name__}", last_checked_at=now,
            )


_probe: BackendProbe | None = None


def get_backend_probe(settings: Settings | None = None) -> BackendProbe:
    global _probe
    if _probe is None:
        if settings is None:
            from app.config import get_settings
            settings = get_settings()
        _probe = BackendProbe(settings)
    return _probe


def reset_backend_probe() -> None:
    global _probe
    _probe = None
