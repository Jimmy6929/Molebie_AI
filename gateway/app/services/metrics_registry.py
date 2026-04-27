"""In-memory metrics registry for the live terminal monitor.

Holds a bounded ring buffer of recent chat request records plus per-tier
counters (active / completed / errors). Stays process-local — nothing is
persisted to disk, and no prompt or response text is stored. The
`/metrics/live` route reads snapshots from here; the chat route writes
records into it at request completion.

Designed for single-process, single-worker FastAPI deployments. If the
gateway is ever run multi-worker, each worker would hold its own view;
aggregation is a v2 concern.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Reasonable ceiling — 500 requests covers ~4 minutes of steady 2/s traffic,
# which is plenty to compute rolling p95 without unbounded memory growth.
_MAX_RECORDS = 500

# Latency quantiles (p50/p95/p99, ttft, tpot) use the LONG window. Chosen
# to match the backend-probe "warm/cold" threshold: if the operator sends
# a chat and walks away, the Requests panel still shows stats when they
# come back. A 60-second window reset faster than anyone can read it.
_LATENCY_WINDOW_SEC = 5 * 60.0

# Throughput (`req/s`) uses the SHORT window: it's "are things flowing
# *right now*" — the dashboard's live-pulse signal. On the long window
# a single old request would register as 0.003 req/s forever; on 60 s
# it correctly drops to 0 once traffic stops.
_THROUGHPUT_WINDOW_SEC = 60.0

# Errors get the short window too — they're an *alert* signal, not a
# running average. A 5-minute error count hides recovery; a 60-second
# count says "something is wrong right now."
_ERROR_WINDOW_SEC = 60.0


@dataclass(slots=True)
class RequestRecord:
    """One completed chat request. Metadata only — no prompt/response text."""

    ended_at: float             # wall-clock seconds (time.time()) when request finished
    tier: str                   # 'instant' | 'thinking' | 'thinking_harder'
    model: str | None
    streaming: bool
    ok: bool
    # ttft_ms = time-to-first-token for streaming; approximated as total for
    # non-streaming paths (documented caller-side).
    ttft_ms: float | None = None
    tpot_ms: float | None = None   # time-per-output-token, mean
    total_ms: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    fallback: bool = False         # True if thinking tier fell back
    error_type: str | None = None  # exception class name if ok=False


@dataclass(slots=True)
class _TierCounters:
    active: int = 0
    completed: int = 0
    errors: int = 0
    fallback_from: int = 0         # times this tier was the *original* mode when a fallback occurred
    last_completed_at: float | None = None   # wall-clock seconds


class MetricsRegistry:
    """Thread/async-safe rolling metrics store."""

    def __init__(self) -> None:
        self._records: deque[RequestRecord] = deque(maxlen=_MAX_RECORDS)
        self._tiers: dict[str, _TierCounters] = {}
        self._errors_60s: deque[float] = deque()   # wall-clock timestamps
        self._lock = asyncio.Lock()
        self._started_at = time.time()

    # ── write path (called from chat route) ──────────────────────────

    async def mark_active(self, tier: str) -> None:
        async with self._lock:
            self._tiers.setdefault(tier, _TierCounters()).active += 1

    async def mark_inactive(self, tier: str) -> None:
        async with self._lock:
            t = self._tiers.setdefault(tier, _TierCounters())
            if t.active > 0:
                t.active -= 1

    async def record(self, rec: RequestRecord) -> None:
        async with self._lock:
            self._records.append(rec)
            t = self._tiers.setdefault(rec.tier, _TierCounters())
            t.completed += 1
            t.last_completed_at = time.time()
            if not rec.ok:
                t.errors += 1
                self._errors_60s.append(time.time())
            if rec.fallback:
                # original_mode lives on the record as `tier` when we recorded
                # against the intended (pre-fallback) tier. Kept simple here.
                t.fallback_from += 1

    # ── read path (called from /metrics/live route) ───────────────────

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            now = time.time()
            # Prune error timestamps outside the error window (60 s).
            while self._errors_60s and (now - self._errors_60s[0]) > _ERROR_WINDOW_SEC:
                self._errors_60s.popleft()

            # Latency stats over the long (5 min) window so they persist
            # between casual glances at the dashboard.
            recent_total_ms = [
                r.total_ms for r in self._records
                if r.total_ms is not None and (now - r.ended_at) < _LATENCY_WINDOW_SEC
            ]
            recent_ttft_ms = [
                r.ttft_ms for r in self._records
                if r.ttft_ms is not None and (now - r.ended_at) < _LATENCY_WINDOW_SEC
            ]
            recent_tpot_ms = [
                r.tpot_ms for r in self._records
                if r.tpot_ms is not None and (now - r.ended_at) < _LATENCY_WINDOW_SEC
            ]

            p50, p95, p99 = _quantiles(recent_total_ms)
            ttft_p50, _, _ = _quantiles(recent_ttft_ms)

            # Throughput gets the short window — the "is anything flowing?"
            # read. Uses a separate scan so the latency window can stay big.
            throughput_count = sum(
                1 for r in self._records
                if r.total_ms is not None and (now - r.ended_at) < _THROUGHPUT_WINDOW_SEC
            )
            req_per_sec = throughput_count / _THROUGHPUT_WINDOW_SEC

            # Last 20 TTFT samples for the sparkline (oldest → newest).
            ttft_series = [r.ttft_ms for r in list(self._records)[-20:] if r.ttft_ms is not None]
            # Last 20 TPOT for the "tpot_series" sparkline.
            tpot_series = [r.tpot_ms for r in list(self._records)[-20:] if r.tpot_ms is not None]

            # Recent event strip for footer — last 5, metadata only.
            recent_events = [
                {
                    "ts": r.ended_at,
                    "tier": r.tier,
                    "ok": r.ok,
                    "completion_tokens": r.completion_tokens,
                    "total_ms": r.total_ms,
                    "fallback": r.fallback,
                }
                for r in list(self._records)[-5:]
            ]

            fallback_count = sum(t.fallback_from for t in self._tiers.values())

            return {
                "uptime_sec": now - self._started_at,
                "ttft_p50_ms": ttft_p50,
                "tpot_mean_ms": (sum(recent_tpot_ms) / len(recent_tpot_ms)) if recent_tpot_ms else None,
                "total_p50_ms": p50,
                "total_p95_ms": p95,
                "total_p99_ms": p99,
                "req_per_sec": req_per_sec,
                "fallback_count": fallback_count,
                "errors_60s": len(self._errors_60s),
                "tiers": {
                    name: {
                        "active": t.active,
                        "completed": t.completed,
                        "errors": t.errors,
                        "last_completed_at": t.last_completed_at,
                    }
                    for name, t in self._tiers.items()
                },
                "ttft_series": ttft_series,
                "tpot_series": tpot_series,
                "recent_events": recent_events,
            }


def _quantiles(values: list[float]) -> tuple[float | None, float | None, float | None]:
    """Return (p50, p95, p99) from a list. None if fewer than 2 samples."""
    if not values:
        return None, None, None
    if len(values) == 1:
        v = values[0]
        return v, v, v
    sorted_v = sorted(values)

    def _pick(pct: float) -> float:
        idx = min(len(sorted_v) - 1, int(pct * len(sorted_v)))
        return sorted_v[idx]

    return _pick(0.50), _pick(0.95), _pick(0.99)


# ── module-level singleton ─────────────────────────────────────────────

_registry: MetricsRegistry | None = None


def get_metrics_registry() -> MetricsRegistry:
    """Return the process-wide registry (created lazily)."""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


def reset_metrics_registry() -> None:
    """Test-only helper — drop the singleton so each test gets a fresh one."""
    global _registry
    _registry = None
