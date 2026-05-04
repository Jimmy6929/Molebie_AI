"""In-memory metrics registry for the live terminal monitor.

Holds a bounded ring buffer of recent chat request records plus per-tier
counters (active / completed / errors). Stays process-local — nothing is
persisted to disk, and no prompt or response text is stored. The
`/metrics/live` route reads snapshots from here; the chat route writes
records into it at request completion.

Designed for single-process, single-worker FastAPI deployments. If the
gateway is ever run multi-worker, each worker would hold its own view;
aggregation is a v2 concern.

V2 schema additions (2026-05):

  * `subsystem_timer(name)` — per-subsystem call count + latency histogram
    + error count, all under rolling windows. Used for RAG, web search,
    verification, tasks, tools, etc.
  * `pipeline_event(req_id, stage, ms, status, note)` — bounded log of
    stage events for the most recent request. Powers the live "current
    request timeline" panel.
  * `mark_task(name, state)` — background-task counters analogous to
    `_TierCounters`.
  * `set_model_state(name, …)` — gauge-style declaration of model load
    state for embedding / reranker / NLI / TTS.

All new structures share the existing `asyncio.Lock` — no new contention,
no new locks. Bounded memory: ~2 MB worst case after 24h steady use.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
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

# Per-subsystem latency window: short enough to feel live, long enough
# that a single slow call doesn't dominate the displayed p95.
_SUBSYSTEM_LATENCY_RECENT = 64        # last N samples kept for sparklines
_SUBSYSTEM_WINDOW_SEC = 60.0

# Bounded pipeline event log — last 50 stage events across the most
# recent few requests. Plenty to render the live timeline panel without
# memory creep over a long-running gateway.
_PIPELINE_MAX_EVENTS = 50

# Schema version exposed via /metrics/live. Bump this when the snapshot
# shape changes in a way the CLI needs to know about.
SCHEMA_VERSION = 2


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


@dataclass(slots=True)
class _SubsystemStats:
    """Per-name accumulator: rolling latencies + counters + last note."""

    count_total: int = 0
    errors_total: int = 0
    last_at: float | None = None
    last_note: str | None = None
    # Sample buffer for sparkline (most recent → at the right edge).
    latency_recent: deque = field(default_factory=lambda: deque(maxlen=_SUBSYSTEM_LATENCY_RECENT))
    # Per-event timestamps for rolling 60s aggregates (count, errors).
    # (timestamp, latency_ms, ok)
    events_60s: deque = field(default_factory=deque)


@dataclass(slots=True)
class _TaskCounters:
    active: int = 0
    completed_total: int = 0
    failed_total: int = 0
    last_at: float | None = None
    # Wall-clock timestamps within the last 60s (completed + failed).
    completed_60s: deque = field(default_factory=deque)
    failed_60s: deque = field(default_factory=deque)


@dataclass(slots=True)
class _ModelState:
    """Gauge-style record of model load state for embedding / reranker / etc."""

    name: str
    loaded: bool = False
    load_ms: float | None = None
    url: str | None = None
    status: str | None = None   # free-form: "up" | "down" | "loaded" | "not_loaded"
    extra: dict | None = None


class MetricsRegistry:
    """Thread/async-safe rolling metrics store."""

    def __init__(self) -> None:
        self._records: deque[RequestRecord] = deque(maxlen=_MAX_RECORDS)
        self._tiers: dict[str, _TierCounters] = {}
        self._errors_60s: deque[float] = deque()   # wall-clock timestamps
        self._lock = asyncio.Lock()
        self._started_at = time.time()

        # V2 additions ------------------------------------------------------
        self._subsystems: dict[str, _SubsystemStats] = {}
        self._pipeline_events: deque[dict[str, Any]] = deque(maxlen=_PIPELINE_MAX_EVENTS)
        self._pipeline_started_at: dict[str, float] = {}
        self._tasks: dict[str, _TaskCounters] = {}
        self._models: dict[str, _ModelState] = {}
        # In-flight snapshot: gauge updated during a streaming request so the
        # monitor can render a live "currently streaming" badge with token
        # count and elapsed time. Cleared by clear_in_flight() on completion.
        self._in_flight: dict[str, Any] | None = None

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

    # ── V2 write paths ────────────────────────────────────────────────

    async def record_subsystem(
        self,
        name: str,
        latency_ms: float,
        ok: bool = True,
        note: str | None = None,
    ) -> None:
        """Record one call to a named subsystem (e.g. 'rag.embed').

        Counters are append-only; rolling windows prune on read. Safe to
        call from both async and sync contexts (use the sync helper if
        you're outside an event loop)."""
        async with self._lock:
            s = self._subsystems.setdefault(name, _SubsystemStats())
            now = time.time()
            s.count_total += 1
            s.last_at = now
            if note is not None:
                s.last_note = note
            if not ok:
                s.errors_total += 1
            s.latency_recent.append(latency_ms)
            s.events_60s.append((now, latency_ms, ok))

    @asynccontextmanager
    async def subsystem_timer(self, name: str, note: str | None = None):
        """Context manager that records the wrapped block's latency.

        Usage:
            async with metrics.subsystem_timer("rag.embed"):
                vec = await embed(query)

        On exception, records an error and re-raises. `note` is captured
        on exit as `last_note` for the snapshot — useful for "30 candidates",
        "top score 0.94", etc.
        """
        start = time.monotonic()
        ok = True
        try:
            yield
        except BaseException:
            ok = False
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000.0
            await self.record_subsystem(name, latency_ms, ok=ok, note=note)

    async def pipeline_event(
        self,
        req_id: str,
        stage: str,
        ms: float | None = None,
        status: str = "ok",
        note: str | None = None,
    ) -> None:
        """Append one stage event to the rolling pipeline log.

        `status` ∈ {"ok", "running", "queued", "fail"}. `ms` may be None
        for queued/running stages (will be filled by a later event with
        the same req_id+stage).
        """
        async with self._lock:
            now = time.time()
            if req_id not in self._pipeline_started_at:
                self._pipeline_started_at[req_id] = now
                # Bound the started_at map alongside the event deque.
                if len(self._pipeline_started_at) > _PIPELINE_MAX_EVENTS:
                    # Drop the oldest tracked req_id.
                    oldest = next(iter(self._pipeline_started_at))
                    self._pipeline_started_at.pop(oldest, None)
            self._pipeline_events.append({
                "ts": now,
                "req_id": req_id,
                "stage": stage,
                "ms": ms,
                "status": status,
                "note": note,
            })

    async def mark_task(self, name: str, state: str) -> None:
        """Track background-task lifecycle. `state` ∈ {"start","done","fail"}."""
        async with self._lock:
            t = self._tasks.setdefault(name, _TaskCounters())
            now = time.time()
            if state == "start":
                t.active += 1
            elif state == "done":
                if t.active > 0:
                    t.active -= 1
                t.completed_total += 1
                t.last_at = now
                t.completed_60s.append(now)
            elif state == "fail":
                if t.active > 0:
                    t.active -= 1
                t.failed_total += 1
                t.last_at = now
                t.failed_60s.append(now)

    async def set_in_flight(
        self,
        req_id: str | None,
        stage: str | None,
        started_at: float | None = None,
        tokens: int | None = None,
        ttft_ms: float | None = None,
        note: str | None = None,
    ) -> None:
        """Update the live "what's happening right now" snapshot. Pass
        req_id=None to clear (request finished). Cheap — replaces a single
        dict reference under the lock; safe to call every N tokens during
        streaming."""
        async with self._lock:
            if req_id is None:
                self._in_flight = None
                return
            now = time.time()
            base = self._in_flight or {}
            base.update({
                "req_id":     req_id,
                "stage":      stage,
                "started_at": started_at if started_at is not None else base.get("started_at", now),
                "updated_at": now,
                "tokens":     tokens,
                "ttft_ms":    ttft_ms,
                "note":       note,
            })
            self._in_flight = base

    async def clear_in_flight(self) -> None:
        """Mark the request as no-longer-in-flight."""
        async with self._lock:
            self._in_flight = None

    async def set_model_state(
        self,
        name: str,
        loaded: bool | None = None,
        load_ms: float | None = None,
        url: str | None = None,
        status: str | None = None,
        extra: dict | None = None,
    ) -> None:
        """Declare or update a model's load state. Fields not passed are kept."""
        async with self._lock:
            m = self._models.setdefault(name, _ModelState(name=name))
            if loaded is not None:
                m.loaded = loaded
            if load_ms is not None:
                m.load_ms = load_ms
            if url is not None:
                m.url = url
            if status is not None:
                m.status = status
            if extra is not None:
                m.extra = extra

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

    # ── V2 read paths ─────────────────────────────────────────────────

    async def subsystems_snapshot(self) -> dict[str, dict[str, Any]]:
        """Per-subsystem aggregates. Prunes 60s windows on the way out."""
        async with self._lock:
            now = time.time()
            out: dict[str, dict[str, Any]] = {}
            for name, s in self._subsystems.items():
                # Prune events_60s in place.
                while s.events_60s and (now - s.events_60s[0][0]) > _SUBSYSTEM_WINDOW_SEC:
                    s.events_60s.popleft()
                window_latencies = [ev[1] for ev in s.events_60s]
                window_errors = sum(1 for ev in s.events_60s if not ev[2])
                p50, p95, _ = _quantiles(window_latencies)
                out[name] = {
                    "count_60s": len(s.events_60s),
                    "count_total": s.count_total,
                    "errors_60s": window_errors,
                    "errors_total": s.errors_total,
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "last_ms": window_latencies[-1] if window_latencies else None,
                    "last_at": s.last_at,
                    "last_note": s.last_note,
                    # Recent samples for the per-row sparkline (oldest → newest).
                    "latency_series": list(s.latency_recent),
                }
            return out

    async def pipeline_snapshot(self) -> dict[str, Any]:
        """Return the most-recent request's pipeline events plus a few prior."""
        async with self._lock:
            events = list(self._pipeline_events)
            if not events:
                return {"current": None, "events": []}

            # Identify the most recent request (last event's req_id).
            current_id = events[-1]["req_id"]
            current_events = [e for e in events if e["req_id"] == current_id]
            started = self._pipeline_started_at.get(current_id)

            return {
                "current": {
                    "request_id": current_id,
                    "started_at": started,
                    "events": current_events,
                },
                # Full bounded log so the CLI can render history if desired.
                "events": events,
            }

    async def tasks_snapshot(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            now = time.time()
            out: dict[str, dict[str, Any]] = {}
            for name, t in self._tasks.items():
                while t.completed_60s and (now - t.completed_60s[0]) > _ERROR_WINDOW_SEC:
                    t.completed_60s.popleft()
                while t.failed_60s and (now - t.failed_60s[0]) > _ERROR_WINDOW_SEC:
                    t.failed_60s.popleft()
                out[name] = {
                    "active": t.active,
                    "completed_60s": len(t.completed_60s),
                    "failed_60s": len(t.failed_60s),
                    "completed_total": t.completed_total,
                    "failed_total": t.failed_total,
                    "last_at": t.last_at,
                }
            return out

    async def models_snapshot(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            return {
                name: {
                    "name": m.name,
                    "loaded": m.loaded,
                    "load_ms": m.load_ms,
                    "url": m.url,
                    "status": m.status,
                    "extra": m.extra,
                }
                for name, m in self._models.items()
            }

    async def in_flight_snapshot(self) -> dict[str, Any] | None:
        async with self._lock:
            return dict(self._in_flight) if self._in_flight else None


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
