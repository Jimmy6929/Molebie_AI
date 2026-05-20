"""Backend pool, selector, and per-backend health for inference routing.

Replaces the old single-URL-per-tier model in InferenceService with a small
abstraction that holds *one or more* backends per tier and decides which one
gets each request. Today the pools have one entry each (the local instant +
thinking endpoints). Once Plan B compute satellites join, they're appended
to the pool and the selector picks among them — no further refactor needed.

Three things this module owns:

1. **BackendHealth** — a three-state circuit breaker with retry budgets,
   driven by inference-call outcomes (not yet by probe data; that wiring is
   a separate branch). Same shape as Hystrix / resilience4j / Envoy outlier
   detection.

2. **BackendPool** — a trivial container of `InferenceBackend` per tier.

3. **BackendSelector** — the routing decision tree: filter by health and
   retry budget, honor session affinity for KV-cache reuse, voice-mode picks
   the closest backend, otherwise least-loaded. Cross-tier fallback when a
   tier has no healthy backends and the operator allowed it.

State placeholders `CircuitState.LOADING` and `CircuitState.COLD` exist in
the enum but have no transitions yet — they belong to follow-up branches
(post-join self-check and idle detection, respectively).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(str, Enum):
    """Backend availability state. Three live states + two reserved.

    Live states drive routing decisions today:
      * CLOSED    — healthy; all traffic eligible.
      * HALF_OPEN — recently failed; one trial selection allowed.
      * OPEN      — circuit tripped; no traffic until cooldown elapses.

    Reserved states (no transitions yet):
      * LOADING   — post-join self-check pending. Set by future join flow.
      * COLD      — healthy but idle 5+ minutes. Driven by probe data (future).
    """

    CLOSED = "closed"
    HALF_OPEN = "half_open"
    OPEN = "open"
    LOADING = "loading"
    COLD = "cold"


# Tunables — picked to match Hystrix / Envoy defaults that have proven sane
# in production routing layers. Adjust at the selector boundary, not here.
_FAILURE_THRESHOLD = 3
# Number of failures within ``_ERROR_WINDOW_SEC`` that trips the circuit even
# if they weren't all consecutive (a flapping backend with intermittent
# successes still gets opened). Real per-request error-rate computation
# requires per-attempt sample storage and arrives with the BackendProbe
# extension branch; until then this density trigger covers the gap.
_WINDOW_FAILURE_THRESHOLD = 4
_ERROR_WINDOW_SEC = 60.0
_INITIAL_COOLDOWN_SEC = 30.0
_MAX_COOLDOWN_SEC = 5 * 60.0  # cap exponential backoff at 5 minutes
_RETRY_BUDGET_MAX = 5
_RETRY_BUDGET_REFILL_SEC = 12.0  # 5 retries per minute steady state


@dataclass
class BackendHealth:
    """Per-backend circuit breaker + retry budget.

    Mutated synchronously inside a single asyncio loop — no locks needed.
    """

    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    last_failure_at: float | None = None
    open_until: float | None = None
    consecutive_trips: int = 0  # for exponential backoff on repeat trips
    retry_budget_remaining: float = float(_RETRY_BUDGET_MAX)
    # 0.0 sentinel — the first refill call will compute elapsed from whatever
    # clock the caller passes, which works for both production (time.monotonic)
    # and tests (explicit `now`). Default-factory time.monotonic() leaks the
    # construction clock into elapsed math and breaks deterministic tests.
    last_retry_refill_at: float = 0.0
    # Rolling failure timestamps within _ERROR_WINDOW_SEC. Used to compute
    # error_rate. Append on every failure; we don't track successes here
    # because the rate is "failures per attempt" and attempt count is read
    # via in_flight/throughput from elsewhere — for the breaker we use a
    # simpler "failure density" trigger: if N failures occurred in window,
    # transition. Keeps memory bounded and avoids needing per-call sample
    # storage.
    _failure_window: deque = field(default_factory=lambda: deque(maxlen=64))

    def record_success(self, now: float | None = None) -> None:
        """A real inference call succeeded against this backend."""
        now = now if now is not None else time.monotonic()
        self._refill_retry_budget(now)
        # Any success resets the breaker. Clear the failure window too so a
        # post-success burst can't trip the rate-based threshold via stale
        # failures from before the success.
        self.consecutive_failures = 0
        self._failure_window.clear()
        if self.state in (CircuitState.OPEN, CircuitState.HALF_OPEN):
            self.state = CircuitState.CLOSED
            self.open_until = None
            self.consecutive_trips = 0

    def record_failure(self, now: float | None = None) -> None:
        """A real inference call against this backend failed (timeout, 5xx,
        connection error). Drives the CLOSED → OPEN transition and consumes
        one slot from the retry budget."""
        now = now if now is not None else time.monotonic()
        self._refill_retry_budget(now)
        self.consecutive_failures += 1
        self.last_failure_at = now
        self._failure_window.append(now)
        # Retry budget tracks *failures* within the window, not successful
        # selections. A misbehaving backend bleeds budget; a healthy one
        # keeps it topped up.
        self.retry_budget_remaining = max(0.0, self.retry_budget_remaining - 1.0)

        if self.state == CircuitState.HALF_OPEN:
            # Trial probe failed → re-open with extended cooldown.
            self._trip_open(now, extended=True)
            return

        if self.state == CircuitState.CLOSED:
            window_failures = self._recent_failure_count(now)
            if (
                self.consecutive_failures >= _FAILURE_THRESHOLD
                or window_failures >= _WINDOW_FAILURE_THRESHOLD
            ):
                self._trip_open(now, extended=False)

    def is_eligible(self, now: float | None = None) -> bool:
        """Can this backend receive a real request right now?

        Side effect: may transition OPEN → HALF_OPEN if the cooldown elapsed.
        Side effect: may transition HALF_OPEN → CLOSED only via record_success.
        """
        now = now if now is not None else time.monotonic()
        if self.state == CircuitState.OPEN:
            if self.open_until is not None and now >= self.open_until:
                self.state = CircuitState.HALF_OPEN
                self.open_until = None
                return True
            return False
        if self.state in (CircuitState.LOADING,):
            return False
        # CLOSED, HALF_OPEN, COLD are all eligible
        return True

    # ------------------------------ internals ------------------------------

    def _recent_failure_count(self, now: float) -> int:
        cutoff = now - _ERROR_WINDOW_SEC
        # deque is in insertion order; drop stale from the left.
        while self._failure_window and self._failure_window[0] < cutoff:
            self._failure_window.popleft()
        return len(self._failure_window)

    def _trip_open(self, now: float, extended: bool) -> None:
        self.consecutive_trips += 1
        # Exponential backoff: 30s, 60s, 120s, 240s, capped at 5 min.
        # `extended=True` (HALF_OPEN trial failed) adds one extra doubling.
        exponent = self.consecutive_trips + (1 if extended else 0)
        cooldown = min(_INITIAL_COOLDOWN_SEC * (2 ** (exponent - 1)), _MAX_COOLDOWN_SEC)
        self.state = CircuitState.OPEN
        self.open_until = now + cooldown

    def _refill_retry_budget(self, now: float) -> None:
        elapsed = now - self.last_retry_refill_at
        if elapsed <= 0:
            return
        refill = elapsed / _RETRY_BUDGET_REFILL_SEC
        self.retry_budget_remaining = min(
            self.retry_budget_remaining + refill, float(_RETRY_BUDGET_MAX)
        )
        self.last_retry_refill_at = now


@dataclass
class InferenceBackend:
    """One endpoint that can serve inference for a given tier.

    `node_id` is the stable handle used by the affinity table — must be unique
    across all pools. Use `local-<tier>` for the local backend and
    `satellite-<host>-<tier>` for future satellites.
    """

    url: str
    api_prefix: str
    model: str
    node_id: str
    tier: str
    health: BackendHealth = field(default_factory=BackendHealth)
    in_flight: int = 0
    avg_latency_ms: float = 0.0
    last_inference_at: float | None = None


@dataclass
class BackendPool:
    """All backends serving a single tier (instant or thinking)."""

    tier: str
    backends: list[InferenceBackend] = field(default_factory=list)

    def eligible(self, now: float | None = None) -> list[InferenceBackend]:
        """Backends with a healthy circuit AND retry budget remaining.

        The retry budget shrinks on failures (see ``BackendHealth.record_failure``)
        and refills over time; a backend that has burned through its tolerance
        is skipped even if its circuit state would otherwise allow traffic.
        Prevents retry storms during partial outages (Google SRE-book pattern)."""
        now = now if now is not None else time.monotonic()
        return [
            b for b in self.backends
            if b.health.is_eligible(now)
            and b.health.retry_budget_remaining >= 1.0
        ]


class NoHealthyBackendError(RuntimeError):
    """Raised by BackendSelector when no backend can serve a request.

    Carries the requested mode and per-tier-per-backend state so the caller
    can surface a useful error to the user."""

    def __init__(self, mode: str, attempted: list[tuple[str, CircuitState]]):
        self.mode = mode
        self.attempted = attempted
        summary = ", ".join(f"{n}={s.value}" for n, s in attempted) or "(empty pools)"
        super().__init__(f"No healthy backend for mode={mode}: {summary}")


class BackendSelector:
    """Routes a request to one backend across one or more per-tier pools.

    Routing decision tree (Compute Ext doc §"Routing policy in detail"):
      1. Get the pool for `mode`. If empty → cross-tier fallback (if enabled)
         or NoHealthyBackendError.
      2. Filter pool to backends with eligible health + retry budget.
      3. If session affinity binding exists and that backend is in the
         eligible set → return it. KV cache wins.
      4. Voice mode → pick lowest avg_latency_ms.
      5. Otherwise → least-loaded by `in_flight`. CLOSED beats HALF_OPEN.
      6. Record the affinity binding so follow-up turns stick.

    `record_success` / `record_failure` are called by the inference service
    after each call — these drive the per-backend circuit breaker.
    """

    def __init__(
        self,
        pools: dict[str, BackendPool],
        *,
        fallback_to_instant: bool = True,
    ) -> None:
        self.pools = pools
        self.fallback_to_instant = fallback_to_instant
        self.affinity: dict[str, str] = {}

    # ----- selection -----

    def select(
        self,
        mode: str,
        session_id: str | None = None,
        voice_mode: bool = False,
    ) -> InferenceBackend:
        tier = _tier_for_mode(mode)
        now = time.monotonic()

        backend = self._select_within_tier(tier, session_id, voice_mode, now)
        if backend is not None:
            self._consume_and_record(backend, session_id, now)
            return backend

        # Cross-tier fallback: requested thinking but no eligible thinking
        # backend; try instant if the operator allowed it.
        if tier == "thinking" and self.fallback_to_instant:
            backend = self._select_within_tier("instant", session_id, voice_mode, now)
            if backend is not None:
                self._consume_and_record(backend, session_id, now)
                return backend

        raise NoHealthyBackendError(mode, self._attempted_state(tier))

    def _select_within_tier(
        self,
        tier: str,
        session_id: str | None,
        voice_mode: bool,
        now: float,
    ) -> InferenceBackend | None:
        pool = self.pools.get(tier)
        if pool is None or not pool.backends:
            return None

        candidates = pool.eligible(now)
        if not candidates:
            return None

        # 1. Session affinity: same backend → same KV cache.
        if session_id is not None and (pinned := self.affinity.get(session_id)):
            for b in candidates:
                if b.node_id == pinned:
                    return b
            # The pinned backend is no longer eligible — drop the binding so
            # a fresh pick can be recorded.
            self.affinity.pop(session_id, None)

        # 2. Voice mode → minimize latency (CLOSED preferred).
        if voice_mode:
            closed = [b for b in candidates if b.health.state == CircuitState.CLOSED]
            pool_to_sort = closed or candidates
            return min(pool_to_sort, key=lambda b: (b.avg_latency_ms, b.in_flight))

        # 3. Least-load by in_flight. CLOSED beats HALF_OPEN so we don't
        #    accidentally route real traffic to a probe-only state.
        closed = [b for b in candidates if b.health.state == CircuitState.CLOSED]
        pool_to_sort = closed or candidates
        return min(pool_to_sort, key=lambda b: b.in_flight)

    def _consume_and_record(
        self,
        backend: InferenceBackend,
        session_id: str | None,
        now: float,
    ) -> None:
        # Selection itself does NOT consume retry budget — only failures do
        # (see BackendHealth.record_failure). The budget tracks how much
        # tolerance we have for a misbehaving backend, not raw selection rate.
        if session_id is not None:
            self.affinity[session_id] = backend.node_id

    def _attempted_state(self, tier: str) -> list[tuple[str, CircuitState]]:
        out: list[tuple[str, CircuitState]] = []
        for t in (tier, "instant") if tier == "thinking" else (tier,):
            pool = self.pools.get(t)
            if not pool:
                continue
            for b in pool.backends:
                out.append((b.node_id, b.health.state))
        return out

    # ----- outcome reporting -----

    def record_success(
        self,
        node_id: str,
        latency_ms: float | None = None,
        now: float | None = None,
    ) -> None:
        backend = self._lookup(node_id)
        if backend is None:
            return
        now = now if now is not None else time.monotonic()
        backend.health.record_success(now)
        backend.last_inference_at = now
        if latency_ms is not None:
            # Simple EWMA on latency, alpha=0.2 — smooth enough that a single
            # cold-start spike doesn't dominate the average.
            if backend.avg_latency_ms == 0.0:
                backend.avg_latency_ms = latency_ms
            else:
                backend.avg_latency_ms = (
                    0.8 * backend.avg_latency_ms + 0.2 * latency_ms
                )

    def record_failure(self, node_id: str, now: float | None = None) -> None:
        backend = self._lookup(node_id)
        if backend is None:
            return
        backend.health.record_failure(now)
        # Invalidate any affinity bindings that pointed at this backend so
        # follow-up turns get a fresh pick.
        if backend.health.state == CircuitState.OPEN:
            for sid, nid in list(self.affinity.items()):
                if nid == node_id:
                    self.affinity.pop(sid, None)

    def _lookup(self, node_id: str) -> InferenceBackend | None:
        for pool in self.pools.values():
            for b in pool.backends:
                if b.node_id == node_id:
                    return b
        return None


def _tier_for_mode(mode: str) -> str:
    """Map a request mode to the tier it should route to."""
    if mode in ("thinking", "thinking_harder"):
        return "thinking"
    return "instant"
