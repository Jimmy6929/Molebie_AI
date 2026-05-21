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
# Rolling error-rate trip: if failures/total within ``_ERROR_WINDOW_SEC``
# exceeds ``_ERROR_RATE_TRIP`` AND we have at least ``_ERROR_RATE_MIN_SAMPLES``
# samples, trip the circuit. Min-samples guard prevents two failures with no
# successes (rate = 1.0) from tripping on noise. Both probe outcomes and
# request-path outcomes feed the same window.
_ERROR_WINDOW_SEC = 60.0
_ERROR_RATE_TRIP = 0.25
_ERROR_RATE_MIN_SAMPLES = 4
_ERROR_WINDOW_MAXLEN = 128  # bound memory; >> 60s at typical probe + request rates
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
    # Drift-induced OPEN is sticky: the probe observed a model_fingerprint
    # mismatch (operator swapped the model behind the backend's back) and the
    # backend can't auto-recover until a subsequent probe sees the fingerprint
    # match again. ``is_eligible`` skips the OPEN → HALF_OPEN auto-transition
    # while this flag is set.
    drift_open: bool = False
    # Rolling (timestamp, ok) samples within _ERROR_WINDOW_SEC. Fed by both
    # the probe (every 5s) and the request path (on each inference call).
    # Used by error_rate_60s for rate-based trip decisions. Maxlen bounds
    # memory on heavily-loaded backends; the time-based eviction in
    # _evict_old keeps the rate honest.
    _error_window: deque = field(
        default_factory=lambda: deque(maxlen=_ERROR_WINDOW_MAXLEN)
    )

    def record_success(self, now: float | None = None) -> None:
        """A real inference call succeeded against this backend."""
        now = now if now is not None else time.monotonic()
        self._refill_retry_budget(now)
        self._error_window.append((now, True))
        self._close_circuit(now)

    def record_failure(self, now: float | None = None) -> None:
        """A real inference call against this backend failed (timeout, 5xx,
        connection error). Drives the CLOSED → OPEN transition and consumes
        one slot from the retry budget."""
        now = now if now is not None else time.monotonic()
        self._refill_retry_budget(now)
        self._error_window.append((now, False))
        self.consecutive_failures += 1
        self.last_failure_at = now
        # Retry budget tracks *failures* within the window, not successful
        # selections. A misbehaving backend bleeds budget; a healthy one
        # keeps it topped up.
        self.retry_budget_remaining = max(0.0, self.retry_budget_remaining - 1.0)
        self._maybe_trip_open(now)

    def record_probe_success(self, now: float | None = None) -> None:
        """Health probe poll succeeded against this backend.

        Same state-transition semantics as `record_success` except the retry
        budget isn't consumed/refilled here — that's the request path's
        concern. A probe success in HALF_OPEN closes the circuit (recovery)."""
        now = now if now is not None else time.monotonic()
        self._error_window.append((now, True))
        self._close_circuit(now)

    def record_probe_failure(self, now: float | None = None) -> None:
        """Health probe poll failed (timeout, transport error, non-2xx)."""
        now = now if now is not None else time.monotonic()
        self._error_window.append((now, False))
        self.consecutive_failures += 1
        self.last_failure_at = now
        self._maybe_trip_open(now)

    def mark_fingerprint_drift(self, now: float | None = None) -> None:
        """The probe saw a model_fingerprint mismatch — the backend is serving
        a different model than the pool expects, which would silently corrupt
        results. Trip OPEN immediately and stickily; no cooldown progression
        will free the backend until ``clear_drift`` is called."""
        now = now if now is not None else time.monotonic()
        self.state = CircuitState.OPEN
        self.drift_open = True
        # Sentinel: any time check `now >= open_until` returns False, so
        # is_eligible never auto-transitions. clear_drift is the only escape.
        self.open_until = None
        self.last_failure_at = now

    def clear_drift(self) -> None:
        """Probe observed the fingerprint matching expectations again —
        release the sticky OPEN and reset the breaker. Pure state reset,
        no timestamp needed."""
        self.drift_open = False
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.consecutive_trips = 0
        self.open_until = None

    def error_rate_60s(self, now: float | None = None) -> float:
        """Failures / total samples within the last 60 seconds.

        Returns 0.0 when the window holds fewer than ``_ERROR_RATE_MIN_SAMPLES``
        — keeps the rate from spiking to 1.0 on a single failed probe and
        prevents the breaker from over-reacting on noise.
        """
        now = now if now is not None else time.monotonic()
        self._evict_old(now)
        total = len(self._error_window)
        if total < _ERROR_RATE_MIN_SAMPLES:
            return 0.0
        failures = sum(1 for _, ok in self._error_window if not ok)
        return failures / total

    def is_eligible(self, now: float | None = None) -> bool:
        """Can this backend receive a real request right now?

        Side effect: may transition OPEN → HALF_OPEN if the cooldown elapsed,
        but only when the OPEN isn't drift-induced (drift_open stays sticky).
        Side effect: HALF_OPEN → CLOSED happens only via record_success /
        record_probe_success.
        """
        now = now if now is not None else time.monotonic()
        if self.state == CircuitState.OPEN:
            if self.drift_open:
                return False
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

    def _evict_old(self, now: float) -> None:
        cutoff = now - _ERROR_WINDOW_SEC
        while self._error_window and self._error_window[0][0] < cutoff:
            self._error_window.popleft()

    def _close_circuit(self, now: float) -> None:
        """Shared OPEN/HALF_OPEN → CLOSED transition on any success."""
        # Drift-induced OPEN is sticky — only clear_drift releases it.
        if self.drift_open:
            return
        self.consecutive_failures = 0
        if self.state in (CircuitState.OPEN, CircuitState.HALF_OPEN):
            self.state = CircuitState.CLOSED
            self.open_until = None
            self.consecutive_trips = 0

    def _maybe_trip_open(self, now: float) -> None:
        """Common trip check shared between record_failure and record_probe_failure."""
        if self.state == CircuitState.HALF_OPEN:
            # Trial selection / probe failed → re-open with extended cooldown.
            self._trip_open(now, extended=True)
            return
        if self.state == CircuitState.CLOSED:
            if (
                self.consecutive_failures >= _FAILURE_THRESHOLD
                or self.error_rate_60s(now) > _ERROR_RATE_TRIP
            ):
                self._trip_open(now, extended=False)

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
    # Populated by the BackendProbe on each /v1/models poll. Drift between
    # this and the pool's expected_fingerprint triggers a sticky OPEN.
    model_fingerprint: str | None = None
    # Best-effort: server software version (vLLM, Ollama, MLX, etc.). Sourced
    # from the Server: response header on probe responses where present.
    server_version: str | None = None


@dataclass
class BackendPool:
    """All backends serving a single tier (instant or thinking)."""

    tier: str
    backends: list[InferenceBackend] = field(default_factory=list)
    # Pinned by the first successful probe of any backend in this pool.
    # Subsequent backends that report a different fingerprint are flagged as
    # drifted (the operator swapped weights behind one of them). Cleared on
    # gateway restart — that's the operator's "trust me, I changed it" signal.
    expected_fingerprint: str | None = None

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
