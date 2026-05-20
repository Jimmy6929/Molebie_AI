"""Tests for BackendHealth (circuit breaker), BackendPool, and BackendSelector.

The selector is the routing chokepoint in Plan B compute. These tests cover:
  * Circuit-breaker transitions (CLOSED → OPEN → HALF_OPEN → CLOSED / re-OPEN)
  * Retry budgets (decrement, refill over time, exhaustion blocks selection)
  * Selector routing (single-backend, multi-backend, affinity, voice-mode,
    cross-tier fallback, error path)

Time is mocked via explicit `now` arguments so transitions are deterministic.
"""

from __future__ import annotations

import pytest

from app.services.inference_pool import (
    BackendHealth,
    BackendPool,
    BackendSelector,
    CircuitState,
    InferenceBackend,
    NoHealthyBackendError,
)

# ─────────────────────────── BackendHealth ───────────────────────────


class TestBackendHealth:
    def test_starts_closed(self):
        h = BackendHealth()
        assert h.state == CircuitState.CLOSED
        assert h.is_eligible(now=0.0)

    def test_three_consecutive_failures_trip_to_open(self):
        h = BackendHealth()
        for _ in range(3):
            h.record_failure(now=0.0)
        assert h.state == CircuitState.OPEN
        assert h.open_until is not None
        assert not h.is_eligible(now=0.0)

    def test_two_failures_alone_do_not_trip(self):
        h = BackendHealth()
        h.record_failure(now=0.0)
        h.record_failure(now=0.0)
        assert h.state == CircuitState.CLOSED

    def test_success_resets_consecutive_failures(self):
        h = BackendHealth()
        h.record_failure(now=0.0)
        h.record_failure(now=0.0)
        h.record_success(now=0.0)
        h.record_failure(now=0.0)
        h.record_failure(now=0.0)
        # Two more failures alone shouldn't trip — counter reset.
        assert h.state == CircuitState.CLOSED

    def test_open_transitions_to_half_open_after_cooldown(self):
        h = BackendHealth()
        for _ in range(3):
            h.record_failure(now=0.0)
        assert h.state == CircuitState.OPEN
        # Cooldown is 30s by default. Just before:
        assert not h.is_eligible(now=29.0)
        assert h.state == CircuitState.OPEN
        # After:
        assert h.is_eligible(now=30.0)
        assert h.state == CircuitState.HALF_OPEN

    def test_success_in_half_open_closes_circuit(self):
        h = BackendHealth()
        for _ in range(3):
            h.record_failure(now=0.0)
        # Trigger HALF_OPEN
        h.is_eligible(now=30.0)
        h.record_success(now=31.0)
        assert h.state == CircuitState.CLOSED
        assert h.consecutive_failures == 0
        assert h.consecutive_trips == 0

    def test_failure_in_half_open_extends_cooldown(self):
        h = BackendHealth()
        for _ in range(3):
            h.record_failure(now=0.0)
        first_open_until = h.open_until
        assert first_open_until is not None
        h.is_eligible(now=30.0)  # → HALF_OPEN
        h.record_failure(now=31.0)  # trial failed → OPEN with extended cooldown
        assert h.state == CircuitState.OPEN
        # Cooldown should be longer than the first one.
        assert h.open_until is not None
        # Second cooldown = 30 * 2^(trips + 1 extended). trips=2 here, extended=True
        # → 30 * 2^2 = 120. Pre-trial open_until was 31 + 30 = 61; new should be 31 + 120 = 151.
        assert h.open_until > first_open_until

    def test_retry_budget_decrements_on_failure_not_on_select(self):
        h = BackendHealth()
        # The budget is for tolerance to misbehavior — failures drain it,
        # not selections. After 5 failures the budget is exhausted.
        for _ in range(5):
            h.record_failure(now=0.0)
        assert h.retry_budget_remaining == 0.0

    def test_retry_budget_refills_over_time(self):
        h = BackendHealth()
        # Drain via failures (also trips the breaker, but that's separate).
        for _ in range(5):
            h.record_failure(now=0.0)
        assert h.retry_budget_remaining == 0.0
        # Refill rate is 1/12s. After 12s a single slot returns.
        h._refill_retry_budget(now=12.0)
        assert h.retry_budget_remaining == pytest.approx(1.0)
        # Just before 24s, still under 2.
        h._refill_retry_budget(now=23.9)
        assert h.retry_budget_remaining < 2.0
        # At 24s, exactly 2 (capped against the consume side, none here).
        h._refill_retry_budget(now=24.0)
        assert h.retry_budget_remaining == pytest.approx(2.0)

    def test_open_skips_eligibility(self):
        h = BackendHealth()
        for _ in range(3):
            h.record_failure(now=0.0)
        assert not h.is_eligible(now=5.0)

    def test_loading_state_not_eligible(self):
        h = BackendHealth(state=CircuitState.LOADING)
        assert not h.is_eligible(now=0.0)


# ─────────────────────────── BackendSelector ───────────────────────────


def _make_backend(node_id: str, tier: str, url: str = "http://x") -> InferenceBackend:
    return InferenceBackend(
        url=url,
        api_prefix="/v1",
        model="dummy",
        node_id=node_id,
        tier=tier,
    )


def _make_selector(*backends: InferenceBackend, fallback: bool = True) -> BackendSelector:
    pools: dict[str, BackendPool] = {
        "instant": BackendPool("instant"),
        "thinking": BackendPool("thinking"),
    }
    for b in backends:
        pools[b.tier].backends.append(b)
    return BackendSelector(pools, fallback_to_instant=fallback)


class TestBackendSelector:
    def test_empty_pool_raises_no_healthy_backend(self):
        sel = _make_selector()  # empty pools
        with pytest.raises(NoHealthyBackendError):
            sel.select("thinking")

    def test_single_backend_always_returned(self):
        sel = _make_selector(_make_backend("local-instant", "instant"))
        assert sel.select("instant").node_id == "local-instant"

    def test_session_affinity_sticks(self):
        a = _make_backend("a", "thinking")
        b = _make_backend("b", "thinking")
        sel = _make_selector(a, b)
        first = sel.select("thinking", session_id="sess-1")
        for _ in range(5):
            again = sel.select("thinking", session_id="sess-1")
            assert again.node_id == first.node_id, "affinity must keep returning the same backend"

    def test_affinity_invalidated_when_backend_opens(self):
        a = _make_backend("a", "thinking")
        b = _make_backend("b", "thinking")
        sel = _make_selector(a, b)
        chosen = sel.select("thinking", session_id="s1")
        # Trip the chosen backend
        for _ in range(3):
            sel.record_failure(chosen.node_id)
        assert chosen.health.state == CircuitState.OPEN
        # Affinity for s1 should now be dropped — second select picks the other.
        new_choice = sel.select("thinking", session_id="s1")
        assert new_choice.node_id != chosen.node_id

    def test_voice_mode_picks_lowest_latency(self):
        a = _make_backend("a", "instant")
        b = _make_backend("b", "instant")
        a.avg_latency_ms = 500.0
        b.avg_latency_ms = 120.0
        sel = _make_selector(a, b)
        chosen = sel.select("instant", voice_mode=True)
        assert chosen.node_id == "b"

    def test_least_load_breaks_tie(self):
        a = _make_backend("a", "instant")
        b = _make_backend("b", "instant")
        a.in_flight = 5
        b.in_flight = 1
        sel = _make_selector(a, b)
        chosen = sel.select("instant")
        assert chosen.node_id == "b"

    def test_cross_tier_fallback_when_thinking_pool_empty(self):
        a = _make_backend("local-instant", "instant")
        sel = _make_selector(a, fallback=True)
        # Asking for thinking should fall back to instant.
        chosen = sel.select("thinking")
        assert chosen.node_id == "local-instant"

    def test_cross_tier_fallback_when_thinking_pool_all_open(self):
        t = _make_backend("local-thinking", "thinking")
        i = _make_backend("local-instant", "instant")
        sel = _make_selector(t, i, fallback=True)
        for _ in range(3):
            sel.record_failure("local-thinking")
        assert t.health.state == CircuitState.OPEN
        chosen = sel.select("thinking")
        assert chosen.node_id == "local-instant"

    def test_no_fallback_when_disabled_and_thinking_open(self):
        t = _make_backend("local-thinking", "thinking")
        i = _make_backend("local-instant", "instant")
        sel = _make_selector(t, i, fallback=False)
        for _ in range(3):
            sel.record_failure("local-thinking")
        with pytest.raises(NoHealthyBackendError):
            sel.select("thinking")

    def test_record_success_clears_open_state(self):
        a = _make_backend("a", "thinking")
        sel = _make_selector(a)
        for _ in range(3):
            sel.record_failure("a")
        assert a.health.state == CircuitState.OPEN
        sel.record_success("a", latency_ms=100.0)
        assert a.health.state == CircuitState.CLOSED

    def test_record_success_updates_latency_ewma(self):
        a = _make_backend("a", "instant")
        sel = _make_selector(a)
        sel.record_success("a", latency_ms=100.0)
        assert a.avg_latency_ms == 100.0  # first sample sets directly
        sel.record_success("a", latency_ms=200.0)
        # EWMA alpha=0.2 → 0.8 * 100 + 0.2 * 200 = 120
        assert a.avg_latency_ms == pytest.approx(120.0)

    def test_record_success_on_unknown_node_is_noop(self):
        sel = _make_selector(_make_backend("a", "instant"))
        sel.record_success("nonexistent")  # must not raise
        sel.record_failure("nonexistent")  # must not raise

    def test_no_healthy_backend_error_carries_state(self):
        t = _make_backend("local-thinking", "thinking")
        sel = _make_selector(t, fallback=False)
        for _ in range(3):
            sel.record_failure("local-thinking")
        try:
            sel.select("thinking")
        except NoHealthyBackendError as e:
            assert e.mode == "thinking"
            assert ("local-thinking", CircuitState.OPEN) in e.attempted
        else:
            pytest.fail("NoHealthyBackendError not raised")

    def test_mode_thinking_harder_maps_to_thinking_tier(self):
        t = _make_backend("local-thinking", "thinking")
        sel = _make_selector(t)
        chosen = sel.select("thinking_harder")
        assert chosen.node_id == "local-thinking"
