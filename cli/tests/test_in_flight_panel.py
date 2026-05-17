"""Tests for the IN FLIGHT panel's TPS spike floor.

decode_sec below ~50 ms is dominated by inter-token jitter — reporting
tokens/decode_sec there will momentarily flash 4-digit tok/s right after
TTFT lands. The floor keeps the placeholder until enough decode time has
elapsed.
"""

import types

from rich.console import Console

import cli.ui.panels.in_flight_panel as panel


def _render_to_string(monkeypatch, started_at: float, now: float, **inflight) -> str:
    monkeypatch.setattr(panel.time, "time", lambda: now)
    state = types.SimpleNamespace(raw={"in_flight": {"started_at": started_at, **inflight}})
    rendered = panel.render(state)
    assert rendered is not None
    buf = Console(record=True, width=200)
    buf.print(rendered)
    return buf.export_text()


def test_tps_suppressed_when_decode_sec_below_floor(monkeypatch):
    # ttft = 100 ms, elapsed = 101 ms → decode_sec = 1 ms. Should NOT
    # report tok/s; the dash placeholder stays.
    out = _render_to_string(
        monkeypatch,
        started_at=1000.0,
        now=1000.101,
        tokens=1,
        ttft_ms=100.0,
        stage="decode",
    )
    assert "tok/s" not in out, f"TPS leaked at sub-floor decode_sec: {out!r}"


def test_tps_reported_once_decode_sec_clears_floor(monkeypatch):
    # ttft = 100 ms, elapsed = 200 ms → decode_sec = 100 ms (≥ 50 ms).
    # 10 tokens / 0.1 s = 100.0 tok/s.
    out = _render_to_string(
        monkeypatch,
        started_at=1000.0,
        now=1000.200,
        tokens=10,
        ttft_ms=100.0,
        stage="decode",
    )
    assert "100.0 tok/s" in out, f"expected 100.0 tok/s in {out!r}"


def test_tps_never_reports_before_ttft(monkeypatch):
    # No TTFT yet → no tok/s line.
    out = _render_to_string(
        monkeypatch,
        started_at=1000.0,
        now=1000.500,
        tokens=5,
        ttft_ms=None,
        stage="prefill",
    )
    assert "tok/s" not in out
