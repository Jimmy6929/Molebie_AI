"""Pipeline panel — live stage timeline for the most recent chat request.

Shows each subsystem stage (rag.retrieve / web.search / inference.thinking /
verify.judge / …) as it fires, with status glyph + latency + a free-form
note (e.g. "30 candidates", "245 tok"). The Pipeline panel is the
"what is happening NOW" axis of the dashboard — paired with Subsystems
which is the "how is it trending" axis.

Stages are taken from `state.raw["pipeline"]["current"]["events"]` —
written by the gateway via `metrics_registry.pipeline_event()`. The list
is bounded to ~50 events, but each request typically emits 5–10, so the
panel renders cleanly even on long windows.

Adaptive height: caller passes `height_hint`; we show the most recent
`max(3, height_hint - 3)` events with a dim "↑ N earlier" hint above.
"""

from __future__ import annotations

import time

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _glyph_for_status(status: str) -> str:
    return {
        "ok":      "[green]✓[/green]",
        "running": "[cyan]▶[/cyan]",
        "queued":  "[dim]·[/dim]",
        "fail":    "[red]✗[/red]",
    }.get(status, "[dim]?[/dim]")


def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "[dim]…[/dim]"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms/1000:.1f} s"


def render(state, height_hint: int = 12) -> Panel:
    pipeline = state.raw.get("pipeline") or {}
    current = pipeline.get("current") or {}
    events = current.get("events") or []
    req_id = current.get("request_id")
    started = current.get("started_at")

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(no_wrap=True)              # glyph
    tbl.add_column(overflow="ellipsis")       # stage
    tbl.add_column(justify="right", no_wrap=True, min_width=8)  # latency
    tbl.add_column(overflow="ellipsis")       # note

    if not events:
        tbl.add_row("", "[dim]— no requests yet — send a chat to see the timeline —[/dim]", "", "")
        title = Text("PIPELINE", style=state.pulses.get("pipeline", state.pulses["request"]).style())
        return Panel(tbl, title=title, border_style=title.style, padding=(0, 1))

    # Bound visible rows by available height (panel chrome + header rows).
    max_rows = max(3, height_hint - 4)
    if len(events) > max_rows:
        dropped = len(events) - max_rows
        tbl.add_row("", f"[dim]↑ {dropped} earlier stage{'s' if dropped != 1 else ''}[/dim]", "", "")
        events = events[-max_rows:]

    for ev in events:
        glyph = _glyph_for_status(ev.get("status", "ok"))
        stage = ev.get("stage") or "?"
        ms_str = _fmt_ms(ev.get("ms"))
        note = ev.get("note") or ""
        # Soften status colour for stage text on terminals that strip glyphs.
        stage_style = ""
        if ev.get("status") == "fail":
            stage_style = "red"
        elif ev.get("status") == "running":
            stage_style = "cyan"
        stage_markup = f"[{stage_style}]{stage}[/{stage_style}]" if stage_style else stage
        tbl.add_row(glyph, stage_markup, ms_str, f"[dim]{note}[/dim]" if note else "")

    age_str = ""
    if started:
        age = time.time() - float(started)
        age_str = f"  [dim]· {age:.1f}s ago[/dim]"
    pulse = state.pulses.get("pipeline", state.pulses["request"])
    title_text = f"PIPELINE  [dim]req {req_id[:6] if req_id else '—'}[/dim]{age_str}"
    title = Text.from_markup(title_text)
    title.stylize(pulse.style())
    return Panel(tbl, title=title, border_style=pulse.style(), padding=(0, 1))
