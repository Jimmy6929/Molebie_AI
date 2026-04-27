"""Recent-activity panel: chronological log of the last 5 requests.

The panel title says what this is, so we don't waste a row on a sub-header.
Error and fallback counters used to live here; they've moved into the
Requests panel where they belong alongside the other rate metrics.
"""

from __future__ import annotations

from datetime import datetime

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def render(state) -> Panel:
    req = state.raw.get("requests", {}) or {}
    recent = req.get("recent_events", []) or []

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(no_wrap=True)   # glyph
    tbl.add_column(no_wrap=True)   # time
    tbl.add_column(no_wrap=True, min_width=9)   # tier
    tbl.add_column(justify="right", no_wrap=True, min_width=8)   # tokens
    tbl.add_column(justify="right", no_wrap=True, min_width=8)   # latency
    tbl.add_column(no_wrap=True, overflow="crop")  # note (fallback, error)

    if not recent:
        tbl.add_row("", "[dim]— no traffic yet —[/dim]", "", "", "", "")
        title = Text("RECENT ACTIVITY", style=state.pulses["quality"].style())
        return Panel(tbl, title=title, border_style=state.pulses["quality"].style(), padding=(0, 1))

    for ev in reversed(recent):   # newest at top
        ok = ev.get("ok", True)
        glyph = "[green]✓[/green]" if ok else "[red]✗[/red]"
        ts = datetime.fromtimestamp(ev.get("ts") or 0).strftime("%H:%M:%S")
        tier = ev.get("tier") or "?"
        tokens = ev.get("completion_tokens")
        tokens_str = f"{tokens} tok" if tokens else "—"
        ms = ev.get("total_ms")
        ms_str = f"{ms:.0f} ms" if ms is not None else "—"

        note_parts = []
        if ev.get("fallback"):
            note_parts.append("[yellow]fallback→instant[/yellow]")
        if not ok:
            note_parts.append("[red]err[/red]")
        note = "  ".join(note_parts)

        tbl.add_row(
            glyph,
            f"[dim]{ts}[/dim]",
            tier,
            tokens_str,
            ms_str,
            note,
        )

    title = Text("RECENT ACTIVITY", style=state.pulses["quality"].style())
    return Panel(tbl, title=title, border_style=state.pulses["quality"].style(), padding=(0, 1))
