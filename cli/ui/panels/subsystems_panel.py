"""Subsystems panel — rolling rate + latency table per instrumented subsystem.

One row per `(name, …)` entry from `state.raw["subsystems"]`. Sorted by
recent activity descending so the busy stuff floats to the top. Empty
trailing rows fold into a "+N idle" footer so the layout stays tight.

Sparkline width and visible-row count adapt to `width_hint`/`height_hint`
from the caller — the same panel module handles Compact, Standard, Wide
and Ultra-wide layouts without per-mode forks.
"""

from __future__ import annotations

import time

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.ui.animation import Sparkline


def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms/1000:.1f}s"


def _spark_for(values: list[float], width: int) -> str:
    if not values:
        return " " * width
    s = Sparkline(width=width)
    s.replace(values)
    return s.render()


def _activity_score(stats: dict) -> tuple:
    """Sort key: rows with traffic in the last 60s float to the top, then
    by total count, then by name (stable for empty rows)."""
    return (
        -(stats.get("count_60s") or 0),
        -(stats.get("count_total") or 0),
        stats.get("__name", ""),
    )


def render(state, width_hint: int = 60, height_hint: int = 10) -> Panel:
    subs = state.raw.get("subsystems") or {}

    # Adaptive sparkline width: panel_width minus all the fixed columns + padding.
    fixed_cols = 28 + 7 + 7 + 5    # name + 60s + p50 + p95 + err + padding (rough)
    spark_width = max(8, min(40, width_hint - fixed_cols))

    rows: list[dict] = []
    for name, s in subs.items():
        rows.append({**s, "__name": name})

    rows.sort(key=_activity_score)
    max_visible = max(3, height_hint - 3)
    visible = rows[:max_visible]
    hidden_count = len(rows) - len(visible)

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(no_wrap=True, overflow="ellipsis", min_width=18)   # name
    tbl.add_column(justify="right", no_wrap=True, min_width=4)        # 60s count
    tbl.add_column(justify="right", no_wrap=True, min_width=6)        # p50
    tbl.add_column(justify="right", no_wrap=True, min_width=6)        # p95
    tbl.add_column(justify="right", no_wrap=True, min_width=4)        # err
    tbl.add_column(no_wrap=True)                                      # sparkline

    if not rows:
        tbl.add_row("[dim]— no subsystem traffic yet —[/dim]", "", "", "", "", "")
        pulse = state.pulses.get("subsystems", state.pulses["request"])
        return Panel(tbl, title=Text("SUBSYSTEMS", style=pulse.style()),
                     border_style=pulse.style(), padding=(0, 1))

    # Header row (kept dim so it doesn't dominate).
    tbl.add_row(
        "[dim]name[/dim]",
        "[dim]60s[/dim]",
        "[dim]p50[/dim]",
        "[dim]p95[/dim]",
        "[dim]err[/dim]",
        "[dim]trend[/dim]",
    )

    now = time.time()
    for r in visible:
        name = r["__name"]
        count60 = r.get("count_60s") or 0
        p50 = r.get("p50_ms")
        p95 = r.get("p95_ms")
        err = r.get("errors_60s") or 0
        last_at = r.get("last_at")
        is_stale = last_at is None or (now - last_at) > 30.0
        name_style = "dim" if is_stale else ""
        name_markup = f"[{name_style}]{name}[/{name_style}]" if name_style else name

        err_markup = f"[red]{err}[/red]" if err else f"[dim]{err}[/dim]"

        spark = _spark_for(r.get("latency_series") or [], spark_width)
        # Colour the sparkline cyan when fresh, dim when stale.
        spark_markup = f"[dim]{spark}[/dim]" if is_stale else f"[cyan]{spark}[/cyan]"

        tbl.add_row(
            name_markup,
            str(count60) if count60 else "[dim]0[/dim]",
            _fmt_ms(p50),
            _fmt_ms(p95),
            err_markup,
            spark_markup,
        )

    if hidden_count > 0:
        tbl.add_row(f"[dim]+ {hidden_count} idle[/dim]", "", "", "", "", "")

    pulse = state.pulses.get("subsystems", state.pulses["request"])
    title = Text("SUBSYSTEMS", style=pulse.style())
    return Panel(tbl, title=title, border_style=pulse.style(), padding=(0, 1))
