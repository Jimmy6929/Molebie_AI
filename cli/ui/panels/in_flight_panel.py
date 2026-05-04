"""In-flight banner — single-row "currently streaming" status.

Renders only when `state.raw['in_flight']` is non-null. Shows the live
stage, elapsed seconds, accumulated token count, and (once stamped) TTFT.
This is the "what's happening RIGHT NOW" axis paired with the Pipeline
panel's per-stage timeline.

When idle, render() returns None and the layout caller skips reserving
space — the panel never shows an empty placeholder, since "no in-flight"
is the resting state most operators see most of the time.
"""

from __future__ import annotations

import time

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms/1000:.1f} s"


def render(state) -> Panel | None:
    inflight = state.raw.get("in_flight")
    if not inflight:
        return None

    started = inflight.get("started_at") or time.time()
    elapsed = max(0.0, time.time() - float(started))
    stage = inflight.get("stage") or "?"
    tokens = inflight.get("tokens")
    ttft = inflight.get("ttft_ms")
    note = inflight.get("note") or ""
    req_id = inflight.get("req_id") or "—"

    tokens_str = f"{tokens} tok" if tokens else "[dim]waiting[/dim]"
    ttft_str = _fmt_ms(ttft) if ttft else "[dim]—[/dim]"

    # Tokens-per-second: only after the first token has stamped a TTFT.
    tps_str = "—"
    if tokens and ttft is not None and elapsed * 1000 > ttft:
        decode_sec = elapsed - (ttft / 1000.0)
        if decode_sec > 0:
            tps_str = f"{tokens / decode_sec:.1f} tok/s"

    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(no_wrap=True)
    tbl.add_column(no_wrap=True)
    tbl.add_column(no_wrap=True)
    tbl.add_column(no_wrap=True)
    tbl.add_column(no_wrap=True)
    tbl.add_row(
        f"[cyan]▶[/cyan] [bold]{stage}[/bold]",
        f"[dim]req[/dim] [cyan]{str(req_id)[:6]}[/cyan]",
        f"[dim]elapsed[/dim] [cyan]{elapsed:.1f}s[/cyan]",
        f"[dim]ttft[/dim] {ttft_str}",
        f"[dim]·[/dim] {tokens_str} [dim]·[/dim] {tps_str} [dim]·[/dim] {note}",
    )

    title = Text("⚡ IN FLIGHT", style="cyan bold")
    return Panel(tbl, title=title, border_style="cyan", padding=(0, 1))
