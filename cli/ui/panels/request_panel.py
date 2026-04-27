"""Request-layer panel: first-token time, per-token time, latency quantiles,
throughput, fallbacks, errors. Uses full-name labels with a tiny acronym
suffix so both novices and power users can read it at a glance."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.ui.animation import Sparkline


def _fmt_ms(val: float | None) -> str:
    if val is None:
        return "—"
    if val >= 1000:
        return f"{val/1000:.2f} s"
    return f"{val:.0f} ms"


def render(state) -> Panel:
    req = state.raw.get("requests", {}) or {}
    d = state.display
    ttft_spark: Sparkline = state.sparklines["ttft"]
    tpot_spark: Sparkline = state.sparklines["tpot"]

    # Three columns: label (left, never truncated), value (right-aligned),
    # trend sparkline (crop on narrow terminals — the value is still readable
    # without the spark, so it's the best thing to lose first).
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(no_wrap=True, overflow="fold", min_width=28)
    tbl.add_column(justify="right", no_wrap=True, min_width=10)
    tbl.add_column(no_wrap=True, overflow="crop")

    tbl.add_row(
        "[bold]First-token[/bold] [dim](TTFT)[/dim]",
        f"[cyan]{_fmt_ms(d.ttft_ms)}[/cyan]",
        f"[dim]{ttft_spark.render()}[/dim]",
    )
    tbl.add_row(
        "[bold]Per-token[/bold] [dim](TPOT)[/dim]",
        f"[cyan]{_fmt_ms(d.tpot_ms)}[/cyan]",
        f"[dim]{tpot_spark.render()}[/dim]",
    )
    tbl.add_row(
        "[bold]Median[/bold] [dim](p50)[/dim]",
        f"[cyan]{_fmt_ms(d.total_p50_ms)}[/cyan]",
        "",
    )
    tbl.add_row(
        "[bold]Tail[/bold] [dim](p95)[/dim]",
        f"[cyan]{_fmt_ms(d.total_p95_ms)}[/cyan]",
        "",
    )
    # p99 comes straight from the server snapshot — not tweened because it
    # changes rarely and a smooth lerp on a worst-case outlier is misleading.
    tbl.add_row(
        "[bold]Worst[/bold] [dim](p99)[/dim]",
        f"[cyan]{_fmt_ms(req.get('total_p99_ms'))}[/cyan]",
        "",
    )
    rps = d.req_per_sec or 0.0
    tbl.add_row(
        "[bold]Throughput[/bold]",
        f"[cyan]{rps:.2f}[/cyan] [dim]req/s[/dim]",
        "",
    )
    fallbacks = req.get("fallback_count", 0) or 0
    fb_style = "yellow" if fallbacks > 0 else "cyan"
    tbl.add_row(
        "[bold]Fallbacks[/bold] [dim](thinking→instant)[/dim]",
        f"[{fb_style}]{fallbacks}[/{fb_style}]",
        "",
    )
    errors = req.get("errors_60s", 0) or 0
    err_style = "red bold" if errors > 0 else "cyan"
    tbl.add_row(
        "[bold]Errors[/bold] [dim](last 60 s)[/dim]",
        f"[{err_style}]{errors}[/{err_style}]",
        "",
    )

    title = Text("REQUESTS", style=state.pulses["request"].style())
    return Panel(tbl, title=title, border_style=state.pulses["request"].style(), padding=(0, 1))
