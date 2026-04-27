"""Inference-layer panel: per-tier cards (descriptor + model + active count
+ warm/cold + last-latency). Two-row stanzas give each tier room to breathe
while keeping the panel dense overall."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


_TIER_DESCRIPTOR = {
    "instant": "Instant tier (fast)",
    "thinking": "Thinking tier (deep reasoning)",
    "thinking_harder": "Thinking harder tier (long reasoning)",
}


def _status_dot(status: str) -> tuple[str, str]:
    """Return (glyph, rich-colour) for a backend status."""
    return {
        "up":            ("●", "green"),
        "cold":          ("●", "yellow"),
        "down":          ("○", "red"),
        "not_configured":("○", "dim"),
    }.get(status, ("○", "dim"))


def _status_word(status: str) -> str:
    """Human phrase for the status — sits next to model name."""
    return {
        "up":   "warm",
        "cold": "cold",
        "down": "down",
        "not_configured": "not configured",
    }.get(status, status)


def render(state) -> Panel:
    backends = state.raw.get("backends", []) or []
    tiers = state.raw.get("requests", {}).get("tiers", {}) or {}

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(no_wrap=True)    # dot or indent
    tbl.add_column(overflow="ellipsis")  # text

    if not backends:
        tbl.add_row("", "[dim]— no backends configured —[/dim]")

    # Preferred ordering: instant first, then thinking, then anything else.
    order = {"instant": 0, "thinking": 1, "thinking_harder": 2}
    sorted_backends = sorted(backends, key=lambda b: order.get(b["tier"], 99))

    for i, b in enumerate(sorted_backends):
        if i > 0:
            tbl.add_row("", "")  # one-row gap between tier cards

        glyph, colour = _status_dot(b["status"])
        tier = b["tier"]
        descriptor = _TIER_DESCRIPTOR.get(tier, f"{tier} tier")

        # Row A: status dot + descriptor line
        tbl.add_row(
            f"[{colour}]{glyph}[/{colour}]",
            f"[bold]{descriptor}[/bold]",
        )

        # Row B: model + active count + warm/cold + last latency (compact).
        if b["status"] == "not_configured":
            tbl.add_row("", "[dim]— not configured —[/dim]")
            continue

        model = b.get("model") or "(no model)"
        tier_stats = tiers.get(tier, {})
        active = tier_stats.get("active", 0)
        completed = tier_stats.get("completed", 0)
        active_str = f"{active} active" if active else "idle"
        state_word = _status_word(b["status"])
        latency = b.get("last_latency_ms")
        lat_str = f"({latency:.0f} ms)" if latency is not None else ""
        # Dense right-hand side, dim separators so the eye reads the model first.
        rhs = f"{model} [dim]·[/dim] {active_str} [dim]·[/dim] {state_word}"
        if lat_str:
            rhs += f" [dim]{lat_str}[/dim]"
        tbl.add_row("", rhs)

        # Lifetime counter — "yes, things have happened" proof-of-life that
        # doesn't expire with the rolling window. Only shown when there's
        # been at least one request so idle tiers stay visually quiet.
        if completed > 0:
            errors = tier_stats.get("errors", 0)
            counter = f"[dim]{completed} completed since uptime[/dim]"
            if errors > 0:
                counter += f" [red]· {errors} error{'s' if errors != 1 else ''}[/red]"
            tbl.add_row("", counter)

    title = Text("INFERENCE BACKENDS", style=state.pulses["inference"].style())
    return Panel(tbl, title=title, border_style=state.pulses["inference"].style(), padding=(0, 1))
