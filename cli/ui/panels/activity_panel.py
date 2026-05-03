"""Recent-activity panel — chronological log spanning chat completions,
pipeline stage failures, and background-task events.

Replaces the older quality_panel which only showed chat-completion rows.
The wider source set means an idle dashboard still has signal — e.g. a
summary task completing or a verifier flagging unsupported claims will
appear here even if no new chat completions landed.

Adaptive height: caller passes `height_hint`; we show the most recent
`max(3, height_hint - 3)` events.
"""

from __future__ import annotations

from datetime import datetime

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _glyph(status: str | bool) -> str:
    if status is True or status == "ok":
        return "[green]✓[/green]"
    if status is False or status == "fail":
        return "[red]✗[/red]"
    if status == "running":
        return "[cyan]▶[/cyan]"
    if status == "queued":
        return "[dim]·[/dim]"
    return "[dim]?[/dim]"


def _gather_events(state) -> list[dict]:
    """Merge chat completions + pipeline failures + task completions into
    a single chronological list (newest last). Each event has ts, kind,
    label, status, note."""
    out: list[dict] = []
    req = state.raw.get("requests") or {}
    for ev in req.get("recent_events") or []:
        ts = ev.get("ts") or 0
        ok = ev.get("ok", True)
        tier = ev.get("tier") or "?"
        tokens = ev.get("completion_tokens")
        ms = ev.get("total_ms")
        note_parts = []
        if tokens:
            note_parts.append(f"{tokens} tok")
        if ms is not None:
            note_parts.append(f"{ms:.0f} ms")
        if ev.get("fallback"):
            note_parts.append("[yellow]fallback→instant[/yellow]")
        out.append({
            "ts": ts,
            "kind": "chat",
            "label": tier,
            "status": ok,
            "note": "  ".join(note_parts) if note_parts else "",
        })

    # Surface failed pipeline stages — the chat completion already covers
    # successes, but a single failed stage often happens without flipping
    # the whole request to fail (e.g. selfcheck timed out but inference
    # succeeded), so the operator wants visibility.
    pipeline = state.raw.get("pipeline") or {}
    for ev in pipeline.get("events") or []:
        if ev.get("status") not in ("fail",):
            continue
        out.append({
            "ts": ev.get("ts") or 0,
            "kind": "pipeline",
            "label": ev.get("stage", "?"),
            "status": "fail",
            "note": ev.get("note") or "",
        })

    # Background-task signals: surface only the *changes* — last-at + counters.
    tasks = state.raw.get("tasks") or {}
    for name, t in tasks.items():
        if t.get("last_at"):
            ok = t.get("failed_60s", 0) == 0
            comp60 = t.get("completed_60s", 0)
            fail60 = t.get("failed_60s", 0)
            note = []
            if comp60:
                note.append(f"{comp60} done/60s")
            if fail60:
                note.append(f"[red]{fail60} fail/60s[/red]")
            out.append({
                "ts": t["last_at"],
                "kind": "task",
                "label": name,
                "status": ok,
                "note": "  ".join(note),
            })

    out.sort(key=lambda e: e["ts"])
    return out


def render(state, height_hint: int = 8) -> Panel:
    events = _gather_events(state)

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(no_wrap=True)                                  # glyph
    tbl.add_column(no_wrap=True, min_width=8)                     # time
    tbl.add_column(no_wrap=True, overflow="ellipsis", min_width=8)# kind
    tbl.add_column(no_wrap=True, overflow="ellipsis", min_width=14)# label
    tbl.add_column(overflow="ellipsis")                           # note

    pulse = state.pulses.get("activity", state.pulses.get("quality"))

    if not events:
        tbl.add_row("", "[dim]— no activity yet —[/dim]", "", "", "")
        title = Text("RECENT ACTIVITY", style=pulse.style())
        return Panel(tbl, title=title, border_style=pulse.style(), padding=(0, 1))

    max_rows = max(3, height_hint - 3)
    visible = events[-max_rows:]
    for ev in reversed(visible):  # newest at top
        glyph = _glyph(ev["status"])
        ts = datetime.fromtimestamp(ev["ts"]).strftime("%H:%M:%S")
        kind = ev["kind"]
        label = ev["label"]
        note = ev["note"]
        tbl.add_row(
            glyph,
            f"[dim]{ts}[/dim]",
            f"[dim]{kind}[/dim]",
            label,
            note,
        )

    title = Text("RECENT ACTIVITY", style=pulse.style())
    return Panel(tbl, title=title, border_style=pulse.style(), padding=(0, 1))
