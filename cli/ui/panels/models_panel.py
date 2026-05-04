"""Models & Storage panel — model load state + DB stats at a glance.

Two stacked groups in one panel:
  • Models: embedding, reranker, NLI (selfcheck backend), TTS — load state
    + URL/source. Renders ● green when loaded/up, ○ dim otherwise.
  • Storage: DB file size + row counts (documents/chunks, sessions/messages,
    memories). Read from the StorageProbe snapshot in /metrics/live.

Designed to be the "static state" panel — it changes on the order of
seconds (model load) or minutes (DB growth). The Subsystems panel
handles the per-second motion.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


_MODEL_LABELS = {
    "embedding": "embedding",
    "reranker":  "reranker",
    "nli":       "nli (selfcheck)",
    "tts":       "tts (kokoro)",
}


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n/1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n/(1024**2):.1f} MB"
    return f"{n/(1024**3):.2f} GB"


def _model_dot(loaded: bool | None, status: str | None) -> str:
    if loaded:
        return "[green]●[/green]"
    if status in ("up",):
        return "[green]●[/green]"
    if status in ("down",):
        return "[red]○[/red]"
    return "[dim]○[/dim]"


def render(state, width_hint: int = 60) -> Panel:
    models = state.raw.get("models") or {}
    storage = state.raw.get("storage") or {}

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(no_wrap=True, min_width=2)
    tbl.add_column(no_wrap=True, min_width=12)
    tbl.add_column(overflow="ellipsis")

    # ── Models block ─────────────────────────────────────────────────
    tbl.add_row("", "[bold]Models[/bold]", "")
    if not models:
        tbl.add_row("", "[dim]— none observed yet —[/dim]", "")
    else:
        # Stable order: known names first, then alphabetical.
        order = ["embedding", "reranker", "nli", "tts"]
        names = [n for n in order if n in models] + sorted(n for n in models if n not in order)
        for name in names:
            m = models[name]
            label = _MODEL_LABELS.get(name, name)
            dot = _model_dot(m.get("loaded"), m.get("status"))
            url = m.get("url") or "—"
            status_word = m.get("status") or ("loaded" if m.get("loaded") else "not loaded")
            load_ms = m.get("load_ms")
            extra = f"  [dim]({load_ms:.0f}ms load)[/dim]" if load_ms else ""
            tbl.add_row(dot, label, f"{url} [dim]·[/dim] {status_word}{extra}")

    # ── Storage block ────────────────────────────────────────────────
    tbl.add_row("", "", "")  # spacer
    tbl.add_row("", "[bold]Storage[/bold]", "")
    if not storage:
        tbl.add_row("", "[dim]— probe unavailable —[/dim]", "")
    else:
        size = storage.get("db_size_bytes")
        docs = storage.get("documents") or {}
        sess = storage.get("sessions") or {}
        mems = storage.get("memories") or {}
        note = storage.get("note")

        tbl.add_row("", "DB", f"{_fmt_bytes(size)}")
        d_count = docs.get("count")
        d_chunks = docs.get("chunks")
        if d_count is not None:
            tbl.add_row(
                "", "Docs",
                f"{d_count} [dim]({d_chunks if d_chunks is not None else 0} chunks)[/dim]",
            )
        s_count = sess.get("count")
        s_msgs = sess.get("messages")
        if s_count is not None:
            tbl.add_row(
                "", "Sessions",
                f"{s_count} [dim]({s_msgs if s_msgs is not None else 0} messages)[/dim]",
            )
        m_count = mems.get("count")
        if m_count is not None:
            tbl.add_row("", "Memories", f"{m_count}")
        if note:
            tbl.add_row("", "", f"[yellow]{note}[/yellow]")

    pulse = state.pulses.get("models", state.pulses["system"])
    title = Text("MODELS & STORAGE", style=pulse.style())
    return Panel(tbl, title=title, border_style=pulse.style(), padding=(0, 1))
