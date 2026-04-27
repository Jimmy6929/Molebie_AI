"""System-layer panel: CPU / RAM / GPU utilisation with colour-threshold
bars (cyan at rest → yellow warm → red hot), plus temperature, power, and
core count. Labels use the full name with a tiny acronym suffix so a new
operator doesn't have to know the jargon."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.ui.animation import colored_bar


def _pct(v: float | None) -> str:
    return f"{v:>5.1f}%" if v is not None else "   —  "


def render(state) -> Panel:
    s = state.raw.get("system", {}) or {}
    d = state.display

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(no_wrap=True)            # label
    tbl.add_column(no_wrap=True)            # bar
    tbl.add_column(justify="right", no_wrap=True)  # percent
    tbl.add_column(no_wrap=True)            # detail (right of percent)

    # ── Processor ─────────────────────────────────────────────
    cores = s.get("cpu_cores")
    cores_phys = s.get("cpu_cores_physical")
    if cores and cores_phys and cores != cores_phys:
        core_str = f"{cores_phys}/{cores} cores"   # physical/logical
    elif cores:
        core_str = f"{cores} cores"
    else:
        core_str = ""
    tbl.add_row(
        "[bold]Processor[/bold] [dim](CPU)[/dim]",
        colored_bar(d.cpu_percent, width=12),
        _pct(d.cpu_percent),
        f"[dim]{core_str}[/dim]",
    )

    # ── Memory ─────────────────────────────────────────────────
    ram_used = s.get("ram_used_gb")
    ram_total = s.get("ram_total_gb")
    ram_txt = f"{ram_used:.1f}/{ram_total:.0f} GB" if ram_used and ram_total else ""
    tbl.add_row(
        "[bold]Memory[/bold] [dim](RAM)[/dim]",
        colored_bar(d.ram_percent, width=12),
        _pct(d.ram_percent),
        f"[dim]{ram_txt}[/dim]",
    )

    # ── Graphics ───────────────────────────────────────────────
    gpu_pct = d.gpu_percent
    gpu_temp = s.get("gpu_temp_c")
    temp_str = f"{gpu_temp:.0f} °C" if gpu_temp is not None else ""
    tbl.add_row(
        "[bold]Graphics[/bold] [dim](GPU)[/dim]",
        colored_bar(gpu_pct, width=12),
        _pct(gpu_pct),
        f"[dim]{temp_str}[/dim]",
    )

    # ── Power ──────────────────────────────────────────────────
    pw = s.get("power_w")
    if pw is not None:
        pw_txt = f"{pw:.1f} W"
        suffix = "[dim](CPU+GPU)[/dim]"  # macmon reports the composite
    else:
        pw_txt = "—"
        suffix = ""
    tbl.add_row(
        "[bold]Power[/bold]",
        "",
        pw_txt,
        suffix,
    )

    # Hint for missing GPU probe, only when present.
    if s.get("note"):
        tbl.add_row("", f"[dim]{s['note']}[/dim]", "", "")

    title = Text("HOST SYSTEM", style=state.pulses["system"].style())
    return Panel(tbl, title=title, border_style=state.pulses["system"].style(), padding=(0, 1))
