"""molebie-ai monitor — live terminal dashboard.

Adaptive layout chosen per render frame from the current terminal size:

  * **Compact**   ≥ 110×28 — classic 4 panels (Requests / Inference /
    System / Activity), so the v1 experience still works on small terms.
  * **Standard**  ≥ 140×36 — 2×3 grid + bottom activity row. Adds the
    Pipeline, Subsystems, and Models & Storage panels.
  * **Wide**      ≥ 180×40 — 3-wide top row (Requests / Pipeline /
    Inference) over a 3-wide middle row (System / Subsystems / Models).
    Sparklines widen, more events shown.
  * **Ultra**     ≥ 220×48 — same as Wide but Activity row doubles in
    height for ~30 events.

Below 110×28 we render a single placeholder rather than a broken grid.

Design calls baked in:

  * **No keyboard input in v1** — Ctrl-C is the only interaction. Adds
    zero dependency surface and there's no v1 behaviour that would need a key.
  * **Fail-soft** — if the gateway is unreachable at startup or mid-run,
    we show a placeholder and keep polling. The monitor never crashes on
    a down server; it's observability that should outlast what it watches.
  * **Schema-version aware** — if the gateway is older than the CLI
    (no `meta.schema_version` or older value), we silently fall back to
    Compact mode and only render the legacy panels.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import typer
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from cli.ui.animation import PulseTracker, Sparkline, tween
from cli.ui.console import console
from cli.ui.panels import (
    activity_panel,
    in_flight_panel,
    inference_panel,
    models_panel,
    pipeline_panel,
    request_panel,
    subsystems_panel,
    system_panel,
)


_MIN_COLS = 80
_MIN_ROWS = 20
_DEFAULT_GATEWAY = "http://localhost:8000"
_POLL_HZ = 2.0
_RENDER_HZ = 10.0

# Subsystem name prefixes that count as "user-meaningful" — these trigger
# the Subsystems panel pulse. Background activity (probe heartbeats, HTTP
# polls, DB queries) is excluded so the border doesn't flicker on every
# 500 ms poll just because a probe ticked.
_PULSE_PREFIXES = (
    "rag.", "web.", "verify.", "task.",
    "tool.", "consistency.", "tts.", "inference.", "memory.",
)

# CLI knows about this schema version. If the gateway exposes a higher
# value, we still render but show a hint that the CLI is behind.
_CLI_SCHEMA_VERSION = 2


@dataclass(slots=True)
class _Display:
    """Tweened values — what the panels actually draw. These lerp toward
    the server snapshot each render tick so numbers feel fluid."""

    cpu_percent: float | None = None
    ram_percent: float | None = None
    gpu_percent: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    total_p50_ms: float | None = None
    total_p95_ms: float | None = None
    req_per_sec: float | None = None


@dataclass(slots=True)
class MonitorState:
    """Everything the panels read. Mutable; the poll and render loops both
    touch it, but asyncio gives us the invariant that they don't run
    concurrently in the same event loop."""

    gateway_url: str
    raw: dict = field(default_factory=dict)
    last_poll_ok: bool = False
    last_poll_error: str | None = None
    last_poll_at: float = 0.0
    started_at: float = field(default_factory=time.time)
    display: _Display = field(default_factory=_Display)
    sparklines: dict = field(default_factory=lambda: {
        "ttft": Sparkline(width=28),
        "tpot": Sparkline(width=28),
    })
    pulses: dict = field(default_factory=lambda: {
        "request":    PulseTracker(),
        "inference":  PulseTracker(),
        "system":     PulseTracker(),
        "quality":    PulseTracker(),
        # V2 panels — keyed by panel name.
        "pipeline":   PulseTracker(),
        "subsystems": PulseTracker(),
        "models":     PulseTracker(),
        "activity":   PulseTracker(),
    })
    _last_event_count: int = 0
    _last_pipeline_event_ts: float = 0.0
    _last_subsystems_count_total: int = 0

    def apply_snapshot(self, snap: dict) -> None:
        """Fold a fresh server snapshot into state: update raw, refill
        sparklines, set pulse triggers — but do NOT snap the display
        numbers. Those get lerped on each render tick."""
        self.raw = snap
        req = snap.get("requests", {})
        self.sparklines["ttft"].replace(req.get("ttft_series", []) or [])
        self.sparklines["tpot"].replace(req.get("tpot_series", []) or [])

        # Pulse on each new request completion (request + quality + activity).
        events = req.get("recent_events", []) or []
        if len(events) != self._last_event_count and events:
            last = events[-1]
            self.pulses["request"].pulse()
            self.pulses["quality"].pulse()
            self.pulses["activity"].pulse()
            if last.get("fallback"):
                self.pulses["inference"].pulse()
        self._last_event_count = len(events)

        # Pulse pipeline panel on a new stage event.
        pipe_events = ((snap.get("pipeline") or {}).get("events") or [])
        if pipe_events:
            latest_ts = pipe_events[-1].get("ts") or 0
            if latest_ts > self._last_pipeline_event_ts:
                self.pulses["pipeline"].pulse()
                self._last_pipeline_event_ts = latest_ts

        # Pulse subsystems ONLY on user-meaningful calls — RAG, web search,
        # verification, inference, tasks, tools. Background traffic
        # (probe heartbeats, HTTP middleware, DB queries) is excluded; if
        # we counted those, the border would flicker every poll because
        # the host probe alone records 1 call/sec.
        subs = snap.get("subsystems") or {}
        meaningful_total = sum(
            (s.get("count_total") or 0)
            for name, s in subs.items()
            if name.startswith(_PULSE_PREFIXES)
        )
        if meaningful_total > self._last_subsystems_count_total:
            self.pulses["subsystems"].pulse()
            self._last_subsystems_count_total = meaningful_total

    def tick(self) -> None:
        """Advance tweens + decay pulses. Called once per render tick."""
        sys_data = self.raw.get("system", {})
        req = self.raw.get("requests", {})
        d = self.display
        d.cpu_percent = tween(d.cpu_percent, sys_data.get("cpu_percent"))
        d.ram_percent = tween(d.ram_percent, sys_data.get("ram_percent"))
        d.gpu_percent = tween(d.gpu_percent, sys_data.get("gpu_percent"))
        d.ttft_ms = tween(d.ttft_ms, req.get("ttft_p50_ms"))
        d.tpot_ms = tween(d.tpot_ms, req.get("tpot_mean_ms"))
        d.total_p50_ms = tween(d.total_p50_ms, req.get("total_p50_ms"))
        d.total_p95_ms = tween(d.total_p95_ms, req.get("total_p95_ms"))
        d.req_per_sec = tween(d.req_per_sec, req.get("req_per_sec"))
        for p in self.pulses.values():
            p.tick()


async def _poll_loop(client: httpx.AsyncClient, state: MonitorState, stop: asyncio.Event) -> None:
    """Fetch /metrics/live at _POLL_HZ until `stop` is set."""
    url = f"{state.gateway_url.rstrip('/')}/metrics/live"
    interval = 1.0 / _POLL_HZ
    while not stop.is_set():
        try:
            resp = await client.get(url, timeout=2.0)
            if resp.status_code == 200:
                state.apply_snapshot(resp.json())
                state.last_poll_ok = True
                state.last_poll_error = None
            else:
                state.last_poll_ok = False
                state.last_poll_error = f"HTTP {resp.status_code}"
        except httpx.TimeoutException:
            state.last_poll_ok = False
            state.last_poll_error = "timeout"
        except httpx.TransportError as exc:
            state.last_poll_ok = False
            state.last_poll_error = f"{type(exc).__name__}"
        except Exception as exc:    # never let the poll loop die
            state.last_poll_ok = False
            state.last_poll_error = f"{type(exc).__name__}: {exc}"
        state.last_poll_at = time.time()
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def _fmt_uptime(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _gateway_schema_version(state: MonitorState) -> int:
    return int((state.raw.get("meta") or {}).get("schema_version") or 1)


def _header(state: MonitorState, mode_label: str) -> Panel:
    up = _fmt_uptime(time.time() - state.started_at)
    dot = "[green]●[/green]" if state.last_poll_ok else "[red]●[/red]"
    err = "" if state.last_poll_ok else f" [dim]({state.last_poll_error})[/dim]"
    schema_v = _gateway_schema_version(state)
    schema_hint = ""
    if schema_v < _CLI_SCHEMA_VERSION:
        schema_hint = f" [yellow]· gateway schema v{schema_v} (upgrade for full view)[/yellow]"
    elif schema_v > _CLI_SCHEMA_VERSION:
        schema_hint = f" [yellow]· CLI behind gateway (v{_CLI_SCHEMA_VERSION} < v{schema_v})[/yellow]"
    left = f"[bold]Molebie AI[/bold] · [cyan]live monitor[/cyan] · uptime [cyan]{up}[/cyan]"
    right = f"{state.gateway_url} {dot}{err}  [dim]· {mode_label}[/dim]{schema_hint}"
    text = Text.from_markup(f"{left}   {right}")
    return Panel(text, style="blue", padding=(0, 1))


def _footer(state: MonitorState) -> Panel:
    recent = state.raw.get("requests", {}).get("recent_events", []) or []
    if not recent:
        msg = "[dim]no recent requests — send a chat to populate[/dim]"
    else:
        ev = recent[-1]
        ts = datetime.fromtimestamp(ev.get("ts") or time.time()).strftime("%H:%M:%S")
        tier = ev.get("tier", "?")
        tokens = ev.get("completion_tokens") or "—"
        ms = ev.get("total_ms")
        ms_str = f"{ms:.0f} ms" if ms is not None else "—"
        ok = "[green]ok[/green]" if ev.get("ok") else "[red]err[/red]"
        msg = (
            f"[dim]recent:[/dim] {ts}  {tier}  {ok}  "
            f"[dim]·[/dim] {tokens} tok  [dim]·[/dim] {ms_str}"
        )
    hint = "[dim]Ctrl-C to quit · resize terminal for more panels[/dim]"
    return Panel(Text.from_markup(f"{msg}    {hint}"), padding=(0, 1))


def _gateway_down_placeholder(state: MonitorState) -> Layout:
    """Full-screen placeholder when the gateway hasn't yet returned a snapshot."""
    err = state.last_poll_error or "no response yet"
    body = Align.center(
        Text.from_markup(
            f"[warn]— gateway unreachable at {state.gateway_url} —[/warn]\n"
            f"[dim]{err}[/dim]\n\n"
            "[dim]Start the gateway with [bold]molebie-ai run[/bold] in another terminal.[/dim]\n"
            "[dim]This monitor will reconnect automatically.[/dim]"
        ),
        vertical="middle",
    )
    layout = Layout()
    layout.split(
        Layout(_header(state, "waiting"), size=3, name="header"),
        Layout(Panel(body), name="body"),
        Layout(_footer(state), size=3, name="footer"),
    )
    return layout


def _too_small_placeholder() -> Panel:
    return Panel(
        Align.center(
            Text.from_markup(
                f"[warn]Terminal too small[/warn]\n\n"
                f"[dim]Resize to at least {_MIN_COLS} × {_MIN_ROWS}[/dim]"
            ),
            vertical="middle",
        ),
        title="Molebie AI · monitor",
    )


# ── Layout modes ──────────────────────────────────────────────────────────


def _choose_mode(width: int, height: int) -> str:
    """Pick a discrete layout mode from terminal dimensions.

    Discrete modes (rather than continuous resize) give predictable layouts
    and one switch in `_build_layout` rather than per-panel reflow logic.

    Thresholds are conservative — most modern terminals are at least 100×30,
    so the default Standard mode (which shows the new v2 panels) kicks in
    almost immediately.

    The five tiers cover everything from a small SSH session (Tiny) to a
    maxed-out external monitor (Ultra) without ever leaving the operator
    on a "too small" placeholder unless the terminal is truly unusable.
    """
    if width < _MIN_COLS or height < _MIN_ROWS:
        return "too_small"
    if width >= 200 and height >= 44:
        return "ultra"
    if width >= 160 and height >= 36:
        return "wide"
    if width >= 110 and height >= 30:
        return "standard"
    if width >= 90 and height >= 26:
        return "compact"
    return "tiny"


def _has_data(state: MonitorState, key: str) -> bool:
    """Whether a top-level snapshot key actually contains rendered content.

    Used by smart-hide so panels with no data step aside and let elastic
    siblings (Pipeline, Subsystems, Activity) take their space.
    """
    val = state.raw.get(key)
    if val is None:
        return False
    if isinstance(val, (list, dict)):
        return bool(val)
    return True


def _activity_rows(height: int) -> int:
    """Adaptive row budget for the Activity panel.

    Short terminals (≤ 32 rows) get a slim 4-row Activity strip so the
    body panels keep room. Tall terminals get up to 12 rows for ~30
    events visible at once.
    """
    return min(12, max(4, height - 22))


def _has_v2_data(state: MonitorState) -> bool:
    """True if the gateway exposes the v2 fields the new panels need."""
    if _gateway_schema_version(state) < 2:
        return False
    keys = state.raw.keys()
    return "subsystems" in keys or "pipeline" in keys


def _maybe_in_flight(state: MonitorState) -> Layout | None:
    """Build a 3-row in-flight banner Layout if a request is streaming, else None."""
    panel = in_flight_panel.render(state)
    if panel is None:
        return None
    return Layout(panel, size=3, name="in_flight")


def _build_layout_tiny(state: MonitorState) -> Layout:
    """Three stacked panels — Pipeline / Subsystems / Activity — for tiny
    terminals (80×20 ≤ size < 90×26). Drops Inference / System / Models
    so the operator still sees the live timeline + rolling counts on a
    cramped SSH session."""
    layout = Layout()
    inflight = _maybe_in_flight(state)
    height = console.size.height

    rows: list[Layout] = [Layout(_header(state, "tiny"), size=3, name="header")]
    if inflight is not None:
        rows.append(inflight)
    rows.extend([
        Layout(name="body"),
        Layout(_footer(state), size=3, name="footer"),
    ])
    layout.split(*rows)

    # Body splits into three elastic regions; ratios let them share space.
    pipeline = Layout(pipeline_panel.render(state, height_hint=max(4, height // 4)),
                      ratio=1, name="pipeline")
    subs = Layout(subsystems_panel.render(state, width_hint=80, height_hint=max(4, height // 4)),
                  ratio=1, name="subsystems")
    activity = Layout(activity_panel.render(state, height_hint=max(4, height // 4)),
                      ratio=1, name="activity")
    layout["body"].split(pipeline, subs, activity)
    return layout


def _split_with_optional_in_flight(
    layout: Layout, *children: Layout, in_flight: Layout | None
) -> None:
    """Helper: split `layout` into header + (optional in_flight) + children..."""
    parts = [children[0]]
    if in_flight is not None:
        parts.insert(0, in_flight)
    layout.split(*parts, *children[1:])


def _build_layout_compact(state: MonitorState) -> Layout:
    """Minimum viable v2 layout — 2×3 + activity row, even at small terms.

    Trades panel density for full v2 coverage: the operator sees Pipeline,
    Subsystems, Models alongside the legacy panels even on an 90-col
    terminal. Smart-hides Models and Inference if they have no data so
    Pipeline / Subsystems can grow into the freed space.
    """
    layout = Layout()
    inflight = _maybe_in_flight(state)
    height = console.size.height
    act_rows = _activity_rows(height)

    rows: list[Layout] = [Layout(_header(state, "compact"), size=3, name="header")]
    if inflight is not None:
        rows.append(inflight)
    rows.extend([
        Layout(name="body"),
        Layout(activity_panel.render(state, height_hint=act_rows), size=act_rows, name="activity"),
        Layout(_footer(state), size=3, name="footer"),
    ])
    layout.split(*rows)
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))

    # Left column: Requests + Pipeline + System (Pipeline is elastic via ratio).
    left_panels = [
        Layout(request_panel.render(state), name="requests", size=11),
        Layout(pipeline_panel.render(state, height_hint=10), name="pipeline", ratio=1),
        Layout(system_panel.render(state), name="system", size=8),
    ]
    layout["body"]["left"].split(*left_panels)

    # Right column: Inference (if data) + Subsystems (always) + Models (if data).
    right_panels: list[Layout] = []
    if _has_data(state, "backends"):
        right_panels.append(Layout(inference_panel.render(state), name="inference", size=10))
    right_panels.append(
        Layout(subsystems_panel.render(state, width_hint=60, height_hint=10),
               name="subsystems", ratio=1)
    )
    if _has_data(state, "models"):
        right_panels.append(Layout(models_panel.render(state, width_hint=60), name="models", size=11))
    layout["body"]["right"].split(*right_panels)
    return layout


def _build_layout_legacy(state: MonitorState) -> Layout:
    """Strict v1 fallback — only used when the gateway is older than v2."""
    layout = Layout()
    layout.split(
        Layout(_header(state, "legacy v1"), size=3, name="header"),
        Layout(name="body"),
        Layout(_footer(state), size=3, name="footer"),
    )
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))
    layout["body"]["left"].split(
        Layout(request_panel.render(state), name="request"),
        Layout(system_panel.render(state), name="system"),
    )
    layout["body"]["right"].split(
        Layout(inference_panel.render(state), name="inference"),
        Layout(activity_panel.render(state, height_hint=12), name="activity"),
    )
    return layout


def _build_layout_standard(state: MonitorState) -> Layout:
    """2×3 + activity row at the bottom (activity row sized adaptively)."""
    layout = Layout()
    inflight = _maybe_in_flight(state)
    height = console.size.height
    act_rows = _activity_rows(height)

    rows: list[Layout] = [Layout(_header(state, "standard"), size=3, name="header")]
    if inflight is not None:
        rows.append(inflight)
    rows.extend([
        Layout(name="body"),
        Layout(activity_panel.render(state, height_hint=act_rows), size=act_rows, name="activity"),
        Layout(_footer(state), size=3, name="footer"),
    ])
    layout.split(*rows)
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))

    layout["body"]["left"].split(
        Layout(request_panel.render(state), name="requests", size=12),
        Layout(pipeline_panel.render(state, height_hint=12), name="pipeline", ratio=1),
        Layout(system_panel.render(state), name="system", size=9),
    )

    right_panels: list[Layout] = []
    if _has_data(state, "backends"):
        right_panels.append(Layout(inference_panel.render(state), name="inference", size=10))
    right_panels.append(
        Layout(subsystems_panel.render(state, width_hint=80, height_hint=12),
               name="subsystems", ratio=1)
    )
    if _has_data(state, "models"):
        right_panels.append(Layout(models_panel.render(state, width_hint=80), name="models", size=12))
    layout["body"]["right"].split(*right_panels)
    return layout


def _build_layout_wide(state: MonitorState, height_activity: int | None = None) -> Layout:
    """3-wide top + 3-wide middle + activity row.

    When `height_activity` is None, derive it from the actual terminal height
    so wide-and-short / wide-and-tall both look right.
    """
    layout = Layout()
    inflight = _maybe_in_flight(state)
    height = console.size.height
    if height_activity is None:
        height_activity = _activity_rows(height)
    label = "ultra" if height_activity >= 10 else "wide"

    rows: list[Layout] = [Layout(_header(state, label), size=3, name="header")]
    if inflight is not None:
        rows.append(inflight)
    rows.extend([
        Layout(name="top"),
        Layout(name="mid"),
        Layout(activity_panel.render(state, height_hint=height_activity),
               size=height_activity, name="activity"),
        Layout(_footer(state), size=3, name="footer"),
    ])
    layout.split(*rows)

    # Top row: Requests + Pipeline (elastic) + Inference (collapsed if no backends).
    top_panels: list[Layout] = [
        Layout(request_panel.render(state), name="requests", ratio=1),
        Layout(pipeline_panel.render(state, height_hint=14), name="pipeline", ratio=1),
    ]
    if _has_data(state, "backends"):
        top_panels.append(Layout(inference_panel.render(state), name="inference", ratio=1))
    layout["top"].split_row(*top_panels)

    # Mid row: System + Subsystems (elastic) + Models (collapsed if no models).
    mid_panels: list[Layout] = [
        Layout(system_panel.render(state), name="system", ratio=1),
        Layout(subsystems_panel.render(state, width_hint=100, height_hint=14),
               name="subsystems", ratio=1),
    ]
    if _has_data(state, "models"):
        mid_panels.append(Layout(models_panel.render(state, width_hint=100), name="models", ratio=1))
    layout["mid"].split_row(*mid_panels)
    return layout


def _build_layout(state: MonitorState) -> Layout | Panel:
    size = console.size
    mode = _choose_mode(size.width, size.height)

    if mode == "too_small":
        return _too_small_placeholder()

    if not state.raw:
        return _gateway_down_placeholder(state)

    # If the gateway is older than v2, fall back to the legacy 4-panel layout
    # — the new panels would be empty anyway.
    if not _has_v2_data(state):
        return _build_layout_legacy(state)

    if mode == "tiny":
        return _build_layout_tiny(state)
    if mode == "compact":
        return _build_layout_compact(state)
    if mode == "standard":
        return _build_layout_standard(state)
    if mode == "wide":
        return _build_layout_wide(state, height_activity=8)
    if mode == "ultra":
        return _build_layout_wide(state, height_activity=12)
    return _build_layout_compact(state)


async def _run(gateway_url: str) -> None:
    state = MonitorState(gateway_url=gateway_url)
    stop = asyncio.Event()

    async with httpx.AsyncClient() as client:
        poll_task = asyncio.create_task(_poll_loop(client, state, stop))
        try:
            with Live(
                _build_layout(state),
                console=console,
                refresh_per_second=_RENDER_HZ,
                screen=True,
                transient=False,
            ) as live:
                interval = 1.0 / _RENDER_HZ
                while not stop.is_set():
                    state.tick()
                    live.update(_build_layout(state))
                    await asyncio.sleep(interval)
        finally:
            stop.set()
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass


def monitor(
    gateway_url: str = typer.Option(
        _DEFAULT_GATEWAY,
        "--gateway",
        "-g",
        help="Gateway base URL to poll for metrics.",
    ),
) -> None:
    """Live terminal dashboard: requests, pipeline, inference, subsystems,
    system, models & storage, and a recent-activity log. Layout adapts to
    terminal size — resize the window to see more panels."""
    try:
        asyncio.run(_run(gateway_url))
    except KeyboardInterrupt:
        # Rich Live has already restored the screen; print nothing noisy.
        console.print()
