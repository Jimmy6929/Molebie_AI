"""molebie-ai monitor — live terminal dashboard.

Four quadrants (request / inference / system / quality) + header + footer.
Polls the gateway's `/metrics/live` at 2 Hz and renders at 10 Hz; the gap
is what makes numbers tween smoothly rather than snap.

Design calls baked in:

  * **No keyboard input in v1** — Ctrl-C is the only interaction. Adds
    zero dependency surface (no readchar/termios) and there's no v1
    behaviour that would need a key (no pause / zoom / filter).
  * **Fail-soft** — if the gateway is unreachable at startup or mid-run,
    we show a placeholder and keep polling. The monitor never crashes on
    a down server; it's observability that should outlast what it watches.
  * **Min-size guard** — below 120 × 30 we render a single "resize
    terminal" panel rather than a broken layout. Partial renders look
    like bugs, so we refuse them instead.
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
from cli.ui.panels import inference_panel, quality_panel, request_panel, system_panel


_MIN_COLS = 120
_MIN_ROWS = 30
_DEFAULT_GATEWAY = "http://localhost:8000"
_POLL_HZ = 2.0
_RENDER_HZ = 10.0


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
        "request": PulseTracker(),
        "inference": PulseTracker(),
        "system": PulseTracker(),
        "quality": PulseTracker(),
    })
    _last_event_count: int = 0

    def apply_snapshot(self, snap: dict) -> None:
        """Fold a fresh server snapshot into state: update raw, refill
        sparklines, set pulse triggers — but do NOT snap the display
        numbers. Those get lerped on each render tick."""
        self.raw = snap
        req = snap.get("requests", {})
        self.sparklines["ttft"].replace(req.get("ttft_series", []) or [])
        self.sparklines["tpot"].replace(req.get("tpot_series", []) or [])

        # Pulse on each new request completion.
        events = req.get("recent_events", []) or []
        if len(events) != self._last_event_count and events:
            last = events[-1]
            # Pulse request + quality always; pulse inference on fallback;
            # pulse quality red-hot on error (same style, different signal).
            self.pulses["request"].pulse()
            self.pulses["quality"].pulse()
            if last.get("fallback"):
                self.pulses["inference"].pulse()
        self._last_event_count = len(events)

    def tick(self) -> None:
        """Advance tweens + decay pulses. Called once per render tick."""
        sys = self.raw.get("system", {})
        req = self.raw.get("requests", {})
        d = self.display
        d.cpu_percent = tween(d.cpu_percent, sys.get("cpu_percent"))
        d.ram_percent = tween(d.ram_percent, sys.get("ram_percent"))
        d.gpu_percent = tween(d.gpu_percent, sys.get("gpu_percent"))
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


def _header(state: MonitorState) -> Panel:
    up = _fmt_uptime(time.time() - state.started_at)
    dot = "[green]●[/green]" if state.last_poll_ok else "[red]●[/red]"
    err = "" if state.last_poll_ok else f" [dim]({state.last_poll_error})[/dim]"
    left = f"[bold]Molebie AI[/bold] · [cyan]live monitor[/cyan] · uptime [cyan]{up}[/cyan]"
    right = f"{state.gateway_url} {dot}{err}"
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
    hint = "[dim]Ctrl-C to quit[/dim]"
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
        Layout(_header(state), size=3, name="header"),
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


def _build_layout(state: MonitorState) -> Layout | Panel:
    size = console.size
    if size.width < _MIN_COLS or size.height < _MIN_ROWS:
        return _too_small_placeholder()

    if not state.raw:
        return _gateway_down_placeholder(state)

    layout = Layout()
    layout.split(
        Layout(_header(state), size=3, name="header"),
        Layout(name="body"),
        Layout(_footer(state), size=3, name="footer"),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["body"]["left"].split(
        Layout(request_panel.render(state), name="request"),
        Layout(system_panel.render(state), name="system"),
    )
    layout["body"]["right"].split(
        Layout(inference_panel.render(state), name="inference"),
        Layout(quality_panel.render(state), name="quality"),
    )
    return layout


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
    """Live terminal dashboard: request, inference, system, and quality layers."""
    try:
        asyncio.run(_run(gateway_url))
    except KeyboardInterrupt:
        # Rich Live has already restored the screen; print nothing noisy.
        console.print()
