"""Small animation primitives for the terminal monitor.

These are deliberately dependency-free — Rich handles layout and colour;
animation is just (a) interpolating displayed numbers toward their target
to hide jitter and (b) turning a value series into a unicode sparkline.

Why tween at all: the server polls at 2 Hz but we render at 10 Hz. Without
easing, numbers snap every 500 ms, which reads as strobing. A 3-tick
(~300 ms) lerp makes the display feel alive without lying about the data.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# Unicode eighths blocks, low → high.
_BLOCKS = "▁▂▃▄▅▆▇█"


def tween(current: float | None, target: float | None, progress: float = 0.3) -> float | None:
    """Move `current` a fraction of the way toward `target`.

    `progress` ∈ (0, 1]. At 0.3 the number arrives in ~3 render ticks —
    fast enough to feel responsive, slow enough to smooth out jitter.
    Returns None if target is None (preserves "unavailable" rendering).
    """
    if target is None:
        return None
    if current is None:
        return target
    return current + (target - current) * progress


@dataclass(slots=True)
class Sparkline:
    """Rolling buffer of recent samples rendered as a unicode bar chart."""

    width: int = 28
    _buf: deque[float] = field(default_factory=lambda: deque(maxlen=28))

    def __post_init__(self) -> None:
        # Resize the deque if the user passed a non-default width.
        if self._buf.maxlen != self.width:
            self._buf = deque(self._buf, maxlen=self.width)

    def push(self, value: float | None) -> None:
        if value is None:
            return
        self._buf.append(float(value))

    def replace(self, values: list[float]) -> None:
        """Replace the buffer with a fresh series (used when the server
        returns the full recent window on each poll)."""
        self._buf.clear()
        for v in values[-self.width:]:
            self._buf.append(float(v))

    def render(self) -> str:
        if not self._buf:
            return " " * self.width
        lo = min(self._buf)
        hi = max(self._buf)
        # Zero variance: a steady high value would otherwise collapse to ▁
        # and read as "low". Render a mid-height row so the operator sees
        # "data exists, no movement" without implying magnitude.
        if hi == lo:
            mid = _BLOCKS[(len(_BLOCKS) - 1) // 2]  # ▄ for an 8-block scale
            return (mid * len(self._buf)).rjust(self.width)
        span = hi - lo
        blocks = []
        for v in self._buf:
            norm = (v - lo) / span  # 0..1
            idx = min(len(_BLOCKS) - 1, int(norm * (len(_BLOCKS) - 1)))
            blocks.append(_BLOCKS[idx])
        # Right-align so the most recent sample sits at the right edge,
        # which is what operators expect (time flows left → right).
        return "".join(blocks).rjust(self.width)


@dataclass(slots=True)
class PulseTracker:
    """Fade-style highlight: when `pulse()` is called, the style returned by
    `style()` stays hot for `duration_ticks` frames, then decays back to
    `base_style`. Used on panel titles when a new request lands."""

    base_style: str = "dim"
    hot_style: str = "cyan"
    duration_ticks: int = 5
    _remaining: int = 0

    def pulse(self) -> None:
        self._remaining = self.duration_ticks

    def tick(self) -> None:
        if self._remaining > 0:
            self._remaining -= 1

    def style(self) -> str:
        return self.hot_style if self._remaining > 0 else self.base_style


def hbar(percent: float | None, width: int = 8, filled: str = "█", empty: str = "░") -> str:
    """Horizontal unicode bar. `percent` is 0..100 (or None → all empty).

    Values > 100 render as a fully-filled bar plus a trailing ▶ overflow
    marker so over-subscription is visually distinct from a saturated 100%
    bar. The marker spills outside `width`; callers that align bars in a
    grid should reserve an extra cell if they care about strict columns.
    """
    if percent is None:
        return empty * width
    if percent > 100.0:
        return filled * width + "▶"
    pct = max(0.0, percent)
    fill = int(round((pct / 100.0) * width))
    return filled * fill + empty * (width - fill)


def colored_bar(
    percent: float | None,
    width: int = 12,
    thresholds: tuple[float, float] = (50.0, 80.0),
) -> str:
    """Horizontal bar wrapped in a Rich markup span whose colour depends
    on the value — cyan at rest, yellow getting warm, red under load.
    Mirrors btop's convention so the colour itself carries meaning and
    the operator can read the dashboard at a glance.

    Returns Rich markup (not raw ANSI); the caller must render through
    Rich for the styling to apply.
    """
    bar = hbar(percent, width=width)
    if percent is None:
        return f"[dim]{bar}[/dim]"
    warn, crit = thresholds
    if percent >= crit:
        colour = "red"
    elif percent >= warn:
        colour = "yellow"
    else:
        colour = "cyan"
    return f"[{colour}]{bar}[/{colour}]"
