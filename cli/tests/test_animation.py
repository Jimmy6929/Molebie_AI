"""Tests for the monitor TUI animation primitives.

Covers two correctness fixes:
- Sparkline rendering of a zero-variance series (steady values must read
  as steady, not collapse to the bottom row).
- hbar overflow rendering (>100% must be visually distinct from 100%).
"""

from cli.ui.animation import Sparkline, hbar

# The mid-height glyph used by Sparkline.render when hi == lo.
# _BLOCKS = "▁▂▃▄▅▆▇█" → (8 - 1) // 2 = 3 → ▄.
_MID_BLOCK = "▄"


def test_sparkline_steady_high_renders_mid_row_not_low():
    sp = Sparkline(width=8)
    sp.replace([95.0] * 8)
    out = sp.render()
    # All blocks must be the mid-height glyph — never ▁ (which would
    # imply "low") and never blank.
    assert out == _MID_BLOCK * 8, f"expected steady mid-row, got {out!r}"


def test_sparkline_steady_zero_also_mid_row():
    # Zero variance at zero — same treatment: shows "steady", not collapsed.
    sp = Sparkline(width=6)
    sp.replace([0.0] * 6)
    assert sp.render() == _MID_BLOCK * 6


def test_sparkline_varied_series_spans_glyphs():
    sp = Sparkline(width=8)
    sp.replace([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    out = sp.render()
    # Lowest sample → ▁, highest → █. Regression guard for the new branch.
    assert out[0] == "▁"
    assert out[-1] == "█"


def test_sparkline_empty_buffer_is_blank():
    sp = Sparkline(width=4)
    assert sp.render() == "    "


def test_hbar_overflow_marker_for_above_100():
    out = hbar(150.0, width=8)
    assert out.endswith("▶"), f"expected overflow marker, got {out!r}"
    # Bar itself must be fully filled.
    assert out.startswith("█" * 8)


def test_hbar_exact_100_has_no_overflow_marker():
    out = hbar(100.0, width=8)
    assert out == "█" * 8
    assert "▶" not in out


def test_hbar_partial_fill_unchanged():
    # 50% → half the bar filled, no overflow.
    out = hbar(50.0, width=8)
    assert out == "█" * 4 + "░" * 4


def test_hbar_none_is_all_empty():
    assert hbar(None, width=6) == "░" * 6
