"""Tests for the linear-interpolation _quantiles helper.

The previous floor-based nearest-rank implementation collapsed p95 to
max(values) for any n ≤ 20 — so the dashboard's "p95 latency" card
silently tracked the slowest request. Linear interpolation matches
numpy's default `linear` method (Excel PERCENTILE.INC) and gives
meaningful separation even for tiny windows.
"""

import math

from app.services.metrics_registry import _quantiles


def test_quantiles_empty():
    assert _quantiles([]) == (None, None, None)


def test_quantiles_single_value():
    assert _quantiles([42.0]) == (42.0, 42.0, 42.0)


def test_quantiles_small_n_does_not_collapse_to_max():
    # Five samples evenly spaced. With floor-based nearest-rank, p95
    # and p99 both returned 50 (the max). Linear interp must separate
    # them from the max.
    p50, p95, p99 = _quantiles([10.0, 20.0, 30.0, 40.0, 50.0])
    assert math.isclose(p50, 30.0, abs_tol=1e-6)
    # pos = 0.95 * 4 = 3.8 → 40 + 0.8*(50-40) = 48
    assert math.isclose(p95, 48.0, abs_tol=1e-6)
    # pos = 0.99 * 4 = 3.96 → 40 + 0.96*(50-40) = 49.6
    assert math.isclose(p99, 49.6, abs_tol=1e-6)
    assert p95 < 50.0 and p99 < 50.0


def test_quantiles_resists_outliers_under_ties():
    # 100 samples at 5, one outlier at 500. The bulk of the
    # distribution sits at 5, so p95 must stay there — not jump to 500
    # because of one slow request.
    values = [5.0] * 100 + [500.0]
    _, p95, p99 = _quantiles(values)
    assert math.isclose(p95, 5.0, abs_tol=0.01), f"p95 swung to {p95}"
    # p99 sits in the interpolated tail but must not be the outlier.
    assert p99 < 500.0


def test_quantiles_p50_is_median():
    # Even-length list: linear interp gives the average of the two
    # middle samples.
    p50, _, _ = _quantiles([1.0, 2.0, 3.0, 4.0])
    # pos = 0.5 * 3 = 1.5 → 2 + 0.5*(3-2) = 2.5
    assert math.isclose(p50, 2.5, abs_tol=1e-6)
