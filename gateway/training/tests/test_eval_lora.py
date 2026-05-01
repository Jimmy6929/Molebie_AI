"""Tests for the LoRA before/after evaluation gates."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import eval_lora as el            # noqa: E402


def _report(label, by_cat, mean_latency=5.0):
    return {
        "label": label,
        "summary": {
            "by_category": by_cat,
            "overall": {
                "pass": sum(b.get("pass", 0) for b in by_cat.values()),
                "total": sum(b.get("total", 0) for b in by_cat.values()),
                "pass_rate": (
                    sum(b.get("pass", 0) for b in by_cat.values())
                    / max(1, sum(b.get("total", 0) for b in by_cat.values()))
                ),
            },
            "latency_seconds": {"mean": mean_latency},
        },
    }


def _cat(pass_, total):
    return {"pass": pass_, "fail": total - pass_, "error": 0,
            "total": total, "pass_rate": round(pass_ / total, 3) if total else 0.0}


# ── diff_reports ──────────────────────────────────────────────────────────


def test_diff_per_category_deltas():
    before = _report("base", {
        "rag_grounded":        _cat(12, 15),
        "must_abstain":         _cat(7, 10),
    }, mean_latency=4.0)
    after = _report("post", {
        "rag_grounded":        _cat(13, 15),
        "must_abstain":         _cat(8, 10),
    }, mean_latency=4.5)
    d = el.diff_reports(before, after)
    assert d["by_category"]["rag_grounded"]["delta"] > 0
    assert d["by_category"]["must_abstain"]["delta"] > 0
    assert d["overall_delta"] > 0
    assert d["latency_delta_s"] == 0.5


def test_diff_handles_missing_category():
    before = _report("base", {"rag_grounded": _cat(5, 10)})
    after = _report("post", {"must_abstain": _cat(8, 10)})
    d = el.diff_reports(before, after)
    assert d["by_category"]["rag_grounded"]["delta"] is None
    assert d["by_category"]["must_abstain"]["delta"] is None


# ── apply_gates ───────────────────────────────────────────────────────────


GATES = {
    "faithfulness_min": 0.85,
    "refusal_rate_min": 0.70,
}


def test_gates_keep_when_metrics_clear():
    before = _report("base", {
        "rag_grounded": _cat(13, 15),
        "must_abstain":  _cat(8, 10),
    })
    after = _report("post", {
        "rag_grounded": _cat(14, 15),         # 0.933, ≥ 0.85
        "must_abstain":  _cat(9, 10),         # 0.900, ≥ 0.70
    })
    diff = el.diff_reports(before, after)
    v = el.apply_gates(diff, GATES)
    assert v["all_gates_pass"] is True
    assert "KEEP" in v["verdict_summary"]


def test_gates_discard_on_faithfulness_drop():
    before = _report("base", {"rag_grounded": _cat(14, 15)})
    after = _report("post", {"rag_grounded": _cat(11, 15)})    # 0.733 < 0.85
    diff = el.diff_reports(before, after)
    v = el.apply_gates(diff, GATES)
    assert v["all_gates_pass"] is False
    assert v["verdicts"]["faithfulness"]["ok"] is False


def test_gates_discard_on_2pp_regression():
    """Per build plan task 4.4: any category dropping > 2pp triggers discard."""
    before = _report("base", {
        "rag_grounded": _cat(15, 15),         # 1.0
        "must_abstain":  _cat(10, 10),        # 1.0
    })
    after = _report("post", {
        "rag_grounded": _cat(14, 15),         # 0.933 → -0.067 delta (worse than 2pp)
        "must_abstain":  _cat(10, 10),        # unchanged
    })
    diff = el.diff_reports(before, after)
    v = el.apply_gates(diff, GATES)
    assert v["all_gates_pass"] is False
    assert v["verdicts"]["regression__rag_grounded"]["ok"] is False


def test_gates_pass_within_2pp_jitter():
    """A 1pp regression is within tolerance — don't discard for noise."""
    before = _report("base", {"rag_grounded": _cat(100, 100), "must_abstain": _cat(100, 100)})
    after = _report("post", {"rag_grounded": _cat(99, 100), "must_abstain": _cat(100, 100)})
    diff = el.diff_reports(before, after)
    v = el.apply_gates(diff, GATES)
    assert v["all_gates_pass"] is True


def test_gates_overall_regression_check():
    before = _report("base", {"rag_grounded": _cat(15, 15), "must_abstain": _cat(10, 10)})
    after = _report("post", {"rag_grounded": _cat(13, 15), "must_abstain": _cat(8, 10)})
    diff = el.diff_reports(before, after)
    v = el.apply_gates(diff, GATES)
    # overall_delta = 0.84 - 1.0 = -0.16, well below -0.02
    assert v["verdicts"]["overall_regression"]["ok"] is False
