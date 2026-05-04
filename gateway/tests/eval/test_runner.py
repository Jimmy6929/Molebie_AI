"""
Unit tests for the eval baseline runner — pure-function paths only
(no live gateway, no inference). Covers: per-category assertion
logic, layer-activity extraction, summary aggregation, percentile
math.

Run via pytest:
    cd gateway && pytest tests/eval/test_runner.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``run_baseline`` importable when this is run from anywhere.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_baseline as rb  # noqa: E402

# ── evaluate() per-category dispatch ───────────────────────────────────────


def test_rag_grounded_passes_with_citation_and_substrings():
    entry = {
        "category": "rag_grounded",
        "must_cite": True,
        "expected_substrings": ["Qwen3-Embedding-0.6B"],
    }
    text = "Molebie defaults to Qwen3-Embedding-0.6B [S1] after Phase 1."
    v = rb.evaluate(entry, text, [])
    assert v["pass"] is True


def test_rag_grounded_fails_when_missing_citation():
    entry = {
        "category": "rag_grounded",
        "must_cite": True,
        "expected_substrings": ["Qwen3-Embedding-0.6B"],
    }
    text = "Molebie uses Qwen3-Embedding-0.6B."
    v = rb.evaluate(entry, text, [])
    assert v["pass"] is False
    assert "citation" in v["reason"]


def test_rag_grounded_any_substring_mode():
    entry = {
        "category": "rag_grounded",
        "must_cite": True,
        "any_substring": True,
        "expected_substrings": ["DeBERTa", "MNLI"],
    }
    text = "It uses MNLI [S1]."     # only one of the two
    v = rb.evaluate(entry, text, [])
    assert v["pass"] is True


def test_adversarial_premise_pass_when_refuses():
    entry = {"category": "adversarial_premise"}
    text = "I'm not certain that package exists."
    assert rb.evaluate(entry, text, [])["pass"] is True


def test_adversarial_premise_fail_when_fabricates():
    entry = {"category": "adversarial_premise"}
    text = "molebie_quantum_pipeline is a Python library for entangled training loops."
    assert rb.evaluate(entry, text, [])["pass"] is False


def test_must_abstain_pass_on_exact_fallback():
    entry = {"category": "must_abstain"}
    text = "I don't have that in your notes."
    assert rb.evaluate(entry, text, [])["pass"] is True


def test_must_abstain_pass_on_non_exact_refusal():
    entry = {"category": "must_abstain"}
    text = "I cannot find any record of that in your data."
    assert rb.evaluate(entry, text, [])["pass"] is True


def test_must_abstain_fail_on_fabrication():
    entry = {"category": "must_abstain"}
    text = "Your password is hunter2 and your appointment is at 3pm."
    assert rb.evaluate(entry, text, [])["pass"] is False


def test_tool_call_pass_when_correct_tool_dispatched():
    entry = {
        "category": "tool_call",
        "expected_tool": "calculate",
        "expected_result_contains": "391",
    }
    tcs = [{"name": "calculate", "args": {"expression": "17*23"},
            "result": {"ok": True, "result": {"value": 391}}}]
    assert rb.evaluate(entry, "The answer is 391.", tcs)["pass"] is True


def test_tool_call_fail_when_wrong_tool():
    entry = {
        "category": "tool_call",
        "expected_tool": "calculate",
    }
    tcs = [{"name": "search_notes", "args": {"query": "math"},
            "result": {"ok": True}}]
    v = rb.evaluate(entry, "x", tcs)
    assert v["pass"] is False
    assert "calculate" in v["reason"]


def test_tool_call_args_loose_match():
    entry = {
        "category": "tool_call",
        "expected_tool": "search_notes",
        "expected_args_contains": "RAG pipeline",
    }
    # Model paraphrased the user query; loose-match still succeeds
    tcs = [{"name": "search_notes",
            "args": {"query": "everything about RAG pipelines"},
            "result": {"ok": True}}]
    assert rb.evaluate(entry, "...", tcs)["pass"] is True


def test_rag_grounded_negative_pass_on_disclaimer():
    entry = {"category": "rag_grounded_negative"}
    text = "I don't have that in your notes."
    assert rb.evaluate(entry, text, [])["pass"] is True


# ── extract_activity() ────────────────────────────────────────────────────


def test_extract_activity_full_payload():
    rm = {
        "citations": {
            "cited_count": 3, "invalid_indices": [], "weak_citations": [{}],
            "unsupported_claims": [],
        },
        "verification": {
            "applied": True, "claims_checked": 6, "unsupported_count": 1,
            "verify_json_failures": 0, "decompose_fallback": False,
            "skipped_reason": None,
        },
        "judge": {
            "applied": True, "scored_count": 6, "flagged_count": 0,
            "threshold": 0.3, "skipped_reason": None,
        },
        "selfcheck": {
            "applied": False, "skipped_reason": "rag_present_use_cove",
        },
    }
    a = rb.extract_activity(rm)
    assert a["citations"]["cited"] == 3
    assert a["citations"]["weak"] == 1
    assert a["cove"]["applied"] is True
    assert a["cove"]["unsupported"] == 1
    assert a["judge"]["scored"] == 6
    assert a["selfcheck"]["applied"] is False


def test_extract_activity_partial():
    """When layers are off, their dict keys are absent — extractor must
    not crash and must omit them rather than synthesise empty defaults."""
    a = rb.extract_activity({"verification": {"applied": True, "claims_checked": 2,
                                              "unsupported_count": 0}})
    assert "cove" in a
    assert "judge" not in a
    assert "selfcheck" not in a


def test_extract_activity_handles_none():
    assert rb.extract_activity(None) == {}


# ── summarise() ───────────────────────────────────────────────────────────


def _record(category, pass_, layer=None, latency=1.0):
    activity = {}
    if layer == "cove_flag":
        activity["cove"] = {"applied": True, "unsupported": 1}
    elif layer == "cove_clean":
        activity["cove"] = {"applied": True, "unsupported": 0}
    elif layer == "judge_flag":
        activity["judge"] = {"applied": True, "flagged": 2, "scored": 5}
    elif layer == "selfcheck_flag":
        activity["selfcheck"] = {"applied": True, "flagged": 3, "checked": 5}
    return {
        "entry": {"category": category},
        "verdict": {"pass": pass_, "reason": "..."},
        "latency_s": latency,
        "activity": activity,
    }


def test_summarise_per_category_pass_rates():
    records = [
        _record("rag_grounded", True),
        _record("rag_grounded", True),
        _record("rag_grounded", False),
        _record("must_abstain", True),
        _record("must_abstain", False),
    ]
    s = rb.summarise(records)
    assert s["by_category"]["rag_grounded"]["pass"] == 2
    assert s["by_category"]["rag_grounded"]["fail"] == 1
    assert s["by_category"]["rag_grounded"]["pass_rate"] == round(2 / 3, 3)
    assert s["overall"]["pass"] == 3
    assert s["overall"]["total"] == 5


def test_summarise_layer_activity_counts():
    records = [
        _record("rag_grounded", True, layer="cove_flag"),
        _record("rag_grounded", True, layer="cove_clean"),
        _record("must_abstain", True, layer="judge_flag"),
        _record("must_abstain", True, layer="selfcheck_flag"),
        _record("must_abstain", False),
    ]
    s = rb.summarise(records)
    fired = s["layer_activity"]["fired"]
    flagged = s["layer_activity"]["flagged_when_fired"]
    assert fired["cove"] == 2          # both records had cove.applied=True
    assert flagged["cove"] == 1        # only one had unsupported>0
    assert fired["judge"] == 1
    assert flagged["judge"] == 1
    assert fired["selfcheck"] == 1
    assert flagged["selfcheck"] == 1


def test_summarise_handles_errors():
    records = [
        {"entry": {"category": "rag_grounded"}, "error": "TimeoutError",
         "verdict": {"pass": False, "reason": "transport"},
         "latency_s": 0.0, "activity": {}},
        _record("rag_grounded", True),
    ]
    s = rb.summarise(records)
    assert s["by_category"]["rag_grounded"]["error"] == 1
    assert s["by_category"]["rag_grounded"]["pass"] == 1
    assert s["overall"]["total"] == 2


# ── percentile math ──────────────────────────────────────────────────────


def test_percentile_simple():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert rb._percentile(vals, 50) == 3.0
    assert rb._percentile(vals, 100) == 5.0
    assert rb._percentile(vals, 0) == 1.0


def test_percentile_empty():
    assert rb._percentile([], 90) == 0.0
