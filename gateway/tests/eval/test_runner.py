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


# ── run_pass_rate_eval.assess() — negative-category semantics ──────────────
# Regression tests for the inverted rag_grounded_negative check (2026-08-14):
# the old assess() treated the rows' disclaimer phrasings as FORBIDDEN text,
# failing honest abstention and passing fabrication. run_baseline.evaluate
# is canonical; both runners must agree on these cases.

import json  # noqa: E402

import run_pass_rate_eval as rp  # noqa: E402

_NEGATIVE_ENTRY = {
    "category": "rag_grounded_negative",
    "expected_substrings": ["I don't have", "no notes", "not found"],
    "any_substring": True,
}


def test_negative_passes_on_exact_fallback_string():
    ok, reason = rp.assess(_NEGATIVE_ENTRY, "I don't have that in your notes.")
    assert ok is True


def test_negative_passes_on_row_disclaimer_phrasing():
    ok, reason = rp.assess(
        _NEGATIVE_ENTRY, "That fact was not found in your notes, sorry."
    )
    assert ok is True


def test_negative_fails_on_fabrication():
    fabricated = "Your note records Mount Everest's elevation as 8,849 metres [S1]."
    ok, reason = rp.assess(_NEGATIVE_ENTRY, fabricated)
    assert ok is False
    assert "disclaim" in reason


def test_negative_runners_agree():
    """run_baseline and run_pass_rate_eval must give the same verdict on the
    same negative-category texts (they diverged before the 2026-08-14 fix)."""
    abstain = "I don't have that in your notes."
    fabricated = "The quarterly review deadline in your notes is March 3rd [S1]."
    for text, expected in ((abstain, True), (fabricated, False)):
        assert rp.assess(_NEGATIVE_ENTRY, text)[0] is expected
        assert rb.evaluate(_NEGATIVE_ENTRY, text, [])["pass"] is expected


# ── golden-set + corpus data integrity ─────────────────────────────────────
# The retrieval-regression rows added from the 2026-08-14 eval (G3/G5
# findings) must stay AND-mode + must_cite, and every DEFAULT_CORPUS file
# must exist so ablation.sh seeds what the rows assume.


def _load_golden() -> dict[str, dict]:
    path = HERE / "golden_set.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return {r["id"]: r for r in rows}


def test_ports_regression_row_shape():
    row = _load_golden()["rag_016"]
    assert row["category"] == "rag_grounded"
    assert row["must_cite"] is True
    assert not row.get("any_substring", False)  # AND-mode over all four ports
    assert set(row["expected_substrings"]) == {"3000", "8000", "8080", "8081"}


def test_multihop_regression_row_shape():
    row = _load_golden()["rag_017"]
    assert row["category"] == "rag_grounded"
    assert row["must_cite"] is True
    assert not row.get("any_substring", False)  # AND-mode emulates the 2-doc join
    assert set(row["expected_substrings"]) == {"SelfCheck", "DeBERTa"}


def test_default_corpus_files_exist():
    import seed_corpus

    repo_root = HERE.parent.parent.parent
    missing = [p for p in seed_corpus.DEFAULT_CORPUS if not (repo_root / p).is_file()]
    assert not missing, f"DEFAULT_CORPUS entries missing on disk: {missing}"


def test_ports_corpus_doc_contains_all_expected_ports():
    doc = (HERE / "corpus" / "ports-and-topology.md").read_text()
    for port in ("3000", "8000", "8080", "8081"):
        assert port in doc
