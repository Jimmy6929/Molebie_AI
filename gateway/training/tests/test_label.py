"""Tests for the heuristic label + abstention rewriter."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
EVAL = TRAINING.parent / "tests" / "eval"
for p in (TRAINING, EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import label                      # noqa: E402
from probe import heuristic_label  # noqa: E402


def test_canonical_abstention_is_full_replacement():
    """The rewriter must NOT edit the fabrication. Always full replacement."""
    out = label.rewrite_to_abstention("Mount Everest is 1234 metres tall.")
    assert out == label.CANONICAL_ABSTENTION
    assert "1234" not in out


def test_heuristic_flags_uncited_specific_facts():
    entry = {"category": "fake_paper"}
    text = "The 2024 Marquez et al. paper proves Lemma 4.3 about FlexAttention."
    h = heuristic_label(entry, text, {})
    assert h["label"] == "likely_fabrication"
    assert "no_citation" in h["signals"]


def test_heuristic_passes_proper_abstention():
    entry = {"category": "fake_api"}
    text = "I don't have that in your notes."
    h = heuristic_label(entry, text, {})
    assert h["label"] == "proper_abstention"


def test_heuristic_passes_refusal_phrase():
    entry = {"category": "fake_api"}
    text = "That package doesn't exist as far as I'm aware."
    h = heuristic_label(entry, text, {})
    assert h["label"] == "proper_abstention"


def test_heuristic_passes_cited_response():
    entry = {"category": "rag_grounded"}
    text = "Mount Everest is 8849 metres [S1] above sea level."
    h = heuristic_label(entry, text, {})
    assert h["label"] in ("cited_response", "ground_truth_match")


def test_heuristic_uses_layer_signals_to_raise_confidence():
    """When a Phase-3 layer already flagged the response, the
    fabrication label should be high confidence."""
    entry = {"category": "fake_event"}
    text = "Sam Altman gave a keynote at the May 2026 AI Summit in Seoul."
    rag_metrics = {"verification": {"applied": True, "unsupported_count": 1}}
    h = heuristic_label(entry, text, {"rag_metrics": rag_metrics})
    assert h["label"] == "likely_fabrication"
    assert h["confidence"] == "high"
    assert "post_layer_flagged" in h["signals"]


def test_heuristic_ambiguous_for_chatty_short_answer():
    entry = {"category": "format_test"}
    text = "Hi, how can I help?"
    h = heuristic_label(entry, text, {})
    assert h["label"] == "ambiguous"


# ── relabel() ────────────────────────────────────────────────────────────


def _record(label_name, signals=()):
    return {
        "entry": {"category": "x"},
        "response": {"content": "..."},
        "heuristic_label": {
            "label": label_name, "confidence": "medium", "signals": list(signals),
        },
    }


def test_relabel_strict_promotes_likely_fabrication():
    rec = label.relabel(_record("likely_fabrication"), fabrication_strict=True)
    assert rec["effective_label"] == "fabrication"


def test_relabel_lenient_keeps_only_layer_flagged():
    rec_no_flag = label.relabel(
        _record("likely_fabrication"), fabrication_strict=False,
    )
    rec_flagged = label.relabel(
        _record("likely_fabrication", signals=["post_layer_flagged"]),
        fabrication_strict=False,
    )
    assert rec_no_flag["effective_label"] == "skip"
    assert rec_flagged["effective_label"] == "fabrication"


def test_relabel_keeps_grounded_and_abstention():
    g = label.relabel(_record("ground_truth_match"))
    c = label.relabel(_record("cited_response"))
    a = label.relabel(_record("proper_abstention"))
    assert g["effective_label"] == "grounded_kept"
    assert c["effective_label"] == "grounded_kept"
    assert a["effective_label"] == "abstention_kept"


def test_relabel_skips_ambiguous():
    rec = label.relabel(_record("ambiguous"))
    assert rec["effective_label"] == "skip"
