"""Tests for the answer scaffolding strip (T5).

Fixtures are verbatim answers from the 2026-08-14 blind eval
(tests/fixtures/scaffolded_answers.json): Qwen3.5 opened with a wrong
bold headline, self-corrected mid-answer, and put the right answer in a
terminal block. The strip keeps the terminal block only when BOTH the
correction marker and the terminal heading fire; everything else must
pass through byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.routes.chat import _strip_scaffolding

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "scaffolded_answers.json").read_text()
)


# ── positives: scaffolded answers collapse to their terminal block ─────────


def test_a1_instant_keeps_summary_block():
    out = _strip_scaffolding(FIXTURES["A1_instant"])
    assert out.startswith("**Summary:**")
    # The corrected answers survive; the wrong bold headline does not.
    assert "11 hours 15 minutes" in out
    assert "£54" in out
    assert "180 minutes" not in out
    assert "Wait," not in out


def test_a1_thinking_keeps_final_answer_formulation():
    out = _strip_scaffolding(FIXTURES["A1_thinking"])
    assert out.lower().startswith("**final answer")
    assert "£54" in out
    assert "1 hour 30 minutes" not in out


def test_b3_instant_keeps_final_answer():
    out = _strip_scaffolding(FIXTURES["B3_instant"])
    assert out.lower().startswith("**final answer")
    assert "0" in out
    assert "Tom has **1** brother" not in out


# ── pass-throughs: anything without BOTH signals is untouched ──────────────


@pytest.mark.parametrize("key", ["A3_thinking", "C2_instant", "G4_instant", "E2_instant"])
def test_pass_through_byte_identical(key):
    # A3_thinking self-corrects but has no terminal heading (documented
    # limitation); C2/G4 are clean answers; E2 is a truncated spiral with
    # no terminal block. All must survive untouched.
    assert _strip_scaffolding(FIXTURES[key]) == FIXTURES[key]


def test_summary_without_correction_marker_untouched():
    text = "Here is the plan.\n\n**Summary:**\n- step one\n- step two"
    assert _strip_scaffolding(text) == text


def test_early_heading_untouched():
    # A heading in the first 40% must never nuke the rest of the answer.
    text = "**Summary:** short\n" + ("Wait, let me recalculate. More detail here. " * 30)
    assert _strip_scaffolding(text) == text


def test_empty_terminal_block_untouched():
    text = "Something.\nWait, let me recalculate.\nMore working text here to pad this out.\n**Summary:**"
    assert _strip_scaffolding(text) == text


def test_idempotent():
    once = _strip_scaffolding(FIXTURES["A1_instant"])
    assert _strip_scaffolding(once) == once
