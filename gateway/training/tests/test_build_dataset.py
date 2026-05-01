"""Tests for SFT + ORPO dataset construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import build_dataset as bd        # noqa: E402
import label                      # noqa: E402


# ── build_sft_pair ────────────────────────────────────────────────────────


def _record(effective, content="x"):
    return {
        "entry": {"query": "What is X?"},
        "response": {"content": content},
        "effective_label": effective,
    }


def test_sft_pair_for_grounded_keeps_response():
    pair = bd.build_sft_pair(_record("grounded_kept", "X is 42 [S1]."))
    assert pair["prompt"] == "What is X?"
    assert pair["response"] == "X is 42 [S1]."
    assert pair["source"] == "grounded_kept"


def test_sft_pair_for_abstention_keeps_response():
    pair = bd.build_sft_pair(
        _record("abstention_kept", "I don't have that in your notes."),
    )
    assert pair["response"] == "I don't have that in your notes."


def test_sft_pair_for_fabrication_replaces_with_canonical_abstention():
    fab_text = "X is 1234 according to the 2024 Smith paper."
    pair = bd.build_sft_pair(_record("fabrication", fab_text))
    assert pair["response"] == label.CANONICAL_ABSTENTION
    assert "1234" not in pair["response"]
    assert pair["source"] == "fabrication_rewrite"


def test_sft_pair_skips_ambiguous():
    assert bd.build_sft_pair(_record("skip")) is None


# ── build_orpo_triple ─────────────────────────────────────────────────────


def test_orpo_triple_only_for_fabrication():
    fab_text = "Hinton won the 2026 Turing Award for transformer surgery."
    t = bd.build_orpo_triple(_record("fabrication", fab_text))
    assert t["prompt"] == "What is X?"
    assert t["chosen"] == label.CANONICAL_ABSTENTION
    assert t["rejected"] == fab_text


def test_orpo_triple_skips_grounded():
    assert bd.build_orpo_triple(_record("grounded_kept")) is None
    assert bd.build_orpo_triple(_record("abstention_kept")) is None


def test_orpo_triple_skips_empty_response():
    assert bd.build_orpo_triple(_record("fabrication", "")) is None
    assert bd.build_orpo_triple(_record("fabrication", "   ")) is None


# ── end-to-end build_datasets ─────────────────────────────────────────────


def test_build_datasets_writes_three_files(tmp_path):
    labelled = tmp_path / "labelled.jsonl"
    out = tmp_path / "out"
    records = [
        {**_record("fabrication", "Made-up fact."), "_src": "fab1"},
        {**_record("fabrication", "Another invention."), "_src": "fab2"},
        {**_record("grounded_kept", "Real fact [S1]."), "_src": "good1"},
        {**_record("skip"), "_src": "skip1"},
    ]
    with labelled.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    counts = bd.build_datasets(
        labelled=labelled, out_dir=out,
        replay_fraction=0.30, chat_template=False,
    )
    assert counts["sft"] == 3        # 2 fabrication + 1 grounded
    assert counts["orpo"] == 2       # only fabrications make ORPO triples
    assert (out / "sft.jsonl").exists()
    assert (out / "orpo.jsonl").exists()
    assert (out / "replay_required.txt").exists()

    # Replay target should be sized so replay/(replay+domain) = 0.30
    # With domain=3, target ≈ round(3 * 0.30 / 0.70) = round(1.286) = 1
    assert counts["replay_target"] == 1


def test_build_datasets_chat_template_format(tmp_path):
    labelled = tmp_path / "labelled.jsonl"
    out = tmp_path / "out"
    rec = _record("grounded_kept", "Answer [S1].")
    labelled.write_text(json.dumps(rec) + "\n")
    bd.build_datasets(
        labelled=labelled, out_dir=out,
        replay_fraction=0.0, chat_template=True,
    )
    sft = json.loads((out / "sft.jsonl").read_text().strip())
    assert sft["messages"][0]["role"] == "user"
    assert sft["messages"][1]["role"] == "assistant"
    assert sft["messages"][1]["content"] == "Answer [S1]."


def test_replay_target_zero_when_fraction_zero(tmp_path):
    labelled = tmp_path / "labelled.jsonl"
    out = tmp_path / "out"
    labelled.write_text(json.dumps(_record("grounded_kept")) + "\n")
    counts = bd.build_datasets(
        labelled=labelled, out_dir=out,
        replay_fraction=0.0, chat_template=False,
    )
    assert counts["replay_target"] == 0
