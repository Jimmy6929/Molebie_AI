"""
Build SFT + ORPO datasets from labelled probe records.

Two outputs:

1. **SFT** — `(prompt, response)` pairs. Trains the model to *prefer*
   the canonical abstention when it would otherwise fabricate, while
   keeping its grounded-response behavior intact for the cases where
   it currently does the right thing. R-Tuning style (NAACL 2024).

2. **ORPO** — `(prompt, chosen, rejected)` triples. Used in stage 2
   after R-Tuning SFT for a sharper preference signal:
     - chosen   = grounded response with citations, OR canonical
                  abstention for the cases the original was fabricated
     - rejected = the original response (the fabrication itself)
   ORPO doesn't need a separate reference model and uses ~½ the VRAM
   of DPO. See Hong et al. 2024.

Replay data:
   The build plan requires ~25-50% reasoning-heavy replay data (from
   FineTome-100k, NVIDIA Open-Math-Reasoning) mixed in to prevent
   catastrophic forgetting of CoT/reasoning. We don't bundle that
   data — it's a one-time user-side download — but we emit a
   `replay_required.txt` next to the dataset that documents the
   exact mix the operator should construct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from label import CANONICAL_ABSTENTION, rewrite_to_abstention


# Build-plan target ratio: 25-50% reasoning replay, max 75% domain.
# We default to 30% replay — a balance between forgetting risk
# (favours more replay) and abstention-signal strength (favours less).
DEFAULT_REPLAY_FRACTION = 0.30


def _prompt_from_record(rec: dict) -> str:
    """Reconstruct the prompt the model saw. We use only the user query
    here, not the full system+history context — the LoRA learns to
    behave correctly given a query, regardless of what system prompt
    is wrapped around it at inference time."""
    return rec["entry"]["query"]


def build_sft_pair(rec: dict) -> dict | None:
    """Convert one labelled record into an SFT pair, or None to skip.

    Policy:
      * `grounded_kept`   → keep the model's response as the target
      * `abstention_kept` → keep the model's abstention
      * `fabrication`     → REPLACE with canonical abstention
      * everything else   → skip (ambiguous, low-confidence)
    """
    label = rec.get("effective_label") or rec.get("heuristic_label", {}).get("label")
    response_text = (rec.get("response") or {}).get("content", "") or ""
    prompt = _prompt_from_record(rec)
    if label == "fabrication":
        return {
            "prompt": prompt,
            "response": rewrite_to_abstention(response_text),
            "source": "fabrication_rewrite",
        }
    if label in ("grounded_kept", "abstention_kept"):
        return {
            "prompt": prompt,
            "response": response_text,
            "source": label,
        }
    return None


def build_orpo_triple(rec: dict) -> dict | None:
    """Convert one labelled record into an ORPO triple, or None to skip.

    We only build triples for `fabrication` records — the model's
    original response IS the rejected sample, and the canonical
    abstention is the chosen sample. For records the model already
    handled correctly, there's no preference to express (no rejected
    sample exists), so we skip them in the ORPO set.
    """
    label = rec.get("effective_label") or rec.get("heuristic_label", {}).get("label")
    if label != "fabrication":
        return None
    response_text = (rec.get("response") or {}).get("content", "") or ""
    if not response_text.strip():
        return None
    return {
        "prompt": _prompt_from_record(rec),
        "chosen": CANONICAL_ABSTENTION,
        "rejected": response_text,
        "source": "fabrication_pair",
    }


def build_datasets(
    labelled: Path,
    out_dir: Path,
    *,
    replay_fraction: float,
    chat_template: bool,
) -> dict[str, int]:
    """Produce sft.jsonl + orpo.jsonl + replay_required.txt in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_records: list[dict] = []
    orpo_records: list[dict] = []

    with labelled.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sft = build_sft_pair(rec)
            if sft is not None:
                if chat_template:
                    sft = {
                        "messages": [
                            {"role": "user", "content": sft["prompt"]},
                            {"role": "assistant", "content": sft["response"]},
                        ],
                        "source": sft["source"],
                    }
                sft_records.append(sft)
            orpo = build_orpo_triple(rec)
            if orpo is not None:
                orpo_records.append(orpo)

    sft_path = out_dir / "sft.jsonl"
    orpo_path = out_dir / "orpo.jsonl"
    with sft_path.open("w") as f:
        for r in sft_records:
            f.write(json.dumps(r) + "\n")
    with orpo_path.open("w") as f:
        for r in orpo_records:
            f.write(json.dumps(r) + "\n")

    # Document the replay mix the operator must construct externally.
    domain_count = len(sft_records)
    if replay_fraction <= 0:
        replay_target = 0
    elif replay_fraction >= 1:
        replay_target = max(domain_count, 1)
    else:
        # Want: replay / (replay + domain) == replay_fraction
        replay_target = int(round(domain_count * replay_fraction / (1 - replay_fraction)))
    replay_msg = (
        f"# Replay data mix required for Phase 4 SFT\n\n"
        f"Domain (this dataset):    {domain_count} examples\n"
        f"Reasoning replay needed:  {replay_target} examples "
        f"({int(replay_fraction * 100)}% of total mix)\n\n"
        f"Source recommendations from the build plan:\n"
        f"  * mlabonne/FineTome-100k          (general SFT, broad coverage)\n"
        f"  * nvidia/OpenMathReasoning       (math + reasoning chains)\n"
        f"  * Mix at least 75% reasoning-heavy content overall — Qwen3.5\n"
        f"    thinking-mode behavior collapses fast on non-reasoning data.\n\n"
        f"Build the final SFT mix as JSON-Lines with the same schema as\n"
        f"sft.jsonl (chat_template={'on' if chat_template else 'off'}).\n"
    )
    (out_dir / "replay_required.txt").write_text(replay_msg)

    return {
        "sft": len(sft_records),
        "orpo": len(orpo_records),
        "replay_target": replay_target,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelled", type=Path, required=True,
                        help="Output of label.py (relabelled JSONL).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replay-fraction", type=float,
                        default=DEFAULT_REPLAY_FRACTION,
                        help="Fraction of total SFT mix that must be replay data.")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Emit raw {prompt, response} instead of the "
                             "chat-template messages format. Use for backends "
                             "that apply the chat template themselves.")
    args = parser.parse_args()
    counts = build_datasets(
        labelled=args.labelled,
        out_dir=args.out_dir,
        replay_fraction=args.replay_fraction,
        chat_template=not args.no_chat_template,
    )
    print(f"[build_dataset] {counts}")


if __name__ == "__main__":
    main()
