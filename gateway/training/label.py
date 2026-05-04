"""
Heuristic relabelling of probe records into training-ready labels.

The probe runner produces a heuristic first-pass label. This module:

  1. Re-runs the heuristic with looser/stricter thresholds (operator
     can re-label without re-probing — probing is the expensive part).
  2. Provides ``rewrite_to_abstention()`` — the canonical transform
     from a fabricated response into the user-facing abstention text.
     Used by ``build_dataset.py`` when constructing SFT pairs.
  3. Produces a CSV-ish summary so the operator can spot-check labels
     before the dataset goes into training.

Design rule: every transformation here is deterministic. The dataset
builder must produce the same training data given the same probe
records — reproducibility for ablation reasons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Canonical abstention message — straight from the build plan, Phase 4
# task 4.1. The R-Tuning literature (NAACL 2024 outstanding paper) says
# the *exact phrasing* matters less than the *category* of behavior, so
# don't paraphrase per record — keep one canonical form for every
# fabrication rewrite. Prediction stability across abstention cases is
# what we're training the LoRA to learn.
CANONICAL_ABSTENTION = (
    "I don't have reliable information about that. If you can share "
    "context — a note, a doc, or a URL — I can ground an answer in it. "
    "Otherwise, I'd rather flag uncertainty than guess."
)


def rewrite_to_abstention(_probe_response: str) -> str:
    """Return the canonical abstention. Input is intentionally unused —
    we never want to "edit" a fabrication into something almost-right.
    Always full replacement to the canonical form."""
    return CANONICAL_ABSTENTION


def relabel(record: dict, *, fabrication_strict: bool = True) -> dict:
    """Re-evaluate a probe record's label under explicit policies.

    ``fabrication_strict``:
      - True (default): any specific fact (number / URL / date) without
        a citation OR refusal phrase is treated as fabrication. This is
        the safer policy for training data — false positives mean an
        innocent response gets relabelled to abstention, which only
        teaches "abstain more often" (style, not facts).
      - False: requires the post-processing layers (CoVe / Judge /
        SelfCheck) to also have flagged the response. Fewer training
        samples but higher confidence each one is a real fabrication.

    Returns a record with an ``effective_label`` field added.
    """
    label = record.get("heuristic_label") or {}
    name = label.get("label", "ambiguous")
    signals = set(label.get("signals") or [])

    if name == "likely_fabrication" and fabrication_strict:
        effective = "fabrication"
    elif name == "likely_fabrication" and "post_layer_flagged" in signals:
        effective = "fabrication"
    elif name == "proper_abstention":
        effective = "abstention_kept"
    elif name == "ground_truth_match":
        effective = "grounded_kept"
    elif name == "cited_response":
        effective = "grounded_kept"
    else:
        effective = "skip"

    return {**record, "effective_label": effective}


def relabel_file(
    inp: Path,
    out: Path,
    *,
    fabrication_strict: bool,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    with inp.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = relabel(json.loads(line), fabrication_strict=fabrication_strict)
            fout.write(json.dumps(rec) + "\n")
            counts[rec["effective_label"]] = counts.get(rec["effective_label"], 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Probe records JSONL (output of probe.py).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Relabelled JSONL.")
    parser.add_argument("--fabrication-strict", action="store_true",
                        default=True,
                        help="Treat all heuristic fabrications as such.")
    parser.add_argument("--require-layer-flag", action="store_true",
                        help="Require CoVe/Judge/SelfCheck to also have "
                             "flagged the response. Stricter, fewer samples.")
    args = parser.parse_args()
    strict = args.fabrication_strict and not args.require_layer_flag
    counts = relabel_file(args.input, args.output, fabrication_strict=strict)
    print(f"[label] wrote {args.output}")
    print(f"[label] effective labels: {counts}")


if __name__ == "__main__":
    main()
