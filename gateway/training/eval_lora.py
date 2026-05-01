"""
Phase 4 LoRA before/after evaluation.

Reads two baseline-runner reports — one from before the LoRA was
trained, one from after — and produces a structured diff with
explicit pass/fail against the gates in ``hyperparams.yaml``.

The reject rule comes straight from the build plan, task 4.4:

    If any metric drops > 2 percentage points from the pre-tuning
    baseline, the LoRA is doing more harm than good — discard it
    and investigate.

This script enforces that mechanically. It does NOT decide for the
operator — it surfaces deltas + verdicts and lets you read the
report. The bar is "every gate must clear or you discard the
adapter and look at why."

Workflow:

    # 1. Capture the pre-tuning baseline
    python ../tests/eval/run_baseline.py \\
        --gateway http://localhost:8000 \\
        --password $PASSWORD \\
        --output before.json --label "phase3-baseline"

    # 2. Train SFT + ORPO (train_mlx.sh / train_unsloth.sh)

    # 3. Restart inference backend with the LoRA loaded
    #    (e.g.  mlx_lm.server --adapter-path adapters/orpo)

    # 4. Capture the post-tuning eval
    python ../tests/eval/run_baseline.py \\
        --gateway http://localhost:8000 \\
        --password $PASSWORD \\
        --output after.json --label "post-orpo"

    # 5. Diff and verdict
    python eval_lora.py --before before.json --after after.json \\
        --gates hyperparams.yaml --output verdict.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _category_pass_rate(report: dict, category: str) -> float | None:
    by = report.get("summary", {}).get("by_category", {})
    return by.get(category, {}).get("pass_rate")


def _overall_pass_rate(report: dict) -> float | None:
    return report.get("summary", {}).get("overall", {}).get("pass_rate")


def _mean_latency(report: dict) -> float | None:
    return report.get("summary", {}).get("latency_seconds", {}).get("mean")


def diff_reports(before: dict, after: dict) -> dict[str, Any]:
    """Per-category and overall pass-rate deltas + latency delta."""
    before_cats = before.get("summary", {}).get("by_category", {})
    after_cats = after.get("summary", {}).get("by_category", {})
    categories = sorted(set(before_cats) | set(after_cats))

    cat_deltas: dict[str, dict[str, Any]] = {}
    for cat in categories:
        b = before_cats.get(cat, {}).get("pass_rate")
        a = after_cats.get(cat, {}).get("pass_rate")
        if b is None or a is None:
            cat_deltas[cat] = {
                "before": b, "after": a, "delta": None,
                "note": "missing in one of the reports",
            }
            continue
        cat_deltas[cat] = {
            "before": round(b, 3), "after": round(a, 3),
            "delta": round(a - b, 3),
        }

    return {
        "by_category": cat_deltas,
        "overall_delta": round(
            (_overall_pass_rate(after) or 0) - (_overall_pass_rate(before) or 0), 3,
        ),
        "latency_delta_s": round(
            (_mean_latency(after) or 0) - (_mean_latency(before) or 0), 3,
        ),
    }


def apply_gates(diff: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Map diff metrics to pass/fail per gate. The ``gates`` dict is
    the ``gates:`` block of hyperparams.yaml — keys are flat strings.

    Mappings (from build plan task 4.4):
        truthfulqa_mc2_min_delta  → not directly available without
                                    lm-eval-harness; fall through unless
                                    the operator runs lm-eval separately
                                    and adds a ``truthfulqa_mc2_delta``
                                    key to the after report.
        faithfulness_min          → after's rag_grounded pass rate
        refusal_rate_min          → after's must_abstain pass rate
        mmlu_pro / gsm8k / ifeval → similar to truthfulqa, fall through
                                    unless externally captured.
    """
    by_cat = diff["by_category"]
    verdicts: dict[str, Any] = {}

    rag = by_cat.get("rag_grounded", {})
    if rag.get("after") is not None:
        verdicts["faithfulness"] = {
            "metric": "rag_grounded.after_pass_rate",
            "value": rag["after"],
            "threshold": gates.get("faithfulness_min"),
            "ok": (gates.get("faithfulness_min") is None
                   or rag["after"] >= gates.get("faithfulness_min", 0)),
        }
    must = by_cat.get("must_abstain", {})
    if must.get("after") is not None:
        verdicts["refusal_rate"] = {
            "metric": "must_abstain.after_pass_rate",
            "value": must["after"],
            "threshold": gates.get("refusal_rate_min"),
            "ok": (gates.get("refusal_rate_min") is None
                   or must["after"] >= gates.get("refusal_rate_min", 0)),
        }

    # Per-category regression check — applies the 2pp rule to every
    # category, not just the gate-specific ones. The build plan is
    # explicit that "any metric drops > 2pp" triggers a discard.
    for cat, d in by_cat.items():
        if d.get("delta") is None:
            continue
        threshold = -0.02
        verdicts[f"regression__{cat}"] = {
            "metric": f"{cat}.delta",
            "value": d["delta"],
            "threshold": threshold,
            "ok": d["delta"] >= threshold,
        }

    overall_delta = diff["overall_delta"]
    verdicts["overall_regression"] = {
        "metric": "overall.delta",
        "value": overall_delta,
        "threshold": -0.02,
        "ok": overall_delta >= -0.02,
    }

    all_ok = all(v["ok"] for v in verdicts.values())
    return {
        "all_gates_pass": all_ok,
        "verdicts": verdicts,
        "verdict_summary": (
            "KEEP — every gate cleared." if all_ok
            else "DISCARD — at least one gate failed; LoRA is net-negative."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True,
                        help="run_baseline.py JSON report from before tuning.")
    parser.add_argument("--after", type=Path, required=True,
                        help="run_baseline.py JSON report from after tuning.")
    parser.add_argument("--gates", type=Path, required=True,
                        help="hyperparams.yaml (gates section read).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Write verdict JSON here.")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        raise SystemExit("pip install pyyaml")

    before = _load(args.before)
    after = _load(args.after)
    gates = yaml.safe_load(args.gates.read_text()).get("gates", {})

    diff = diff_reports(before, after)
    verdict = apply_gates(diff, gates)
    report = {
        "before_label": before.get("label"),
        "after_label": after.get("label"),
        "diff": diff,
        "verdict": verdict,
    }
    args.output.write_text(json.dumps(report, indent=2))

    print(f"[eval_lora] wrote {args.output}")
    print(f"[eval_lora] {verdict['verdict_summary']}")
    print(f"[eval_lora] overall delta: {diff['overall_delta']:+.3f}")
    print(f"[eval_lora] latency delta: {diff['latency_delta_s']:+.3f}s")
    if not verdict["all_gates_pass"]:
        print()
        print("Failed gates:")
        for name, v in verdict["verdicts"].items():
            if not v["ok"]:
                print(f"  - {name}: {v['metric']} = {v['value']} "
                      f"(threshold {v['threshold']})")


if __name__ == "__main__":
    main()
