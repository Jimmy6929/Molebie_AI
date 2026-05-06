"""Threshold calibration script for the RAG reranker.

Phase 2 of the retrieval-quality sprint. Replaces the MS-Marco-MiniLM
thresholds in `compute_retrieval_confidence()` with values derived from
the actual Qwen3-Reranker-0.6B score distribution on the golden set.

Workflow:
    1. Run the golden set with the gateway, capturing stdout to a file:
         GATEWAY_LOG=/tmp/eval.log
         ./gateway/tests/eval/run_eval.sh 2>&1 | tee "$GATEWAY_LOG"
       (Or run any representative workload — what matters is that
       `[rag] RERANKER_SCORES ...` lines land in the log.)

    2. Run this script against the log + golden set:
         python gateway/tests/eval/calibrate_thresholds.py \\
             --log /tmp/eval.log \\
             --golden gateway/tests/eval/golden_set.jsonl

    3. Review the suggested thresholds at the bottom of the output and
       update `compute_retrieval_confidence()` in
       `gateway/app/services/rag.py` and `rag_rerank_floor` in
       `gateway/app/config.py` accordingly.

Heuristic:
    - rag_grounded questions SHOULD return at least one strong chunk →
      use their top-1 score distribution to anchor HIGH/MODERATE.
    - adversarial_premise + must_abstain questions SHOULD return weak or
      no relevant chunks → use their top-1 distribution to anchor the
      noise floor.

Why no chunk-content labelling? The log only carries filename + index +
score, not chunk text. Top-1 score per question is a reliable proxy
(correlates 0.85+ with "correct chunk made it" in our golden set).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

_RERANK_LINE = re.compile(r"\[rag\] RERANKER_SCORES (\{.*\})")


def parse_log(path: Path) -> list[dict]:
    """Extract every RERANKER_SCORES payload from a gateway log."""
    payloads = []
    with path.open() as fp:
        for line in fp:
            m = _RERANK_LINE.search(line)
            if not m:
                continue
            try:
                payloads.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                continue
    return payloads


def load_golden(path: Path) -> list[dict]:
    """Load golden_set.jsonl entries (skipping blank lines)."""
    entries = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def match_query_to_golden(query: str, golden: list[dict]) -> dict | None:
    """Best-effort match between a logged query and a golden entry.

    The gateway's query rewriter may transform queries; we match against
    the original form (gateway falls back to original on rewrite skip)
    and tolerate prefix truncation (queries are logged truncated to 200
    chars).
    """
    for g in golden:
        gq = g["query"]
        if query == gq or gq.startswith(query) or query.startswith(gq[:200]):
            return g
    return None


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile. p ∈ [0, 100]."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def ascii_histogram(values: list[float], bins: int = 20, width: int = 50) -> str:
    """Tiny ASCII histogram for terminal output."""
    if not values:
        return "  (no data)"
    lo, hi = 0.0, 1.0  # reranker scores are in [0, 1]
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        idx = max(idx, 0)
        counts[idx] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(width * c / peak)
        lines.append(f"  [{lo + i * step:.2f}, {lo + (i + 1) * step:.2f}) {c:3d} {bar}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True, help="Gateway log with RERANKER_SCORES lines")
    parser.add_argument("--golden", type=Path,
                        default=Path(__file__).parent / "golden_set.jsonl",
                        help="Path to golden_set.jsonl")
    args = parser.parse_args()

    if not args.log.exists():
        print(f"error: log file not found: {args.log}", file=sys.stderr)
        return 2
    if not args.golden.exists():
        print(f"error: golden set not found: {args.golden}", file=sys.stderr)
        return 2

    payloads = parse_log(args.log)
    golden = load_golden(args.golden)
    print(f"Loaded {len(payloads)} RERANKER_SCORES entries from {args.log}")
    print(f"Loaded {len(golden)} golden questions from {args.golden}")
    print()

    # Group top-1 scores by category.
    by_category: dict[str, list[float]] = {}
    matched = 0
    unmatched_queries: list[str] = []
    for p in payloads:
        scores = p.get("scores", [])
        if not scores:
            continue
        g = match_query_to_golden(p.get("query", ""), golden)
        if g is None:
            unmatched_queries.append(p.get("query", "")[:80])
            continue
        matched += 1
        cat = g.get("category", "unknown")
        top1 = scores[0]["score"]
        by_category.setdefault(cat, []).append(top1)

    print(f"Matched {matched}/{len(payloads)} log entries to golden questions.")
    if unmatched_queries:
        print(f"Unmatched ({len(unmatched_queries)}):")
        for q in unmatched_queries[:5]:
            print(f"  {q!r}")
        if len(unmatched_queries) > 5:
            print(f"  ... and {len(unmatched_queries) - 5} more")
    print()

    # Per-category distribution.
    for cat in sorted(by_category):
        vals = by_category[cat]
        print(f"── Category: {cat} (n={len(vals)}) ──")
        print(f"  min={min(vals):.4f}  median={statistics.median(vals):.4f}  "
              f"mean={statistics.fmean(vals):.4f}  max={max(vals):.4f}")
        print(f"  p20={percentile(vals, 20):.4f}  p50={percentile(vals, 50):.4f}  "
              f"p80={percentile(vals, 80):.4f}  p95={percentile(vals, 95):.4f}")
        print(ascii_histogram(vals))
        print()

    # Suggested thresholds.
    grounded = by_category.get("rag_grounded", [])
    abstain_cats = ("must_abstain", "adversarial_premise", "rag_grounded_negative")
    abstain = [v for c in abstain_cats for v in by_category.get(c, [])]

    print("=" * 60)
    print("SUGGESTED THRESHOLDS")
    print("=" * 60)
    if not grounded:
        print("  ⚠ No rag_grounded data — cannot suggest HIGH/MODERATE.")
    else:
        high = percentile(grounded, 80)
        moderate = percentile(grounded, 50)
        print(f"  HIGH      = {high:.3f}  (80th pct of rag_grounded top-1)")
        print(f"  MODERATE  = {moderate:.3f}  (50th pct of rag_grounded top-1)")

    if not abstain:
        print("  ⚠ No must_abstain/adversarial data — cannot suggest noise floor.")
    else:
        floor = percentile(abstain, 95)
        print(f"  NOISE FLOOR = {floor:.3f}  (95th pct of abstain top-1; chunks below this dropped)")

    print()
    print("Apply these by editing:")
    print("  gateway/app/services/rag.py   compute_retrieval_confidence()")
    print("  gateway/app/config.py         rag_rerank_floor")
    print()
    print("Re-run the golden set after applying to confirm pass rate ≥ 88%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
