# Threshold recalibration — 2026-08-14

Inputs frozen in this directory; produced with the fixed retrieval pipeline
(relevance floor applied on every path) against the post-T4 golden set
(52 rows, seeded 6-doc corpus) plus 4 off-golden probes run against BOTH the
seeded corpus and the owner's live 101-doc vault.

## Measured distributions (top-1 rerank score per query)

| category | n | min | p20 | p50 | p80 | p95 | max |
|---|---|---|---|---|---|---|---|
| rag_grounded | 17 | 0.373 | 0.917 | 0.992 | 1.000 | 1.000 | 1.000 |
| must_abstain | 9 | 0.076 | 0.121 | 0.130 | 0.158 | 0.181 | 0.190 |
| adversarial_premise | 15 | 0.037 | 0.097 | 0.270 | 0.304 | 0.657 | 0.941 |
| rag_grounded_negative | 2 | 0.110 | — | 0.524 | — | — | 0.938 |

## Why NOT the calibrator's raw suggestions (HIGH=1.0, MOD=0.992, floor=0.837)

Percentiles of a 6-doc corpus overfit: the "noise floor" suggestion was
poisoned by *legitimately relevant* retrievals in abstain-ish categories
(ground_001 asks about Everest; the corpus literally contains the Everest
example passage, scoring 0.94 on merit). A 0.837 floor would blank most
real-vault retrievals.

## Chosen values

- **floor 0.05 → 0.25**: above the truly-unanswerable band (abstain max
  0.190; live riddle probe 0.130) and below the weakest genuine answer
  (0.373, rag_017's context chunk), with margin both sides.
- **MODERATE 0.3 → 0.5**: the 0.3–0.5 band is weakly-related pollution —
  the eval's refusal bug lived here (live-vault probe: "write merge_ranges"
  scored 0.342 against a Paul Graham essay and was answered "not in your
  notes" under the strict template). Now routes LOW → generative.
- **HIGH = 0.7 unchanged**: grounded answers cluster ≥ 0.9; 0.7 keeps
  headroom without demoting any measured real answer.

## Probe validation (live vault)

| probe | top-1 | old behavior | new behavior |
|---|---|---|---|
| "write merge_ranges" (eval E1 refusal repro) | 0.342 | MODERATE → strict → refused | LOW → generative ✓ |
| river-crossing riddle | 0.130 | MODERATE-ish framing pollution | below floor → dropped ✓ |
| "which embedding model did I decide on" | 1.000 | HIGH | HIGH ✓ (no false negative) |
| "3 bullets about local-first software" (eval D3) | 0.844 | HIGH → strict | HIGH → strict (unchanged) |

Known limitation: the D3 case is *semantically* relevant by reranker lights
(the essay genuinely discusses local vs server software) — no score
threshold separates "relevant but answers the opposite concept". The fix
for that class is intent-aware retrieval gating (TARG-style pre-retrieval
draft gate) — tracked as future work under T6b/roadmap.

## Reproduce

```
bash: seed corpus → DATA_DIR=<seeded> python gateway/tests/eval/run_rag_only.py \
        --user-id 00000000-0000-0000-0000-000000000001
python gateway/tests/eval/calibrate_thresholds.py --log <captured log>
```

Note: run_rag_only currently hangs at process exit after completing all
queries (non-daemon model thread) — kill it after "N/N" prints; scores are
already flushed. Harness bug, tracked separately.
