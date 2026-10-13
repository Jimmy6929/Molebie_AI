# Threshold re-check — 2026-10-13

Triggered by two reranker changes: scoring is now batched on the GPU in
half precision, and each pair is wrapped in the chat template from the
Qwen3-Reranker model card (system prefix + assistant suffix). Both logs here
come from the same seeded 6-doc corpus (31 chunks, 30 candidates per query,
query rewrite and contextual retrieval off), so the two columns compare like
for like. They do **not** compare with the 2026-08-14 table, which used a
different corpus build.

## Batching alone changes nothing

Old prompt, old code (CPU, one pair at a time) against old prompt, new code
(MPS, one batch): top-1 scores on the 17 grounded rows differ by at most
0.0032. Rerank time for 30 candidates went from a 23.3 s median to 2.0 s
(M-series MacBook Pro); 2.7 s with the longer official prompt.

## Measured distributions (top-1 rerank score per query)

Official prompt (`golden-reranker-scores.log`):

| category | n | min | p20 | p50 | p80 | p95 | max |
|---|---|---|---|---|---|---|---|
| rag_grounded | 17 | 0.844 | 0.995 | 0.999 | 1.000 | 1.000 | 1.000 |
| must_abstain | 10 | 0.000 | 0.001 | 0.002 | 0.003 | 0.006 | 0.009 |
| adversarial_premise | 15 | 0.000 | 0.001 | 0.002 | 0.014 | 0.496 | 0.993 |
| rag_grounded_negative | 2 | 0.011 | — | 0.475 | — | — | 0.939 |

Old prompt, same corpus (`golden-reranker-scores-old-prompt.log`):

| category | n | min | p20 | p50 | p80 | p95 | max |
|---|---|---|---|---|---|---|---|
| rag_grounded | 17 | 0.861 | 0.932 | 0.994 | 0.999 | 1.000 | 1.000 |
| must_abstain | 10 | 0.271 | 0.300 | 0.339 | 0.408 | 0.467 | 0.475 |
| adversarial_premise | 15 | 0.286 | 0.365 | 0.405 | 0.505 | 0.663 | 0.941 |
| rag_grounded_negative | 2 | 0.316 | — | 0.628 | — | — | 0.939 |

The adversarial maximum is the Everest row again: the corpus contains that
passage, so the high score is earned.

## Verdict: 0.25 / 0.5 / 0.7 unchanged

- **floor 0.25**: unanswerable queries now top out at 0.009 (was 0.475 on
  this corpus, i.e. above the floor). The lowest-scoring chunk that carries
  a grounded row's answer is 0.488. The floor sits in the gap.
- **MODERATE 0.5 / HIGH 0.7**: every grounded row's top-1 is ≥ 0.844, so all
  17 still route HIGH.
- **judge_threshold 0.5**: seven hand-built claim/chunk pairs. Supported
  claims score ≥ 0.997, off-topic fabrications ≤ 0.004 (one sat at 0.477
  under the old prompt). Small sample; the full judge ablation needs the LLM
  backends and was not run.

## What got worse

Ordering among relevant chunks. The chunk containing the expected answer is
ranked first on 11 of 17 grounded rows, down from 13; MRR over the reranked
top 8 is 0.772, down from 0.843. Under the official prompt, short
document-intro chunks ("Notes on Molebie's RAG pipeline as configured after
Phase 1…") score 0.99+ on questions about that configuration and outrank
the chunk with the specific value. Scores were checked against the model
card's reference implementation on CPU in float32 and match, so this is the
model, not the wiring.

It does not cost answers on this set: the answer chunk is in the top 8 on
17 of 17 rows (16 before), and the top 8 all reach the prompt. A
task-specific instruction was tried once ("…retrieve passages that contain
the specific answer…"): MRR 0.814, but unanswerable queries rose to 0.669,
which breaks the floor. Not adopted.

## Live-vault probes

The four 2026-08-14 probes were re-run against a copy of the owner's vault.
The vault has changed since, and two of the four no longer have a matching
note, so the old table cannot be reproduced. Score logs are not kept here
because they name personal notes.

| probe | old prompt | official prompt | routing |
|---|---|---|---|
| "write merge_ranges" | 0.419 | 0.000 | was LOW with a stray chunk → now dropped ✓ |
| river-crossing riddle | 0.045 | 0.000 | dropped ✓ |
| "which embedding model did I decide on" (note no longer in vault) | 0.369 | 0.046 | was LOW with a stray chunk → now dropped ✓ |
| "3 bullets about local-first software" (essay no longer in vault) | 0.096 | 0.121 | dropped, both |

## Reproduce

```
DATA_DIR=<seeded> RAG_QUERY_REWRITE_ENABLED=false \
  python gateway/tests/eval/run_rag_only.py \
    --user-id 00000000-0000-0000-0000-000000000001 2>&1 | tee <log>
python gateway/tests/eval/calibrate_thresholds.py --log <log>
```

The harness now exits on its own and prints hit rate, MRR and rerank time.
