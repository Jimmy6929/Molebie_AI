# Reranker input text — 2026-10-14

The reranker scored each chunk's bare body. The embedder and BM25 both see
more: the heading path (`h1 > h2 > h3`), the note title, and, with
contextual retrieval on, an LLM-written context line. The 2026-10-13
re-check found the cost: a short intro chunk that names the note's subject
("Notes on Molebie's RAG pipeline … after Phase 1") outranked the chunk
holding the answer ("- Chunk size: 512 tokens"), which sits under a heading
and never repeats the subject.

## What the reranker should see

Same 52 golden queries, same corpus, only the reranker's input changed.
Corpus: the six seeded eval docs, built with contextual retrieval on, so
every variant was available.

| reranker input | answer chunk ranked 1st | MRR (top 8) | weakest answer-chunk score | must_abstain top-1 max |
|---|---|---|---|---|
| body only (before) | 11/17 | 0.772 | 0.488 | 0.009 |
| LLM context + body | 14/17 | 0.879 | 0.822 | 0.002 |
| heading path + note title + body | 15/17 | 0.922 | 0.995 | 0.001 |
| **heading path + body (chosen)** | **15/17** | **0.931** | **0.995** | **0.005** |

Chosen: heading path only. It matches or beats the other variants on this
set, needs no LLM, and works with contextual retrieval off.

The note title was dropped after a probe on the owner's live vault. With
the title, "Write exactly 3 bullet points about local-first software"
scored 0.504 against a personal note titled `Software.md` (0.121 without),
crossing the MODERATE line on the strength of one generic word. Heading
path only leaves it at 0.121.

## Distributions with the chosen input

`golden-reranker-scores.log` (corpus without contextual retrieval) and
`golden-reranker-scores-contextual.log` (with it) give the same ranking
results: MRR 0.931, answer first on 15/17, answer chunk in the top 8 on
17/17.

| category | min | median | max |
|---|---|---|---|
| rag_grounded | 0.995 | 1.000 | 1.000 |
| must_abstain | 0.000 | 0.001 | 0.005 |
| adversarial_premise | 0.000 | 0.004 | 0.927 |
| rag_grounded_negative | 0.003 | — | 0.535 |

The adversarial maximum is the Everest row again (the passage is in the
corpus, so the score is earned).

## Thresholds: unchanged

- **floor 0.25**: unanswerable ≤ 0.005, weakest answer chunk 0.995.
- **MODERATE 0.5 / HIGH 0.7**: every grounded row's top-1 is ≥ 0.995.
- **Live-vault probes** (copy of the owner's vault, nomic embedder, 5
  candidates): merge_ranges 0.000, riddle 0.000, local-first 0.121,
  embedding-model question 0.046 — identical routing to before. Logs not
  kept: they name personal notes.

## Limits

17 grounded questions over a 6-doc corpus written alongside them. The
direction is consistent across both corpus builds, but the size of the gain
on a real vault is unmeasured. Notes with no headings get no prefix and
behave exactly as before.
