# Molebie SelfCheck and Grounding-Judge layers

Two additional Phase 3 hallucination-mitigation services that complement
CoVe.

## SelfCheckGPT (Phase 3.3)

Reference-free consistency checker. Samples N additional responses at
higher temperature, then scores each sentence in the main response by
cross-sample disagreement. Hallucinated facts disagree across samples;
real ones stay stable.

Two backends, auto-selected:

1. **`selfcheckgpt` package** when installed. Uses a DeBERTa-v3-MNLI
   classifier (~350M params) to score sentence-vs-sample contradiction
   probability. The DeBERTa MNLI model is the canonical reference.
2. **Pure-function token-coverage fallback** when DeBERTa/MNLI is not
   available. Per-token coverage averaged across samples; needle-swap
   fabrications are caught because the unique tokens (e.g. wrong year,
   wrong location) appear in zero samples.

When SelfCheck fires:
- `selfcheck_enabled = True`
- No RAG chunks were used (orthogonal to CoVe)
- Response has factual content
- Response is at least `selfcheck_min_response_chars` long

## Grounding judge (Phase 3.2)

Reranker-as-judge: reuses Qwen3-Reranker-0.6B's yes/no head to score
each `(claim, cited_chunk)` pair without spending an LLM call.

The reranker is dispatched through one of two backends in
`gateway/app/services/reranker.py`:

1. **`qwen_causal`** — uses the Qwen3-Reranker's causal-LM head and
   computes a softmax over the yes/no logits. This is the primary
   backend for the Qwen reranker model.
2. **`cross_encoder`** — sentence-transformers cross-encoder for any
   non-Qwen reranker.

Both backends return a 0..1 score; below `judge_threshold` (default 0.5)
the claim is flagged. The judge is much cheaper than CoVe — no LLM
calls, just one reranker forward pass per claim — and is intended to
run as a gate before CoVe.

## What the judge does NOT catch

Numeric or date substitutions within an otherwise-relevant chunk
(e.g. "Mount Everest is 1234 metres" against a chunk that says "Mount
Everest is 8849 metres"). The reranker scores semantic relevance,
not literal grounding. CoVe handles those with the LLM-backed
verifier — the layered defense is intentional.
