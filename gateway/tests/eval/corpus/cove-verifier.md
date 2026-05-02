# Molebie Chain-of-Verification (CoVe) verifier

CoVe is the Phase 3.1 post-generation verifier in
`gateway/app/services/verification.py`. After a long factual response
fires, CoVe decomposes it into atomic claims and re-checks each claim
in a fresh inference context against the cited chunk.

## Trigger conditions

CoVe only fires when ALL of the following hold:

1. `cove_enabled = True` in settings (default off)
2. RAG chunks were used in the response — CoVe **always** requires
   chunks to verify against; if no chunks are present, CoVe is
   short-circuited with `applied=False`. RAG-chunk presence is a
   **required** precondition.
3. The response is at least `cove_min_response_chars` long (default 500)
4. The query matches `is_verifiable_query` (numeric/date/specific-fact)

## Verifier sampling

Each per-claim verifier call uses:

- temperature: 0.0 (deterministic)
- top_p: 1.0
- max_tokens: small JSON envelope only
- mode: instant tier

The fresh-context, deterministic, factored design is what makes CoVe
strictly outperform "joint" CoVe on small models — the verifier can't
anchor on what the generator already committed to.

## Limits

- Maximum claims verified per response: 8 (`cove_max_claims`).
- Bounded concurrency for verifier calls: 4 (`cove_verifier_max_concurrent`).
- JSON-parse failure is conservatively treated as `unsupported`.

## Annotate, don't regenerate

Unsupported claims are marked with a ` [?]` suffix on the originating
sentence, NOT regenerated. Regenerating sections often introduces
fresh fabrications on 4B/9B models, so we annotate instead.
