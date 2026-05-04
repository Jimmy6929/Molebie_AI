# Molebie strict-grounding prompt and citation rules

The strict-grounding RAG prompt lives in `gateway/prompts/system_rag.txt`
and is loaded whenever retrieval fires. It enforces the citation
contract Molebie uses to keep responses verifiable.

## Citation format

Every factual claim sourced from a retrieved chunk must carry a `[S#]`
inline citation, where # is the 1-indexed position of the chunk in
the surviving top-5. A claim without a `[S#]` tag is treated as
ungrounded and may be flagged by the citation validator.

The number of citations required per factual claim: **at least one
`[S#]` tag per claim**. Multiple sources for one claim get multiple
tags, e.g. `[S1][S3]`.

## Fallback string when retrieval fails

If the retrieved context can't answer the question, the model is
instructed to emit the exact string:

> I don't have that in your notes.

Verbatim, not paraphrased. The fallback is matched as a literal
substring by the citation validator and the eval harness.

## Why the prompt is strict

Small models (Qwen3.5-4B/9B) drift toward fluent fabrication when
the retrieved context is sparse but the question is specific. The
prompt's UNCERTAINTY / EVIDENCE / NUMBERS / BREVITY / FORMAT rules
push the model toward the fallback rather than confabulating. The
RAG-specific overlay (`system_rag.txt`) adds the inline-citation
requirement.
