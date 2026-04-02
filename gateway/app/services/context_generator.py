"""
Contextual retrieval: generate context prefixes for document chunks.

Uses the local LLM (instant tier) to produce a short context prefix
per chunk, following Anthropic's contextual retrieval technique.
The prefix situates the chunk within the overall document, improving
search retrieval quality by 35-49%.

See: https://www.anthropic.com/news/contextual-retrieval
"""

from typing import List, Optional

import httpx

from app.config import Settings


_CONTEXT_PROMPT = """\
<document>
{doc_text}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the context, nothing else."""


def _truncate_doc(doc_text: str, max_chars: int) -> str:
    """Truncate large documents, keeping start and end for context."""
    if len(doc_text) <= max_chars:
        return doc_text
    # Keep first 75% and last 25% of budget
    head_budget = int(max_chars * 0.75)
    tail_budget = max_chars - head_budget
    return doc_text[:head_budget] + "\n\n[...document truncated...]\n\n" + doc_text[-tail_budget:]


async def _generate_context(
    doc_text: str,
    chunk_text: str,
    settings: Settings,
) -> Optional[str]:
    """Generate a context prefix for a single chunk using the local LLM."""
    mode = settings.rag_context_llm_mode  # "instant" or "thinking"

    # Resolve endpoint URL
    if mode in ("thinking", "thinking_harder"):
        base_url = settings.inference_thinking_url
        api_prefix = settings.inference_thinking_api_prefix
        model = settings.get_model_for_mode("thinking")
    else:
        base_url = settings.inference_instant_url
        api_prefix = settings.inference_instant_api_prefix
        model = settings.get_model_for_mode("instant")

    if not base_url:
        return None

    prompt = _CONTEXT_PROMPT.format(
        doc_text=_truncate_doc(doc_text, settings.rag_context_max_doc_chars),
        chunk_text=chunk_text,
    )

    url = f"{base_url}{api_prefix}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": settings.rag_context_max_tokens,
        "temperature": 0.0,  # deterministic for consistency
        "stream": False,
    }

    try:
        # Short connect timeout (2s) so we fail fast when the LLM is offline.
        # Read timeout stays at 30s for slow inference on large chunks.
        timeout = httpx.Timeout(30.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Strip any thinking tags if present
            if "</think>" in content:
                content = content.split("</think>", 1)[-1].strip()
            return content if content else None
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        # Re-raise connection errors so generate_batch() can bail out early
        raise
    except Exception as exc:
        print(f"[context_gen] LLM call failed: {type(exc).__name__}: {exc}")
        return None


async def generate_batch(
    doc_text: str,
    chunks: List[str],
    settings: Settings,
) -> List[Optional[str]]:
    """Generate context prefixes for all chunks in a document.

    Processes chunks sequentially to avoid overwhelming the local LLM.
    Returns a list of context strings (or None for failures) parallel
    to the input chunks list.
    """
    results: List[Optional[str]] = []
    llm_reachable = True

    for i, chunk in enumerate(chunks):
        # If a previous chunk failed due to connection error, skip the rest
        # rather than waiting for each one to timeout individually.
        if not llm_reachable:
            results.append(None)
            continue

        try:
            context = await _generate_context(doc_text, chunk, settings)
            results.append(context)
            if context:
                print(f"[context_gen] Chunk {i+1}/{len(chunks)}: {len(context)} chars")
            else:
                print(f"[context_gen] Chunk {i+1}/{len(chunks)}: (no context)")
        except Exception as exc:
            print(f"[context_gen] Chunk {i+1}/{len(chunks)} failed: {type(exc).__name__}: {exc}")
            results.append(None)
            # If we can't connect at all, don't waste time on remaining chunks
            if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                print(f"[context_gen] LLM unreachable — skipping remaining {len(chunks) - i - 1} chunks")
                llm_reachable = False

    return results
