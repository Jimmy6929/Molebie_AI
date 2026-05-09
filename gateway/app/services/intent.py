"""Explain-vs-lookup intent classifier.

Used to decide whether a RAG-grounded query wants a terse cited bullet
list (lookup mode) or a multi-paragraph synthesis with inline citations
(explain mode). Cloned from ``WebSearchService._classify_with_llm_inner``
(see ``app/services/web_search.py``) — same shape, same instant-tier
backend, same fail-closed behaviour.

Why a classifier and not regex: production RAG systems (Perplexity,
Glean) use LLM-based intent classification because surface keyword
patterns miss paraphrases ("walk me through", "tell me how this works")
and false-positive on phrasings that contain "explain" but aren't
explanatory ("can you explain in one word"). The instant Qwen 4B is
already loaded, so the classification call is cheap (<300ms) and runs
in parallel with the multi-second RAG rerank — zero wall-clock cost.
"""
from __future__ import annotations

import asyncio

from app.config import get_settings
from app.services.metrics_registry import get_metrics_registry

_PROMPT = (
    "Classify the user's question as EXPLAIN or LOOKUP.\n\n"
    "EXPLAIN — the user wants a multi-paragraph synthesis. Examples:\n"
    "  - \"explain how X works\"\n"
    "  - \"how does Y handle Z\"\n"
    "  - \"walk me through the architecture\"\n"
    "  - \"why does the system behave this way\"\n"
    "  - \"tell me about the trade-offs\"\n\n"
    "LOOKUP — the user wants a fact, name, date, or short specific answer:\n"
    "  - \"what is X\"\n"
    "  - \"when did Y happen\"\n"
    "  - \"who wrote Z\"\n"
    "  - \"list the features\"\n"
    "  - \"find the file where I mentioned X\"\n\n"
    "If unsure, answer LOOKUP.\n\n"
    "Question: \"{message}\"\n\n"
    "One word:"
)


async def classify_explain_intent(message: str) -> bool:
    """Return True iff the question is best served by an explanatory answer.

    Fail-closed: any error / timeout returns False (caller will use the
    strict lookup template, current behaviour). Latency is bounded by
    ``intent_classify_timeout`` (default 3.0s).
    """
    settings = get_settings()
    if not settings.intent_classify_enabled:
        return False
    cleaned = message.strip()
    if not cleaned:
        return False

    metrics = get_metrics_registry()
    async with metrics.subsystem_timer("intent.classify"):
        return await _classify_inner(cleaned)


async def _classify_inner(message: str) -> bool:
    settings = get_settings()
    try:
        from app.services.inference import get_inference_service

        inference = get_inference_service()
        prompt = _PROMPT.format(message=message)
        result = await asyncio.wait_for(
            inference.generate_response(
                messages=[{"role": "user", "content": prompt}],
                mode="instant",
                max_tokens=settings.intent_classify_max_tokens,
                temperature=0.0,
            ),
            timeout=settings.intent_classify_timeout,
        )
        answer = result.get("content", "").strip().upper()
        is_explain = "EXPLAIN" in answer
        preview = message[:60].replace("\n", " ")
        print(f"[intent] LLM classify: '{preview}' → {answer} → explain={is_explain}")
        return is_explain
    except Exception as exc:
        print(f"[intent] LLM classify failed ({exc}), defaulting to lookup")
        return False
