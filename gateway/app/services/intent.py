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
import re

from app.config import get_settings
from app.services.metrics_registry import get_metrics_registry

# Self-referential / identity queries: pronouns + stopwords with no
# retrievable anchors ("what do you know about me?", "who am I"). Detected
# deterministically (regex, no LLM) because it must be cheap, zero-latency,
# and run on the FIRST turn — where the conversation-context query rewrite
# can't (it needs ≥2 prior messages). Matching expands the retrieval query
# with the owner's identity so vector + BM25 reach the owner's notes.
_IDENTITY_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwho\s+am\s+i\b",
        r"\bwhat'?s?\s+(?:do\s+you\s+know|can\s+you\s+tell\s+me)\s+about\s+me\b",
        r"\b(?:tell|teach|remind)\s+me\s+about\s+(?:me|myself)\b",
        r"\babout\s+myself\b",
        r"\bdescribe\s+me\b",
        r"\bwho\s+i\s+am\b",
        r"\bmy\s+(?:background|profile|bio|biography|identity|experience|"
        r"skills?|education|résumé|resume|cv|projects?)\b",
        r"\bwhat\s+do\s+you\s+know\s+about\s+my\b",
    )
)


def is_identity_query(message: str) -> bool:
    """Return True iff the message is a self-referential / identity query.

    Deterministic and side-effect free — safe to call on every turn.
    """
    if not message:
        return False
    return any(pat.search(message) for pat in _IDENTITY_PATTERNS)

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


async def classify_explain_intent(message: str) -> bool | None:
    """Classify the question's intent: True = explain, False = lookup,
    None = classification FAILED (timeout / error) — the caller must not
    treat a failure as a confident "lookup" verdict.

    The tri-state matters for the CoT auto-disable gate (T3): the old
    fail-closed bool meant every classifier timeout under load silently
    stripped chain-of-thought from thinking-mode turns. Definite
    non-answers (classifier disabled, empty message) still return False —
    those are genuine "no explain intent" verdicts, not failures. Latency
    is bounded by ``intent_classify_timeout`` (default 3.0s).
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


async def _classify_inner(message: str) -> bool | None:
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
        # repr(): str(TimeoutError()) is the empty string, which made every
        # load-induced timeout log as an anonymous "failed ()".
        print(f"[intent] LLM classify failed ({exc!r}) — intent unknown")
        return None
