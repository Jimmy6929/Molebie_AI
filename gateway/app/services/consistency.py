"""
Self-consistency for verifiable queries (Phase 2 task 2.3).

For factual / numeric / classification queries, sample N responses and
majority-vote on the extracted answer. Hallucinated answers tend to
disagree across samples; correct answers stay stable.

We use Early-Stopping Self-Consistency (ESC): stop sampling once a
threshold of agreement is reached. Saves ~60–80% of compute compared to
running every sample to completion when the model is confident.

Trigger heuristic — only run on queries that ASK for a specific fact:
greetings, summaries, and open-ended generation are filtered out so the
extra inference cost lands where it actually pays off.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import Any

from app.config import Settings


# ── Trigger heuristic ───────────────────────────────────────────────────

# Phrases that signal "asks for a specific fact" — lowercase substrings.
_VERIFIABLE_TRIGGERS = (
    "what is",
    "what's",
    "what are",
    "how many",
    "how much",
    "which ",
    "who is",
    "who was",
    "when did",
    "when was",
    "where is",
    "is it true",
    "is it possible",
    "does ",
    "do i ",
    "can ",
    "calculate",
    "compute",
)

# Patterns that override the trigger — anything that smells open-ended.
_OPEN_ENDED_BLOCKERS = (
    "summarize",
    "summarise",
    "explain",
    "describe",
    "tell me about",
    "write a",
    "draft",
    "outline",
    "brainstorm",
    "ideas for",
    "how do i",   # "how do i set this up?" reads as instructions, not fact
)


def is_verifiable_query(message: str) -> bool:
    """Return True when the query asks for a specific fact / number /
    classification — the cases where self-consistency actually helps."""
    if not message:
        return False
    lowered = message.lower()
    if any(blocker in lowered for blocker in _OPEN_ENDED_BLOCKERS):
        return False
    return any(trigger in lowered for trigger in _VERIFIABLE_TRIGGERS)


# ── Answer normalisation ────────────────────────────────────────────────

# Strip whitespace, leading articles, citations, trailing punctuation —
# things that vary across samples without changing the actual answer.
_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_CITATION_RE = re.compile(r"\s*\[S\d+\]")
_TRAIL_PUNCT_RE = re.compile(r"[.!?,;:]+$")


def normalise_answer(text: str) -> str:
    """Reduce a response to a canonical form for vote-counting.

    Heuristic — for factual queries the answer is usually the first
    sentence or the first numeric expression. We strip citations/articles/
    punctuation and lowercase. Two samples answering "Paris" and "The Paris."
    must collapse to the same key.
    """
    if not text:
        return ""
    cleaned = _CITATION_RE.sub("", text).strip()
    # Take the first sentence — most factual answers fit in one
    sentence = re.split(r"(?<=[.!?])\s", cleaned, maxsplit=1)[0]
    sentence = _ARTICLE_RE.sub("", sentence).strip()
    sentence = _TRAIL_PUNCT_RE.sub("", sentence)
    return sentence.lower()


# ── Sampling + voting ───────────────────────────────────────────────────


async def _generate_one(
    inference: Any,
    messages: list[dict[str, Any]],
    mode: str,
    enable_thinking: bool | None,
) -> str:
    """Run one inference call and return the cleaned content. We swallow
    failures so a single bad sample doesn't poison the vote — the caller
    sees a shorter list and decides based on what came back."""
    try:
        result = await inference.generate_response(
            messages=messages,
            mode=mode,
            enable_thinking=enable_thinking,
        )
        return result.get("content", "") or ""
    except Exception as exc:
        print(f"[consistency] Sample failed: {type(exc).__name__}: {exc}")
        return ""


async def vote_with_self_consistency(
    inference: Any,
    messages: list[dict[str, Any]],
    mode: str,
    settings: Settings,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    """Sample multiple responses, normalise + vote, return the winner.

    Result keys:
      ``content``         — the response chosen by majority (full text from
                            the first matching sample, not the normalised key)
      ``samples``         — the raw responses, in completion order
      ``vote_counts``     — Counter dict of normalised key → count
      ``confident``       — True when a clear majority emerged
      ``early_stopped``   — True when ESC kicked in
    """
    n = max(2, settings.self_consistency_samples)
    early_stop_at = max(2, min(settings.self_consistency_early_stop, n))
    concurrency = max(1, settings.self_consistency_max_concurrent)

    sem = asyncio.Semaphore(concurrency)

    async def bounded():
        async with sem:
            return await _generate_one(inference, messages, mode, enable_thinking)

    # Schedule all samples; we collect early but cancel pending tasks once
    # an answer hits the early-stop threshold.
    tasks = [asyncio.create_task(bounded()) for _ in range(n)]
    samples: list[str] = []
    vote_counts: Counter[str] = Counter()
    early_stopped = False

    try:
        for fut in asyncio.as_completed(tasks):
            sample = await fut
            samples.append(sample)
            key = normalise_answer(sample)
            if key:
                vote_counts[key] += 1
                if vote_counts[key] >= early_stop_at:
                    early_stopped = True
                    break
    finally:
        # Cancel anything still pending so we don't pay for samples we've
        # already decided we don't need.
        for t in tasks:
            if not t.done():
                t.cancel()
        # Drain cancellations so they don't fire as warnings later.
        await asyncio.gather(*tasks, return_exceptions=True)

    if not vote_counts:
        return {
            "content": "I'm getting inconsistent results — flagging as uncertain.",
            "samples": samples,
            "vote_counts": {},
            "confident": False,
            "early_stopped": early_stopped,
        }

    top_key, top_count = vote_counts.most_common(1)[0]
    confident = top_count >= early_stop_at
    # Pick the raw text of the first sample that matched the winning key —
    # we want a real model-written response back, not the normalised form.
    winner_text = next(
        (s for s in samples if normalise_answer(s) == top_key),
        samples[0] if samples else "",
    )

    return {
        "content": winner_text,
        "samples": samples,
        "vote_counts": dict(vote_counts),
        "confident": confident,
        "early_stopped": early_stopped,
    }
