"""
SelfCheckGPT — Phase 3 task 3.3.

Reference-free consistency check. We sample N additional responses at
higher temperature, then score each sentence in the main response by
how consistent it is across those samples. Hallucinated facts disagree
across samples; real facts stay stable.

Crucially this needs **no RAG context and no reference**. That's what
makes 3.3 orthogonal to CoVe (3.1) and the Judge (3.2): they verify
*against retrieved chunks*, this verifies *against the model's own
multi-sample distribution*. The combined defense:

  - 3.1 CoVe       → claims that don't match cited chunk
  - 3.2 Judge      → cheap gate using reranker
  - 3.3 SelfCheck  → fabrications when there's no chunk to compare to

Two backends, auto-selected:

  1. ``selfcheckgpt`` package (DeBERTa-v3-MNLI, ~350M params). The
     canonical reference implementation. Use when installed.
  2. Pure-function token-consistency fallback. For each sentence we
     extract distinctive tokens (numbers, dates, proper-noun pairs)
     and compute the fraction of samples that contain those tokens.
     Inconsistent ↔ low fraction. Not as precise as NLI but the
     pipeline still produces signal on machines where transformers
     isn't available.

Default off — N extra full-response generations is expensive.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.services.verification import (
    _DATE_RE,
    _NUMBER_RE,
    _PROPER_NOUN_PAIR_RE,
    _SENTENCE_SPLIT_RE,
    _URL_RE,
    Claim,
    Verdict,
    annotate_response,
)

# ── Sentence selection ────────────────────────────────────────────────────


_HEDGE_PHRASES = (
    "i'm not sure", "i don't know", "i think", "perhaps", "maybe",
    "it depends", "could be",
)


def _factual_sentences(response: str) -> list[str]:
    """Pick out the sentences worth checking. Numbers / URLs / dates /
    proper-noun pairs — same heuristic as ``verification._has_specific_claim``,
    plus filter out hedges and questions which don't carry assertions."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(response) if s.strip()]
    out: list[str] = []
    for s in sentences:
        lowered = s.lower()
        if any(h in lowered for h in _HEDGE_PHRASES):
            continue
        if s.endswith("?"):
            continue
        if (
            _NUMBER_RE.search(s)
            or _URL_RE.search(s)
            or _DATE_RE.search(s)
            or _PROPER_NOUN_PAIR_RE.search(s)
        ):
            out.append(s)
    return out


# ── Trigger heuristic ─────────────────────────────────────────────────────


def should_selfcheck(
    response: str,
    rag_chunks: list[dict[str, Any]] | None,
    settings: Settings,
) -> tuple[bool, str]:
    """Returns ``(applicable, reason_if_not)``.

    SelfCheck fires when there's NO RAG context. With RAG context, CoVe
    + Judge are strictly better — they have ground truth to compare
    against. SelfCheck is for the no-reference case (general-knowledge
    Qs, off-topic chats) where we need *some* signal."""
    if not settings.selfcheck_enabled:
        return False, "selfcheck_disabled"
    if rag_chunks:
        return False, "rag_present_use_cove"
    if len(response) < settings.selfcheck_min_response_chars:
        return False, "response_too_short"
    if not _factual_sentences(response):
        return False, "no_factual_sentences"
    return True, ""


# ── Backends ───────────────────────────────────────────────────────────────


# Distinctive tokens extracted from a sentence — used by the fallback
# scorer. Multiple narrow regexes instead of one alternated regex: that
# way ``Mount Everest`` produces tokens for both the *phrase* (low recall,
# high precision) AND the individual words ``Mount`` / ``Everest`` (higher
# recall). With the alternated form, the phrase match would consume both
# words at once, and a sample mentioning only ``Everest`` would falsely
# look "missing".
_NUMBER_FALLBACK_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_URL_FALLBACK_RE = re.compile(r"\bhttps?://\S+", re.IGNORECASE)
_YEAR_FALLBACK_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_PROPER_PAIR_FALLBACK_RE = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")
# ≥5-char alphanumeric — broader catch-net for proper nouns and content
# words. Apostrophe-S (`Everest's`) gets clipped to the bare stem.
_CONTENT_WORD_FALLBACK_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{4,}\b")


def _distinctive_tokens(text: str) -> set[str]:
    """Lowercase set of distinctive tokens — numbers, URLs, years,
    proper-noun pairs, and 5+-char content words. Tokens overlap on
    purpose: ``Mount Everest`` produces ``mount everest``, ``mount``,
    AND ``everest`` so cross-sample matching has multiple footholds."""
    out: set[str] = set()
    for rx in (
        _NUMBER_FALLBACK_RE,
        _URL_FALLBACK_RE,
        _YEAR_FALLBACK_RE,
        _PROPER_PAIR_FALLBACK_RE,
        _CONTENT_WORD_FALLBACK_RE,
    ):
        out.update(t.lower() for t in rx.findall(text))
    return out


def fallback_inconsistency(
    sentence: str,
    samples: list[str],
) -> float:
    """Return a 0..1 inconsistency score: 0 = every distinctive token in
    the sentence appears in every sample, 1 = no sample contains any of
    them. Approximates NLI's contradiction signal cheaply.

    Uses **per-token coverage**, not "any-overlap":
      Coverage(t)  = (# samples that contain t) / (# samples)
      Inconsistency = 1 - mean(coverage(t) for t in sentence_tokens)

    This matters for needle-swap fabrications. "Apollo 11 landed on
    Saturn in 1969" against samples talking about "Apollo 11 landed on
    the Moon" would falsely score consistent under any-overlap (Apollo,
    11 are present everywhere). Per-token coverage notices that
    ``saturn`` and ``1969`` are missing in 3-of-3 samples and pulls the
    score above threshold.
    """
    if not samples:
        return 0.0
    sent_tokens = _distinctive_tokens(sentence)
    if not sent_tokens:
        return 0.0          # nothing distinctive to check; treat as consistent
    sample_token_sets = [_distinctive_tokens(s) for s in samples]
    coverages: list[float] = []
    for tok in sent_tokens:
        present_in = sum(1 for st in sample_token_sets if tok in st)
        coverages.append(present_in / len(samples))
    avg_coverage = sum(coverages) / len(coverages)
    return max(0.0, 1.0 - avg_coverage)


class _NLIBackend:
    """Wraps either selfcheckgpt's SelfCheckNLI or the rule-based
    fallback behind a uniform ``score(sentences, samples) -> list[float]``
    interface. Returns one inconsistency score per input sentence."""

    def __init__(self):
        self._impl: str | None = None
        self._nli = None

    def _load(self) -> None:
        if self._impl is not None:
            return
        try:
            from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
            # Use CPU by default; users with MPS/CUDA can override the
            # device on the env if they wire it through.
            self._nli = SelfCheckNLI(device="cpu")
            self._impl = "selfcheckgpt_nli"
            print("[selfcheck] Using selfcheckgpt SelfCheckNLI (DeBERTa-v3-MNLI)")
        except ImportError:
            self._impl = "fallback"
            print("[selfcheck] selfcheckgpt not installed — using token-overlap fallback")

    def score(
        self,
        sentences: list[str],
        samples: list[str],
    ) -> list[float]:
        self._load()
        if not sentences or not samples:
            return [0.0] * len(sentences)
        if self._impl == "selfcheckgpt_nli" and self._nli is not None:
            try:
                # SelfCheckNLI returns one score per sentence in [0, 1]
                # where higher = more inconsistent (per the package's
                # contradiction-probability convention).
                arr = self._nli.predict(
                    sentences=sentences, sampled_passages=samples,
                )
                return [float(s) for s in arr]
            except Exception as exc:
                print(f"[selfcheck] NLI failed, falling back: {exc}")
                # fall through to fallback
        return [fallback_inconsistency(s, samples) for s in sentences]


_backend: _NLIBackend | None = None


def get_backend() -> _NLIBackend:
    global _backend
    if _backend is None:
        _backend = _NLIBackend()
    return _backend


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class SentenceVerdict:
    sentence: str
    score: float          # 0..1; higher = more inconsistent
    flagged: bool


@dataclass
class SelfCheckReport:
    applied: bool
    revised_response: str
    samples_used: int = 0
    sentences_checked: int = 0
    flagged_count: int = 0
    threshold: float = 0.5
    backend: str = "fallback"
    verdicts: list[SentenceVerdict] = field(default_factory=list)
    skipped_reason: str | None = None


# ── Service ────────────────────────────────────────────────────────────────


class SelfCheckService:
    """Sample → score → annotate. Constructed with an inference service
    + settings; samples reuse the existing OpenAI-compatible client."""

    def __init__(self, inference: Any, settings: Settings):
        self.inference = inference
        self.settings = settings

    async def maybe_check(
        self,
        response: str,
        messages_for_resample: list[dict[str, Any]],
        rag_chunks: list[dict[str, Any]] | None,
        mode: str = "instant",
    ) -> SelfCheckReport:
        """Entry point. ``messages_for_resample`` is the prompt history
        that produced ``response`` — we re-call inference with it at
        higher temperature to get the consistency samples.

        Never raises. Failures degrade to ``applied=False`` with a
        descriptive ``skipped_reason``."""
        applicable, reason = should_selfcheck(response, rag_chunks, self.settings)
        if not applicable:
            return SelfCheckReport(
                applied=False, revised_response=response,
                threshold=self.settings.selfcheck_threshold,
                skipped_reason=reason,
            )

        sentences = _factual_sentences(response)
        if not sentences:
            return SelfCheckReport(
                applied=False, revised_response=response,
                threshold=self.settings.selfcheck_threshold,
                skipped_reason="no_factual_sentences",
            )

        try:
            samples = await self._sample(messages_for_resample, mode=mode)
        except Exception as exc:
            return SelfCheckReport(
                applied=False, revised_response=response,
                skipped_reason=f"sampling_failed: {type(exc).__name__}",
            )
        if not samples:
            return SelfCheckReport(
                applied=False, revised_response=response,
                skipped_reason="no_samples_returned",
            )

        backend = get_backend()
        scores = backend.score(sentences=sentences, samples=samples)
        threshold = self.settings.selfcheck_threshold
        verdicts = [
            SentenceVerdict(sentence=s, score=score, flagged=score >= threshold)
            for s, score in zip(sentences, scores, strict=False)
        ]
        revised = self._annotate(response, verdicts)
        flagged = sum(1 for v in verdicts if v.flagged)
        return SelfCheckReport(
            applied=True, revised_response=revised,
            samples_used=len(samples),
            sentences_checked=len(verdicts),
            flagged_count=flagged,
            threshold=threshold,
            backend=backend._impl or "fallback",
            verdicts=verdicts,
        )

    async def _sample(
        self,
        messages: list[dict[str, Any]],
        mode: str,
    ) -> list[str]:
        n = max(1, self.settings.selfcheck_samples)
        sem = asyncio.Semaphore(self.settings.selfcheck_max_concurrent)

        async def one():
            async with sem:
                try:
                    res = await self.inference.generate_response(
                        messages=messages,
                        mode=mode,
                        enable_thinking=False,
                        temperature=self.settings.selfcheck_temperature,
                    )
                    return (res.get("content") or "").strip()
                except Exception as exc:
                    print(f"[selfcheck] sample failed: {type(exc).__name__}: {exc}")
                    return ""

        results = await asyncio.gather(*[one() for _ in range(n)])
        return [r for r in results if r]

    def _annotate(
        self,
        response: str,
        verdicts: list[SentenceVerdict],
    ) -> str:
        """Reuse verification.annotate_response by adapting SentenceVerdict
        into Verdict shape. Idempotent — won't double-mark a sentence
        already flagged by CoVe or the Judge."""
        adapted: list[Verdict] = [
            Verdict(
                claim=Claim(text=v.sentence, sentence=v.sentence),
                supported=not v.flagged,
                reason=f"selfcheck score {v.score:.2f}",
            )
            for v in verdicts
        ]
        return annotate_response(response, adapted)


_selfcheck: SelfCheckService | None = None


def get_selfcheck_service(inference: Any, settings: Settings) -> SelfCheckService:
    global _selfcheck
    if _selfcheck is None:
        _selfcheck = SelfCheckService(inference, settings)
    return _selfcheck
