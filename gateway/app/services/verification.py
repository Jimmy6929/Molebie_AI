"""
Chain-of-Verification (CoVe) post-processor — Phase 3 task 3.1.

Decomposes a long-form factual response into atomic claims, verifies
each claim against the cited chunk in a *separate* (factored)
inference context, and flags unsupported claims inline.

Why factored, not joint:
    Factored CoVe runs each verifier call in a fresh context with only
    the single claim + the relevant chunk — no original generation
    context bleeds in. This consistently outperforms joint CoVe because
    (a) the verification question is simpler than the full task, and
    (b) the verifier can't anchor on whatever the generator already
    committed to.

Why annotate-don't-regenerate:
    Regenerating "fixed" sections on a 4B/9B model often introduces
    fresh fabrications. We annotate unsupported claims inline with
    ``[?]`` and extend the existing footnote instead. Cheaper, safer.

Default off — same calculus as self-consistency: 5–9× extra inference
calls per applicable response. Flip on once the eval framework shows
the precision gain is worth the latency.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings


# ── Regex (mirror chat.py — keep behaviour consistent) ─────────────────────

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
# Capitalised token (proper noun-ish) — used by the rule-based decompose
# fallback. Two-word minimum so we don't pick up sentence-initial caps.
_PROPER_NOUN_PAIR_RE = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+")

# CoVe's own marker for unsupported claims. Distinct from [S#] so the
# citation validator won't try to parse it as a citation index.
UNSUPPORTED_MARKER = " [?]"


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class Claim:
    text: str               # the claim itself, verbatim from the response
    sentence: str           # the surrounding sentence (for inline annotation)
    cited_indices: list[int] = field(default_factory=list)   # parsed [S#]


@dataclass
class Verdict:
    claim: Claim
    supported: bool
    reason: str
    chunk_index: int | None = None    # which chunk we routed to


@dataclass
class VerificationReport:
    applied: bool
    revised_response: str
    claims_checked: int = 0
    unsupported_count: int = 0
    verdicts: list[Verdict] = field(default_factory=list)
    skipped_reason: str | None = None
    decompose_fallback: bool = False    # rule-based fallback fired
    verify_json_failures: int = 0        # how many verifier calls returned bad JSON


# ── Pure-function helpers (testable without inference) ─────────────────────


def _has_specific_claim(sentence: str) -> bool:
    """True when the sentence contains a number / URL / date / proper-noun
    pair — the kinds of facts a small model is most likely to fabricate."""
    return bool(
        _NUMBER_RE.search(sentence)
        or _URL_RE.search(sentence)
        or _DATE_RE.search(sentence)
        or _PROPER_NOUN_PAIR_RE.search(sentence)
    )


def should_verify(
    response: str,
    query: str,
    rag_chunks: list[dict[str, Any]] | None,
    settings: Settings,
) -> tuple[bool, str]:
    """Trigger heuristic. Returns ``(applicable, reason_if_not)``."""
    if not settings.cove_enabled:
        return False, "cove_disabled"
    if not rag_chunks:
        return False, "no_rag_chunks"
    if len(response) < settings.cove_min_response_chars:
        return False, "response_too_short"
    # Re-use the verifiable-query classifier from self-consistency. CoVe
    # only buys you something on factual responses; on chatty turns it
    # wastes inference on hedges and transitions.
    from app.services.consistency import is_verifiable_query
    if not is_verifiable_query(query):
        return False, "query_not_verifiable"
    return True, ""


def parse_decompose_json(raw: str, max_claims: int) -> list[str] | None:
    """Parse the LLM's decompose response. Returns None on failure so the
    caller can fall back. Tolerant of leading/trailing whitespace and a
    common small-model habit of wrapping JSON in code fences."""
    if not raw:
        return None
    # Strip ```json fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    claims = obj.get("claims") if isinstance(obj, dict) else None
    if not isinstance(claims, list):
        return None
    cleaned_claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
    return cleaned_claims[:max_claims]


def rule_based_decompose(response: str, max_claims: int) -> list[str]:
    """Fallback decomposition when the LLM call fails or returns bad JSON.
    Splits into sentences, keeps those carrying specific factual content."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(response) if s.strip()]
    facts = [s for s in sentences if _has_specific_claim(s)]
    return facts[:max_claims]


def attach_sentences(claims: list[str], response: str) -> list[Claim]:
    """For each decomposed claim, find the originating sentence in the
    response and capture its ``[S#]`` citation indices. We use sentence-
    level routing so the inline ``[?]`` marker lands on the whole
    statement, not mid-claim.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(response) if s.strip()]
    result: list[Claim] = []
    for claim_text in claims:
        sentence = _find_owning_sentence(claim_text, sentences) or claim_text
        cited = [int(m) for m in _CITATION_RE.findall(sentence)]
        result.append(Claim(text=claim_text, sentence=sentence, cited_indices=cited))
    return result


def _find_owning_sentence(claim: str, sentences: list[str]) -> str | None:
    """Locate the sentence that contains this claim. Tries verbatim
    substring first, then falls back to highest-token-overlap match.
    Returns None if nothing plausible — caller treats the claim itself
    as the sentence."""
    if not sentences:
        return None
    if claim in sentences:
        return claim
    for s in sentences:
        if claim in s:
            return s
    # Fuzzy: highest token overlap
    claim_tokens = {t.lower() for t in _TOKEN_RE.findall(claim)}
    if not claim_tokens:
        return None
    best_overlap = 0
    best_sentence: str | None = None
    for s in sentences:
        s_tokens = {t.lower() for t in _TOKEN_RE.findall(s)}
        overlap = len(claim_tokens & s_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = s
    # Demand at least 2 token matches; anything less is noise
    return best_sentence if best_overlap >= 2 else None


def route_claim_to_chunk(
    claim: Claim,
    rag_chunks: list[dict[str, Any]],
) -> tuple[int | None, dict[str, Any] | None]:
    """Pick the chunk to verify this claim against.

    Returns ``(chunk_index_1_based, chunk_dict)``.

    Strategy:
      1. If the originating sentence has a ``[S#]`` citation, use chunk #S.
      2. Otherwise pick the chunk with highest token overlap with the
         claim text. Demand ≥1 overlapping content token; below that we
         return None and the caller marks the claim unsupported with
         "no relevant context found" — no LLM call.
    """
    if not rag_chunks:
        return None, None

    if claim.cited_indices:
        # 1-based to chunk-list-0-based; first cited that exists wins.
        for idx in claim.cited_indices:
            zero = idx - 1
            if 0 <= zero < len(rag_chunks):
                return idx, rag_chunks[zero]

    claim_tokens = {t.lower() for t in _TOKEN_RE.findall(claim.text)}
    if not claim_tokens:
        return None, None

    best_idx = -1
    best_overlap = 0
    for i, chunk in enumerate(rag_chunks):
        chunk_tokens = {
            t.lower() for t in _TOKEN_RE.findall(chunk.get("content", ""))
        }
        overlap = len(claim_tokens & chunk_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i
    if best_idx < 0 or best_overlap < 1:
        return None, None
    return best_idx + 1, rag_chunks[best_idx]


def parse_verify_json(raw: str) -> dict[str, Any] | None:
    """Parse a verifier response. Returns None on failure; caller treats
    None as conservative-fail (supported=false)."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "supported" not in obj:
        return None
    if not isinstance(obj["supported"], bool):
        return None
    return obj


def annotate_response(
    response: str,
    verdicts: list[Verdict],
) -> str:
    """Append ``[?]`` to each sentence flagged unsupported, before the
    closing punctuation/citation. Idempotent — won't double-annotate.

    We do a sentence-substring replace instead of a sentence-by-sentence
    rebuild so original whitespace, line breaks, and code fences in the
    response are preserved.
    """
    out = response
    seen: set[str] = set()
    for v in verdicts:
        if v.supported:
            continue
        sentence = v.claim.sentence
        if sentence in seen:
            continue
        seen.add(sentence)
        if UNSUPPORTED_MARKER.strip() in sentence:
            continue          # already annotated
        annotated = _annotate_sentence(sentence)
        out = out.replace(sentence, annotated, 1)
    return out


def _annotate_sentence(sentence: str) -> str:
    """Insert the ``[?]`` marker just before the trailing punctuation /
    citation, so it reads naturally inline."""
    # Match optional [S#] tags + trailing punctuation at the end
    m = re.search(r"(\s*(?:\[S\d+\])*\s*[.!?]?)$", sentence)
    if not m or not m.group(1):
        return sentence + UNSUPPORTED_MARKER
    insert_at = m.start()
    return sentence[:insert_at] + UNSUPPORTED_MARKER + sentence[insert_at:]


# ── Service ────────────────────────────────────────────────────────────────


class ChainOfVerification:
    """Decompose → verify (per claim, factored) → annotate.

    Constructed with an inference service and settings; one instance is
    fine to reuse across requests since the methods are stateless aside
    from the prompt cache.
    """

    def __init__(self, inference: Any, settings: Settings):
        self.inference = inference
        self.settings = settings
        self._decompose_prompt: str | None = None
        self._verify_prompt: str | None = None

    def _load_prompts(self) -> tuple[str, str]:
        if self._decompose_prompt is None or self._verify_prompt is None:
            base = Path(__file__).resolve().parent.parent.parent / self.settings.prompt_dir
            self._decompose_prompt = (base / "cove_decompose.txt").read_text()
            self._verify_prompt = (base / "cove_verify.txt").read_text()
        return self._decompose_prompt, self._verify_prompt

    async def maybe_verify(
        self,
        response: str,
        query: str,
        rag_chunks: list[dict[str, Any]] | None,
        mode: str = "instant",
    ) -> VerificationReport:
        """Entry point. Returns a VerificationReport — `applied=False`
        means the trigger heuristic short-circuited and the response is
        unchanged. Never raises; failures degrade to applied=False with a
        descriptive `skipped_reason`."""
        applicable, reason = should_verify(response, query, rag_chunks, self.settings)
        if not applicable:
            return VerificationReport(
                applied=False, revised_response=response, skipped_reason=reason,
            )

        try:
            claims, fallback = await self._decompose(response, mode=mode)
        except Exception as exc:
            return VerificationReport(
                applied=False, revised_response=response,
                skipped_reason=f"decompose_failed: {type(exc).__name__}",
            )
        if not claims:
            return VerificationReport(
                applied=False, revised_response=response,
                skipped_reason="no_claims_extracted",
                decompose_fallback=fallback,
            )

        verdicts, json_failures = await self._verify_claims(
            claims, rag_chunks or [], mode=mode,
        )
        revised = annotate_response(response, verdicts)
        unsupported = sum(1 for v in verdicts if not v.supported)
        return VerificationReport(
            applied=True,
            revised_response=revised,
            claims_checked=len(verdicts),
            unsupported_count=unsupported,
            verdicts=verdicts,
            decompose_fallback=fallback,
            verify_json_failures=json_failures,
        )

    async def _decompose(
        self, response: str, mode: str,
    ) -> tuple[list[Claim], bool]:
        """Returns (claims, fell_back_to_rules)."""
        decompose_template, _ = self._load_prompts()
        prompt = decompose_template.format(
            response=response, max_claims=self.settings.cove_max_claims,
        )
        result = await self.inference.generate_response(
            messages=[{"role": "user", "content": prompt}],
            mode=mode,
            enable_thinking=False,             # decompose is a JSON formatting task
            temperature=self.settings.cove_verifier_temperature,
        )
        raw = (result.get("content") or "").strip()
        parsed = parse_decompose_json(raw, self.settings.cove_max_claims)
        if parsed:
            return attach_sentences(parsed, response), False
        # Fallback: rule-based. We still try to return *something* —
        # silent skip would mean a long response slips by uninspected.
        rule_claims = rule_based_decompose(response, self.settings.cove_max_claims)
        return attach_sentences(rule_claims, response), True

    async def _verify_claims(
        self,
        claims: list[Claim],
        rag_chunks: list[dict[str, Any]],
        mode: str,
    ) -> tuple[list[Verdict], int]:
        sem = asyncio.Semaphore(self.settings.cove_verifier_max_concurrent)
        json_failures = 0

        async def check(claim: Claim) -> Verdict:
            nonlocal json_failures
            async with sem:
                return await self._verify_one(claim, rag_chunks)

        # No-context claims short-circuit before consuming a verifier call.
        results: list[Verdict] = []
        tasks: list[asyncio.Task] = []
        for c in claims:
            chunk_idx, chunk = route_claim_to_chunk(c, rag_chunks)
            if chunk is None:
                results.append(Verdict(
                    claim=c, supported=False,
                    reason="no relevant context found",
                    chunk_index=None,
                ))
            else:
                tasks.append(asyncio.create_task(check(c)))

        # Run the LLM-backed checks
        verified = await asyncio.gather(*tasks, return_exceptions=True)
        for v in verified:
            if isinstance(v, Exception):
                # Conservative on inference failures — don't claim support.
                # This is a degraded path; we log and continue.
                print(f"[cove] Verifier call raised {type(v).__name__}: {v}")
                continue
            if v.reason == "__json_fail__":
                json_failures += 1
                v.reason = "verifier returned malformed JSON; conservatively unsupported"
            results.append(v)
        return results, json_failures

    async def _verify_one(
        self,
        claim: Claim,
        rag_chunks: list[dict[str, Any]],
    ) -> Verdict:
        chunk_idx, chunk = route_claim_to_chunk(claim, rag_chunks)
        if chunk is None:
            return Verdict(
                claim=claim, supported=False,
                reason="no relevant context found", chunk_index=None,
            )
        _, verify_template = self._load_prompts()
        prompt = verify_template.format(
            claim=claim.text, context=chunk.get("content", ""),
        )
        result = await self.inference.generate_response(
            messages=[{"role": "user", "content": prompt}],
            mode="instant",                    # verifier is always instant
            enable_thinking=False,
            temperature=self.settings.cove_verifier_temperature,
        )
        raw = (result.get("content") or "").strip()
        parsed = parse_verify_json(raw)
        if parsed is None:
            # Conservative-fail with a sentinel reason; the caller
            # converts the sentinel into a user-readable reason and
            # increments the json-failures counter.
            return Verdict(
                claim=claim, supported=False,
                reason="__json_fail__", chunk_index=chunk_idx,
            )
        return Verdict(
            claim=claim,
            supported=bool(parsed["supported"]),
            reason=str(parsed.get("reason", "")).strip()[:240],
            chunk_index=chunk_idx,
        )


_cove: ChainOfVerification | None = None


def get_chain_of_verification(inference: Any, settings: Settings) -> ChainOfVerification:
    """Cached singleton — prompt files load once."""
    global _cove
    if _cove is None:
        _cove = ChainOfVerification(inference, settings)
    return _cove
