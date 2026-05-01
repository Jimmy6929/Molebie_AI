"""
Grounding judge — Phase 3 task 3.2.

Reuses Qwen3-Reranker-0.6B (already loaded as part of RAG) to score
``(claim, cited_chunk)`` pairs. Below threshold → flag. No extra LLM
calls; just one reranker forward pass per claim.

Why this is worth its own service:

  CoVe (3.1) gives precision via per-claim LLM verification — but at
  ~5–9× the latency of vanilla RAG. The judge is ~1/10 the cost of
  CoVe and catches the easy cases (off-topic fabrications, claims that
  cite a chunk about something else). It runs first as a gate: when
  the judge's flagged-count is below the threshold the chat route
  skips CoVe entirely, saving a stack of inference calls.

What it does NOT catch:

  Numeric-substitution attacks within an otherwise-relevant chunk
  ("Mount Everest is 1234 metres" against a chunk that says "Mount
  Everest is 8849 metres"). The reranker scores semantic relevance,
  not literal grounding. CoVe handles those with the LLM-backed
  verifier — that's the layered defense.

Default off — flip ``judge_enabled=True`` once you've calibrated the
threshold against your reranker score distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.services.verification import (
    Claim,
    UNSUPPORTED_MARKER,
    Verdict,
    _SENTENCE_SPLIT_RE,
    annotate_response,
    attach_sentences,
    rule_based_decompose,
    route_claim_to_chunk,
)


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class JudgeVerdict:
    claim: Claim
    score: float                  # 0..1 from the reranker yes-head
    supported: bool               # score >= threshold
    chunk_index: int | None       # 1-based index into rag_chunks (or None)


@dataclass
class JudgeReport:
    applied: bool
    revised_response: str
    scored_count: int = 0
    flagged_count: int = 0
    threshold: float = 0.5
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    skipped_reason: str | None = None


# ── Pure-function helpers ──────────────────────────────────────────────────


def has_citations(response: str) -> bool:
    """True if the response contains at least one ``[S#]`` marker. Cheap
    pre-trigger filter — the judge can't say anything useful when there
    are no source references."""
    return "[S" in response and any(
        ch.isdigit() for ch in response.split("[S", 1)[1][:5]
    )


def should_judge(
    response: str,
    rag_chunks: list[dict[str, Any]] | None,
    settings: Settings,
) -> tuple[bool, str]:
    """Trigger heuristic. Returns ``(applicable, reason_if_not)``.

    The judge is *cheap*, so the trigger is generous on purpose — we
    let it fire on any cited-RAG response over the min length. Operators
    tune precision via ``judge_threshold``."""
    if not settings.judge_enabled:
        return False, "judge_disabled"
    if not rag_chunks:
        return False, "no_rag_chunks"
    if len(response) < settings.judge_min_response_chars:
        return False, "response_too_short"
    if not has_citations(response):
        return False, "no_citations"
    return True, ""


def claims_for_judge(
    response: str,
    rag_chunks: list[dict[str, Any]],
    max_claims: int,
) -> list[Claim]:
    """Extract claims for judging. Uses the same rule-based decomposer as
    CoVe's fallback (``rule_based_decompose``) to keep the two services
    consistent — when both fire they reason about the same set of claims.
    LLM-based decomposition is reserved for CoVe; the judge stays cheap."""
    candidates = rule_based_decompose(response, max_claims)
    return attach_sentences(candidates, response)


# ── Service ────────────────────────────────────────────────────────────────


class GroundingJudge:
    """Reranker-backed grounding gate. Constructed with a RerankerService
    instance so it shares the loaded model with the RAG pipeline."""

    def __init__(self, reranker: Any, settings: Settings):
        self.reranker = reranker
        self.settings = settings

    def judge(
        self,
        response: str,
        rag_chunks: list[dict[str, Any]] | None,
    ) -> JudgeReport:
        """Synchronous — reranker forward passes are CPU/MPS-bound and
        already used in the RAG hot path that way. Wrapping in asyncio
        would add no parallelism (single model instance)."""
        applicable, reason = should_judge(response, rag_chunks, self.settings)
        if not applicable:
            return JudgeReport(
                applied=False, revised_response=response,
                threshold=self.settings.judge_threshold,
                skipped_reason=reason,
            )

        chunks = rag_chunks or []
        max_claims = max(1, getattr(self.settings, "cove_max_claims", 8))
        claims = claims_for_judge(response, chunks, max_claims)
        if not claims:
            return JudgeReport(
                applied=False, revised_response=response,
                threshold=self.settings.judge_threshold,
                skipped_reason="no_claims_extracted",
            )

        verdicts = self._score_claims(claims, chunks)
        revised = self._annotate(response, verdicts)
        flagged = sum(1 for v in verdicts if not v.supported)
        return JudgeReport(
            applied=True,
            revised_response=revised,
            scored_count=len(verdicts),
            flagged_count=flagged,
            threshold=self.settings.judge_threshold,
            verdicts=verdicts,
        )

    def _score_claims(
        self,
        claims: list[Claim],
        rag_chunks: list[dict[str, Any]],
    ) -> list[JudgeVerdict]:
        threshold = self.settings.judge_threshold
        out: list[JudgeVerdict] = []
        for claim in claims:
            chunk_idx, chunk = route_claim_to_chunk(claim, rag_chunks)
            if chunk is None:
                # No chunk to compare against → conservatively flag with
                # score 0. Same convention CoVe uses; keeps the two
                # services aligned on what "no_context" means.
                out.append(JudgeVerdict(
                    claim=claim, score=0.0, supported=False, chunk_index=None,
                ))
                continue
            score = self._score_pair(claim.text, chunk.get("content", ""))
            out.append(JudgeVerdict(
                claim=claim, score=score,
                supported=score >= threshold, chunk_index=chunk_idx,
            ))
        return out

    def _score_pair(self, claim_text: str, chunk_text: str) -> float:
        """One reranker forward pass. The RerankerService picks the right
        backend — Qwen-causal yes/no head or sentence-transformers
        CrossEncoder — based on the configured model name. Both produce
        0..1 scores."""
        # Touch _load_model once to make sure the backend is initialised.
        # On the hot path this is a no-op after the first call.
        self.reranker._load_model()
        if self.reranker._impl == "qwen_causal":
            return self.reranker._qwen_score(claim_text, chunk_text)
        # CrossEncoder backend
        scores = self.reranker._model.predict([(claim_text, chunk_text)])
        return float(scores[0])

    def _annotate(
        self,
        response: str,
        verdicts: list[JudgeVerdict],
    ) -> str:
        """Reuse verification.annotate_response by adapting JudgeVerdicts
        into Verdict shape. Idempotent — won't double-mark a sentence
        already flagged by CoVe."""
        adapted: list[Verdict] = [
            Verdict(claim=v.claim, supported=v.supported,
                    reason=f"judge score {v.score:.2f}",
                    chunk_index=v.chunk_index)
            for v in verdicts
        ]
        return annotate_response(response, adapted)


_judge: GroundingJudge | None = None


def get_grounding_judge(reranker: Any, settings: Settings) -> GroundingJudge:
    """Cached singleton — reranker is already cached upstream."""
    global _judge
    if _judge is None:
        _judge = GroundingJudge(reranker, settings)
    return _judge
