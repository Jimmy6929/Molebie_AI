"""
Unit tests for the retrieval core in app/services/rag.py — the pure functions
and the stub-able neighbor expansion that previously had no coverage.

Deliberately complements test_format_context_grouping.py (which covers
format_context grouping + char budget): nothing here duplicates it. No
embedding/reranker models load — chunks are hand-built dicts and the db is
a stub object passed as a parameter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag import (
    _chunk_score,
    _expand_with_neighbors,
    _quality_label,
    _reorder_for_context,
    _reorder_groups_for_context,
    _rrf_fuse,
    compute_retrieval_confidence,
)


def _chunk(
    doc: str = "d1",
    idx: int = 0,
    rerank: float | None = None,
    rrf: float | None = None,
    sim: float | None = None,
) -> dict[str, Any]:
    c: dict[str, Any] = {
        "chunk_id": f"{doc}-{idx}",
        "document_id": doc,
        "chunk_index": idx,
        "content": f"content of {doc} chunk {idx}",
    }
    if rerank is not None:
        c["rerank_score"] = rerank
    if rrf is not None:
        c["rrf_score"] = rrf
    if sim is not None:
        c["similarity"] = sim
    return c


# ───────── _rrf_fuse ─────────


def test_rrf_exact_rank_math():
    vec = [{"chunk_id": "a", "similarity": 0.9}, {"chunk_id": "b", "similarity": 0.8}]
    txt = [{"chunk_id": "b", "text_rank": 5.0}, {"chunk_id": "c", "text_rank": 4.0}]

    fused = {r["chunk_id"]: r for r in _rrf_fuse(vec, txt)}

    assert fused["a"]["rrf_score"] == pytest.approx(0.7 / 61)
    assert fused["b"]["rrf_score"] == pytest.approx(0.7 / 62 + 0.3 / 61)
    assert fused["c"]["rrf_score"] == pytest.approx(0.3 / 62)


def test_rrf_union_carries_fields_and_sorts_desc():
    vec = [{"chunk_id": "a", "similarity": 0.9}]
    txt = [{"chunk_id": "a", "text_rank": 7.0}, {"chunk_id": "z", "text_rank": 6.0}]

    fused = _rrf_fuse(vec, txt)

    assert [r["chunk_id"] for r in fused] == ["a", "z"]  # both-list member wins
    scores = [r["rrf_score"] for r in fused]
    assert scores == sorted(scores, reverse=True)
    a = fused[0]
    assert a["similarity"] == 0.9 and a["text_rank"] == 7.0


def test_rrf_honors_k_and_weight_parameters():
    vec = [{"chunk_id": "a", "similarity": 1.0}]
    txt = [{"chunk_id": "b", "text_rank": 1.0}]

    fused = {
        r["chunk_id"]: r["rrf_score"]
        for r in _rrf_fuse(vec, txt, k=10, vector_weight=0.5, text_weight=0.5)
    }
    assert fused["a"] == pytest.approx(0.5 / 11)
    assert fused["b"] == pytest.approx(0.5 / 11)


def test_rrf_does_not_mutate_inputs():
    vec = [{"chunk_id": "a", "similarity": 0.9}]
    _rrf_fuse(vec, [])
    assert "rrf_score" not in vec[0]


# ───────── _chunk_score ─────────


def test_chunk_score_precedence():
    assert _chunk_score(_chunk(rerank=0.9, rrf=0.5, sim=0.1)) == 0.9
    assert _chunk_score(_chunk(rrf=0.5, sim=0.1)) == 0.5
    assert _chunk_score(_chunk(sim=0.1)) == pytest.approx(0.1)
    assert _chunk_score(_chunk()) == 0.0
    # Explicit None must fall through, not coerce.
    c = _chunk(sim=0.2)
    c["rerank_score"] = None
    c["rrf_score"] = None
    assert _chunk_score(c) == pytest.approx(0.2)


# ───────── compute_retrieval_confidence ─────────


def test_confidence_empty_is_none():
    assert compute_retrieval_confidence([]) == "NONE"


@pytest.mark.parametrize(
    "top,expected",
    [
        (0.7, "HIGH"),
        (0.699, "MODERATE"),
        (0.3, "MODERATE"),
        (0.299, "LOW"),
        (0.0, "LOW"),
    ],
)
def test_confidence_exact_boundaries(top, expected):
    chunks = [_chunk(rerank=0.1), _chunk(rerank=top)]
    assert compute_retrieval_confidence(chunks) == expected


def test_confidence_driven_by_single_best_chunk_and_precedence():
    # One strong chunk among weak ones is enough for HIGH (no corroboration
    # gate — calibrated for Qwen3-Reranker's bimodal distribution).
    chunks = [_chunk(rerank=0.05), _chunk(rerank=0.04), _chunk(rerank=0.95)]
    assert compute_retrieval_confidence(chunks) == "HIGH"
    # rerank_score outranks a high similarity.
    assert compute_retrieval_confidence([_chunk(rerank=0.1, sim=0.99)]) == "LOW"


# ───────── _quality_label ─────────


@pytest.mark.parametrize(
    "sim,expected",
    [
        (0.75, "HIGH MATCH"),
        (0.749, "MODERATE MATCH"),
        (0.6, "MODERATE MATCH"),
        (0.599, "WEAK MATCH"),
    ],
)
def test_quality_label_boundaries(sim, expected):
    assert _quality_label(sim) == expected


# ───────── _reorder_for_context (U-shape) ─────────


def test_reorder_short_lists_unchanged():
    chunks = [_chunk(idx=0, rerank=0.1), _chunk(idx=1, rerank=0.9)]
    assert _reorder_for_context(chunks) == chunks


def test_reorder_u_shape_invariants():
    chunks = [_chunk(idx=i, rerank=s) for i, s in enumerate([0.1, 0.4, 0.2, 0.5, 0.3])]
    out = _reorder_for_context(chunks)

    assert len(out) == len(chunks)
    assert {c["chunk_id"] for c in out} == {c["chunk_id"] for c in chunks}
    scores = [_chunk_score(c) for c in out]
    # Top score first, runner-up last, minimum somewhere in the middle.
    assert scores[0] == max(scores)
    assert scores[-1] == sorted(scores)[-2]
    assert scores.index(min(scores)) not in (0, len(scores) - 1)
    # Exact arrangement for 5 ascending scores: [5th, 3rd, 1st, 2nd, 4th].
    assert scores == [0.5, 0.3, 0.1, 0.2, 0.4]


# ───────── _expand_with_neighbors ─────────


class _StubDB:
    """Stands in for DatabaseService: returns canned chunk ranges."""

    def __init__(
        self,
        ranges: dict[str, list[dict[str, Any]]] | None = None,
        error: bool = False,
    ) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self._ranges = ranges or {}
        self._error = error

    async def get_chunks_in_range(self, user_id, doc_id, lo, hi):
        self.calls.append((doc_id, lo, hi))
        if self._error:
            raise RuntimeError("boom")
        # Copies: the expander tags chunks in place.
        return [dict(c) for c in self._ranges.get(doc_id, [])]


@pytest.mark.asyncio
async def test_expand_empty_input():
    assert await _expand_with_neighbors([], _StubDB(), "u1") == []


@pytest.mark.asyncio
async def test_expand_merges_same_doc_parents_into_one_range():
    parents = [_chunk(doc="d1", idx=5, rerank=0.8), _chunk(doc="d1", idx=6, rerank=0.6)]
    db = _StubDB(ranges={"d1": [_chunk(doc="d1", idx=i) for i in range(4, 8)]})

    groups = await _expand_with_neighbors(parents, db, "u1")

    assert db.calls == [("d1", 4, 7)]  # one merged fetch, not two overlapping
    assert len(groups) == 1
    g = groups[0]
    assert g["parent_score"] == 0.8  # max of the doc's parents
    flags = {c["chunk_index"]: c["is_parent"] for c in g["chunks"]}
    assert flags == {4: False, 5: True, 6: True, 7: False}
    assert all(c["parent_score"] == 0.8 for c in g["chunks"])


@pytest.mark.asyncio
async def test_expand_clamps_lo_at_zero_and_carries_rerank():
    parents = [_chunk(doc="d1", idx=0, rerank=0.9)]
    db = _StubDB(ranges={"d1": [_chunk(doc="d1", idx=0), _chunk(doc="d1", idx=1)]})

    groups = await _expand_with_neighbors(parents, db, "u1")

    assert db.calls == [("d1", 0, 1)]  # lo clamped, no -1
    parent_row = next(c for c in groups[0]["chunks"] if c["chunk_index"] == 0)
    # Fetched rows have no rerank_score — it must be carried from the parent.
    assert parent_row["rerank_score"] == 0.9


@pytest.mark.asyncio
async def test_expand_falls_back_to_parents_when_fetch_empty_or_raises():
    parents = [_chunk(doc="d1", idx=3, rerank=0.7), _chunk(doc="d1", idx=1, rerank=0.5)]

    for db in (_StubDB(), _StubDB(error=True)):  # empty range / raising fetch
        groups = await _expand_with_neighbors(parents, db, "u1")
        assert len(groups) == 1
        idxs = [c["chunk_index"] for c in groups[0]["chunks"]]
        assert idxs == [1, 3]  # parents only, chunk_index order
        assert all(c["is_parent"] for c in groups[0]["chunks"])


@pytest.mark.asyncio
async def test_expand_skips_chunks_without_document_id():
    orphan = {"chunk_id": "x", "chunk_index": 0, "rerank_score": 0.9}
    groups = await _expand_with_neighbors([orphan], _StubDB(), "u1")
    assert groups == []


@pytest.mark.asyncio
async def test_expand_separate_docs_separate_groups():
    parents = [_chunk(doc="d1", idx=2, rerank=0.4), _chunk(doc="d2", idx=9, rerank=0.9)]
    db = _StubDB()

    groups = await _expand_with_neighbors(parents, db, "u1")

    assert {g["document_id"] for g in groups} == {"d1", "d2"}
    assert sorted(db.calls) == [("d1", 1, 3), ("d2", 8, 10)]


# ───────── _reorder_groups_for_context ─────────


def _group(doc: str, score: float, idxs: list[int]) -> dict[str, Any]:
    return {
        "document_id": doc,
        "parent_score": score,
        "chunks": [_chunk(doc=doc, idx=i) for i in idxs],
    }


def test_reorder_groups_under_three_keeps_order():
    groups = [_group("d1", 0.2, [0, 1]), _group("d2", 0.9, [5])]
    flat = _reorder_groups_for_context(groups)
    assert [c["chunk_id"] for c in flat] == ["d1-0", "d1-1", "d2-5"]
    assert _reorder_groups_for_context([]) == []


def test_reorder_groups_u_shape_keeps_chunks_contiguous():
    groups = [
        _group("low", 0.1, [0, 1]),
        _group("top", 0.9, [3, 4]),
        _group("mid", 0.5, [7]),
    ]
    flat = _reorder_groups_for_context(groups)

    docs_in_order = [c["document_id"] for c in flat]
    # U-shape over 3 groups: top first, runner-up (mid) last, lowest in the
    # middle — and each group's chunks stay contiguous.
    assert docs_in_order == ["top", "top", "low", "low", "mid"]
    # Intra-group chunk_index order preserved.
    assert [c["chunk_index"] for c in flat if c["document_id"] == "top"] == [3, 4]
