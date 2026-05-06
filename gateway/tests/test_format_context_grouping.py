"""Tests for parent-child neighbor expansion in RAGService.format_context.

Phase 4 of the retrieval-quality sprint. Consecutive chunks from the
same document_id are merged into a single [S#] block (small-to-big
retrieval). Char-budget pressure first drops neighbors (is_parent=False),
then drops entire groups when even parents don't fit. Citation numbering
stays consecutive — no gaps when a group is dropped.
"""

import pytest

from app.services.rag import (
    RAGService,
    _reorder_groups_for_context,
)


def _make_service(max_chars: int = 12000) -> RAGService:
    """Construct a RAGService skeleton without touching the DB or settings."""
    svc = RAGService.__new__(RAGService)
    svc.max_context_chars = max_chars
    return svc


def _chunk(doc, idx, content, *, parent=True, score=0.5, heading=None):
    return {
        "document_id": doc,
        "chunk_index": idx,
        "content": content,
        "filename": f"{doc}.md",
        "metadata": {"heading": heading} if heading else {},
        "is_parent": parent,
        "parent_score": score,
        "rerank_score": score if parent else None,
    }


def test_consecutive_same_doc_chunks_merge_under_one_citation():
    svc = _make_service()
    chunks = [
        _chunk("A", 4, "intro paragraph", parent=False, score=0.85, heading="Intro"),
        _chunk("A", 5, "the answer here", parent=True, score=0.85, heading="Intro"),
        _chunk("A", 6, "follow-up", parent=False, score=0.85, heading="Intro"),
        _chunk("B", 2, "different doc", parent=True, score=0.62),
    ]
    out = svc.format_context(chunks)
    # Two citations, not four
    assert out.count("[S") == 2
    assert "[S1]" in out and "[S2]" in out
    # Doc A's three chunks merged into the same block, joined by \n
    assert "intro paragraph\nthe answer here\nfollow-up" in out
    # Doc A's heading carried from the first chunk in the group
    assert "section: Intro" in out
    # Doc B is its own [S2] block
    assert "different doc" in out


def test_existing_single_chunk_behaviour_unchanged():
    """When chunks have unique document_ids (no neighbor expansion), the
    output should match the legacy one-citation-per-chunk format."""
    svc = _make_service()
    chunks = [
        _chunk("A", 0, "alpha", parent=True, score=0.9),
        _chunk("B", 0, "beta", parent=True, score=0.7),
        _chunk("C", 0, "gamma", parent=True, score=0.5),
    ]
    out = svc.format_context(chunks)
    assert out.count("[S") == 3
    assert "[S1]" in out and "[S2]" in out and "[S3]" in out
    assert "alpha" in out and "beta" in out and "gamma" in out


def test_oversized_group_falls_back_to_parents_only():
    svc = _make_service(max_chars=200)
    big = "x" * 100
    chunks = [
        # Group A: 3 chunks × 100 chars + header ~340 chars > 200, but
        # parent-only (1 × 100 chars + header ~30) ~130 < 200
        _chunk("A", 4, big, parent=False, score=0.85),
        _chunk("A", 5, big, parent=True, score=0.85),
        _chunk("A", 6, big, parent=False, score=0.85),
    ]
    out = svc.format_context(chunks)
    assert "[S1]" in out
    # Only the parent body should remain; neighbors dropped
    assert out.count(big) == 1


def test_skipped_oversized_group_does_not_leave_citation_gap():
    """If a group can't fit even with parent-only, it's skipped — but
    later groups that do fit must take the next [S#], not the gapped
    one. Otherwise the model sees [S1], [S3] and gets confused."""
    svc = _make_service(max_chars=120)
    huge = "y" * 200  # too big for any budget
    small = "z" * 30
    chunks = [
        _chunk("A", 0, huge, parent=True, score=0.9),       # won't fit
        _chunk("B", 0, small, parent=True, score=0.6),      # fits
    ]
    out = svc.format_context(chunks)
    # [S1] is reused for B, not gapped
    assert "[S1]" in out
    assert "[S2]" not in out
    assert small in out
    assert huge not in out


def test_within_group_chunks_sorted_by_chunk_index():
    """Even if caller passes group members out of order, the merged
    body must read in document order (idx ascending)."""
    svc = _make_service()
    chunks = [
        _chunk("A", 6, "third", parent=False, score=0.85),
        _chunk("A", 4, "first", parent=False, score=0.85),
        _chunk("A", 5, "second", parent=True, score=0.85),
    ]
    out = svc.format_context(chunks)
    # All three appear in document order
    pos_first = out.index("first")
    pos_second = out.index("second")
    pos_third = out.index("third")
    assert pos_first < pos_second < pos_third


def test_reorder_groups_u_shape_by_parent_score():
    """Highest-scored group lands at index 0 (start of context),
    next-highest at index -1 (end), lowest in the middle. Within each
    group, chunks stay in chunk_index order."""
    groups = [
        {"document_id": f"D{i}", "parent_score": s,
         "chunks": [_chunk(f"D{i}", 0, "x", parent=True, score=s)]}
        for i, s in enumerate([0.3, 0.5, 0.7, 0.9, 0.4])
    ]
    flat = _reorder_groups_for_context(groups)
    # First and last positions should be the two highest-scored groups
    scores = [c["parent_score"] for c in flat]
    assert scores[0] == 0.9   # highest at start
    assert scores[-1] == 0.7  # second-highest at end


def test_reorder_preserves_chunk_index_order_within_group():
    groups = [
        {"document_id": "A", "parent_score": 0.9, "chunks": [
            _chunk("A", 4, "a4", parent=False, score=0.9),
            _chunk("A", 5, "a5", parent=True, score=0.9),
            _chunk("A", 6, "a6", parent=False, score=0.9),
        ]},
        {"document_id": "B", "parent_score": 0.5, "chunks": [
            _chunk("B", 2, "b2", parent=True, score=0.5),
        ]},
        {"document_id": "C", "parent_score": 0.7, "chunks": [
            _chunk("C", 1, "c1", parent=True, score=0.7),
        ]},
    ]
    flat = _reorder_groups_for_context(groups)
    # A's three chunks must be consecutive and in idx order
    a_positions = [i for i, c in enumerate(flat) if c["document_id"] == "A"]
    a_indices = [flat[i]["chunk_index"] for i in a_positions]
    assert a_positions == list(range(a_positions[0], a_positions[0] + 3))
    assert a_indices == [4, 5, 6]


def test_empty_input_returns_empty():
    svc = _make_service()
    assert svc.format_context([]) == ""
    assert _reorder_groups_for_context([]) == []
