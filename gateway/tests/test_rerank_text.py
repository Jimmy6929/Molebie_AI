"""What the reranker scores: the chunk as the embedder indexed it.

Pure function — no model loads.
"""

from __future__ import annotations

from app.services.reranker import _rerank_text


def test_prefixes_the_heading_path():
    chunk = {
        "content": "- Chunk size: 512 tokens",
        "filename": "Molebie RAG.md",
        "metadata": {"breadcrumb": ["RAG configuration", "Chunking"]},
    }
    assert _rerank_text(chunk) == "RAG configuration > Chunking\n\n- Chunk size: 512 tokens"


def test_note_title_is_not_used():
    """A generic title ("Software") must not lift unrelated questions."""
    chunk = {"content": "body", "filename": "Software.md", "metadata": {"breadcrumb": []}}
    assert _rerank_text(chunk) == "body"


def test_bare_body_when_metadata_is_missing_or_unparsed():
    # FTS rows whose metadata failed to parse stay a string; never crash.
    assert _rerank_text({"content": "body", "metadata": "not json"}) == "body"
    assert _rerank_text({"content": "body", "metadata": {"breadcrumb": None}}) == "body"
    assert _rerank_text({"content": "body"}) == "body"
