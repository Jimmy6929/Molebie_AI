"""Tests for the pre-inference performance slice (T6b).

The eval + T6a instrumentation showed the pre-inference pipeline costs
more than the model on easy questions: sync embed/rerank froze the event
loop (rerank measured 4.9s on the live vault), memory retrieval sat
serially on the wall path, and the reranker scored all 30 fused
candidates. These tests pin the safety properties of the fixes.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from app.config import get_settings
from app.services.embedding import EmbeddingService
from app.services.memory import MemoryService


def test_new_perf_defaults():
    assert get_settings().memory_retrieval_timeout == 5.0


@pytest.mark.asyncio
async def test_embed_async_serializes_concurrent_callers():
    """Memory retrieval now runs concurrently with RAG retrieval; both
    embed queries. The shared service-level lock must keep model access
    strictly one-at-a-time (the event loop's old implicit serialization
    is gone)."""
    svc = EmbeddingService(get_settings())
    active = {"now": 0, "max": 0}
    guard = threading.Lock()

    def fake_embed(text, prefix="search_query"):
        with guard:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)
        with guard:
            active["now"] -= 1
        return [0.0]

    svc.embed = fake_embed
    await asyncio.gather(
        svc.embed_async("raw user message"),
        svc.embed_async("rewritten rag query"),
        svc.embed_async("third caller"),
    )
    assert active["max"] == 1


@pytest.mark.asyncio
async def test_memory_timeout_resolves_from_settings():
    """timeout=None must resolve from memory_retrieval_timeout — a hanging
    embed is abandoned within the configured budget and memory degrades
    to [] instead of blocking the turn."""

    class _S:
        memory_enabled = True
        memory_extract_interval = 10
        memory_max_facts_per_extraction = 5
        memory_dedup_threshold = 0.9
        memory_retrieval_threshold = 0.5
        memory_retrieval_top_k = 5
        memory_llm_mode = "instant"
        memory_extract_max_tokens = 256
        memory_retrieval_timeout = 0.05

    class _HangingEmbedder:
        async def embed_async(self, text, prefix="search_query"):
            await asyncio.sleep(10)

    svc = MemoryService(_S(), db=None)
    t0 = time.monotonic()
    with patch("app.services.embedding.get_embedding_service",
               return_value=_HangingEmbedder()):
        out = await svc.retrieve_relevant_memories("u1", "query")
    assert out == []
    assert time.monotonic() - t0 < 2.0
