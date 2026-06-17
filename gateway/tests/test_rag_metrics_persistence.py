"""Tests for RAG metrics persistence wiring (`_persist_rag_metrics`).

`insert_rag_metrics` existed but was never called, so `rag_query_metrics` stayed
empty. The helper persists retrieval metrics after each chat turn, best-effort
(a metrics-write failure must never break the chat).
"""

from app.routes.chat import _persist_rag_metrics


class _FakeRag:
    def __init__(self, metrics):
        self._m = metrics

    def get_metrics(self, chunks):
        return self._m


class _FakeDB:
    def __init__(self):
        self.calls = []

    async def insert_rag_metrics(self, user_id, metrics):
        self.calls.append((user_id, metrics))


async def test_persists_metrics_with_query_text():
    db = _FakeDB()
    rag = _FakeRag({"num_candidates": 5, "top_similarity": 0.8})
    await _persist_rag_metrics(db, rag, "u1", "what do you know about me?", [{"x": 1}])
    assert len(db.calls) == 1
    uid, m = db.calls[0]
    assert uid == "u1"
    assert m["query_text"] == "what do you know about me?"
    assert m["num_candidates"] == 5 and m["top_similarity"] == 0.8


async def test_noop_on_empty_chunks():
    db = _FakeDB()
    rag = _FakeRag({"num_candidates": 5})
    await _persist_rag_metrics(db, rag, "u1", "q", [])
    assert db.calls == []


async def test_noop_when_metrics_none():
    # get_metrics returns None when rag_metrics_enabled is off.
    db = _FakeDB()
    rag = _FakeRag(None)
    await _persist_rag_metrics(db, rag, "u1", "q", [{"x": 1}])
    assert db.calls == []


async def test_swallows_db_errors():
    class _BoomDB:
        async def insert_rag_metrics(self, *a):
            raise RuntimeError("boom")

    rag = _FakeRag({"num_candidates": 1})
    # Must not raise — a metrics-write failure cannot break the chat.
    await _persist_rag_metrics(_BoomDB(), rag, "u1", "q", [{"x": 1}])
