"""
RAG evaluation service for measuring retrieval quality.

Tracks per-query metrics (scores, timing, source distribution) and
provides an evaluation endpoint for running test cases against the
full RAG pipeline.
"""

from typing import Any

from app.config import Settings, get_settings
from app.services.rag import get_rag_service


async def evaluate_queries(
    test_cases: list[dict[str, Any]],
    user_id: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run test cases through the RAG pipeline and compute metrics.

    Each test case: {"query": str, "expected_doc_ids": List[str]}

    Returns:
        {
            "total": int,
            "hit_rate": float,  # fraction of queries where expected doc appeared in results
            "mrr": float,       # mean reciprocal rank
            "results": [per-query details]
        }
    """
    settings = settings or get_settings()
    rag = get_rag_service()

    results = []
    hits = 0
    reciprocal_ranks = []

    for case in test_cases:
        query = case["query"]
        expected_ids = set(case.get("expected_doc_ids", []))

        chunks = await rag.retrieve_context(user_id, query)
        metrics = rag.get_metrics(chunks)

        # Check if any expected document was found
        found_doc_ids = [c.get("document_id", "") for c in chunks]
        hit = bool(expected_ids & set(found_doc_ids))
        if hit:
            hits += 1

        # Compute reciprocal rank (position of first expected doc)
        rr = 0.0
        for rank, doc_id in enumerate(found_doc_ids, 1):
            if doc_id in expected_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        results.append({
            "query": query,
            "expected_doc_ids": list(expected_ids),
            "found_doc_ids": found_doc_ids,
            "hit": hit,
            "reciprocal_rank": rr,
            "num_chunks": len(chunks),
            "metrics": metrics,
        })

    total = len(test_cases) or 1
    return {
        "total": len(test_cases),
        "hit_rate": hits / total,
        "mrr": sum(reciprocal_ranks) / total,
        "results": results,
    }
