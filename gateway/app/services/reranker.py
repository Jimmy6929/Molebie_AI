"""
Cross-encoder reranker for RAG pipeline.

Takes the top-N candidates from hybrid search and re-scores each
(query, chunk) pair using a cross-encoder model. This is more accurate
than bi-encoder similarity because the model sees query and document
together.

Model: cross-encoder/ms-marco-MiniLM-L6-v2 (~80MB, runs locally on CPU).
Adds ~200-500ms for 20 candidates but improves precision by 20-35%.
"""

from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings


class RerankerService:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(self, settings: Settings):
        self.model_name = settings.rag_reranker_model
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        print(f"[reranker] Loading model: {self.model_name}")
        self._model = CrossEncoder(self.model_name)
        print(f"[reranker] Model loaded")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Re-score and re-rank chunks using the cross-encoder.

        Args:
            query: The user's search query
            chunks: List of chunk dicts from hybrid/vector search
            top_k: Number of top results to return

        Returns:
            Top-k chunks sorted by rerank score, with rerank_score added
        """
        if not chunks:
            return []

        self._load_model()

        # Build (query, chunk_content) pairs for scoring
        pairs = [(query, c.get("content", "")) for c in chunks]
        scores = self._model.predict(pairs)

        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)
        return ranked[:top_k]


_reranker: Optional[RerankerService] = None


def get_reranker() -> RerankerService:
    """Get cached RerankerService instance."""
    global _reranker
    if _reranker is None:
        _reranker = RerankerService(get_settings())
    return _reranker
