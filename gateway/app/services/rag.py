"""
RAG (Retrieval-Augmented Generation) service.

Embeds the user query, searches document_chunks via pgvector similarity,
and formats matching chunks as context for injection into the LLM prompt.
"""

from typing import Any, Dict, List, Optional

import httpx

from app.config import Settings, get_settings
from app.services.embedding import EmbeddingService, get_embedding_service

# Minimal stop words for query condensation (fallback expansion)
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "it", "its",
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their",
})


def _quality_label(similarity: float) -> str:
    """Return a quality label based on similarity score."""
    if similarity >= 0.75:
        return "HIGH MATCH"
    if similarity >= 0.6:
        return "MODERATE MATCH"
    return "WEAK MATCH"


def _condense_query(query: str) -> str:
    """Remove stop words to produce a keyword-focused query for fallback search."""
    words = query.split()
    condensed = [w for w in words if w.lower().strip("?!.,;:'\"") not in _STOP_WORDS]
    return " ".join(condensed) if condensed else query


class RAGService:
    """Retrieve relevant document context for a user query."""

    def __init__(self, settings: Settings, embedding_service: EmbeddingService):
        self.settings = settings
        self.embedding = embedding_service
        self.enabled = settings.rag_enabled
        self.match_count = settings.rag_match_count
        self.match_threshold = settings.rag_match_threshold
        self.max_context_chars = settings.rag_max_context_chars

    async def user_has_documents(self, user_token: str) -> bool:
        """
        Check if the user has any document chunks (completed RAG documents).
        Skips embedding load when user has no documents.
        """
        if not self.enabled:
            return False
        url = f"{self.settings.supabase_url}/rest/v1/document_chunks?select=id&limit=1"
        headers = {
            "apikey": self.settings.supabase_anon_key,
            "Authorization": f"Bearer {user_token}",
            "Prefer": "count=none",
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                rows = resp.json()
                return len(rows) > 0
        except Exception as exc:
            print(f"[rag] user_has_documents check failed: {type(exc).__name__}: {exc}")
            return False

    async def _search_chunks(
        self,
        user_token: str,
        query_embedding: List[float],
        count: int,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Run a single similarity search against document_chunks."""
        rpc_url = f"{self.settings.supabase_url}/rest/v1/rpc/match_documents_with_metadata"
        headers = {
            "apikey": self.settings.supabase_anon_key,
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": count,
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(rpc_url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            print(f"[rag] RPC HTTP error {exc.response.status_code}: {exc.response.text[:500]}")
            return []
        except Exception as exc:
            print(f"[rag] Similarity search failed: {type(exc).__name__}: {exc}")
            return []

    async def retrieve_context(
        self,
        user_token: str,
        query: str,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed the query and search for matching document chunks.

        Uses the Supabase RPC match_documents_with_metadata which
        enforces RLS via the user's JWT.

        When primary search yields weak results (top similarity < 0.6),
        runs a fallback search with a condensed (stop-word-removed) query
        and merges results.

        Returns list of dicts with keys:
            chunk_id, document_id, filename, content, chunk_index, similarity
        """
        if not self.enabled:
            return []

        count = limit or self.match_count
        thresh = threshold or self.match_threshold

        print(f"[rag] Embedding query ({len(query)} chars)...")
        query_embedding = self.embedding.embed(query)

        results = await self._search_chunks(user_token, query_embedding, count, thresh)

        if not results:
            print(f"[rag] No matching chunks (threshold={thresh}). Try lowering RAG_MATCH_THRESHOLD.")
            return []

        top_similarity = results[0].get("similarity", 0)
        print(f"[rag] Found {len(results)} matching chunks (top similarity: {top_similarity:.3f})")

        # Fallback: if best match is weak, try a condensed query
        if top_similarity < 0.6:
            condensed = _condense_query(query)
            if condensed != query:
                print(f"[rag] Top match weak ({top_similarity:.3f}), trying condensed query...")
                condensed_embedding = self.embedding.embed(condensed)
                fallback_results = await self._search_chunks(
                    user_token, condensed_embedding, count, thresh,
                )
                if fallback_results:
                    # Merge and deduplicate by chunk_id, keep best similarity
                    seen_chunks: dict[str, Dict[str, Any]] = {}
                    for chunk in results + fallback_results:
                        cid = chunk.get("chunk_id", chunk.get("id", id(chunk)))
                        existing = seen_chunks.get(cid)
                        if existing is None or chunk.get("similarity", 0) > existing.get("similarity", 0):
                            seen_chunks[cid] = chunk
                    results = sorted(
                        seen_chunks.values(),
                        key=lambda c: c.get("similarity", 0),
                        reverse=True,
                    )[:count]
                    print(f"[rag] After merge: {len(results)} chunks (top: {results[0].get('similarity', 0):.3f})")

        return results

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a text block for the system message.

        Each chunk is labelled with a quality tier (HIGH / MODERATE / WEAK)
        so the model can calibrate trust accordingly.
        """
        if not chunks:
            return ""

        lines = []
        total_chars = 0
        for i, c in enumerate(chunks, 1):
            filename = c.get("filename", "unknown")
            content = c.get("content", "")
            similarity = c.get("similarity", 0)
            label = _quality_label(similarity)

            entry = f"[{i}] {filename} ({label}, relevance: {similarity:.2f})\n{content}"
            if total_chars + len(entry) > self.max_context_chars:
                break
            lines.append(entry)
            total_chars += len(entry)

        return "\n\n".join(lines)


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get cached RAGService instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(get_settings(), get_embedding_service())
    return _rag_service
