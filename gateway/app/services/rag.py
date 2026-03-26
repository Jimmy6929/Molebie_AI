"""
RAG (Retrieval-Augmented Generation) service.

Embeds the user query, searches document_chunks via hybrid search
(SQLite vector + FTS with RRF), optionally reranks with a cross-encoder,
and formats matching chunks as context for injection into the LLM prompt.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.services.database import DatabaseService, get_database_service
from app.services.embedding import EmbeddingService, get_embedding_service


def _quality_label(similarity: float) -> str:
    """Return a quality label based on similarity score."""
    if similarity >= 0.75:
        return "HIGH MATCH"
    if similarity >= 0.6:
        return "MODERATE MATCH"
    return "WEAK MATCH"


def _rrf_fuse(
    vector_results: List[Dict[str, Any]],
    text_results: List[Dict[str, Any]],
    k: int = 60,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion of vector and full-text search results."""
    vec_ranked = {r["chunk_id"]: (i + 1, r) for i, r in enumerate(vector_results)}
    txt_ranked = {r["chunk_id"]: (i + 1, r) for i, r in enumerate(text_results)}
    all_ids = set(vec_ranked) | set(txt_ranked)

    fused: List[Dict[str, Any]] = []
    for chunk_id in all_ids:
        vec_entry = vec_ranked.get(chunk_id)
        txt_entry = txt_ranked.get(chunk_id)
        rrf_score = 0.0
        if vec_entry:
            rrf_score += vector_weight * (1.0 / (k + vec_entry[0]))
        if txt_entry:
            rrf_score += text_weight * (1.0 / (k + txt_entry[0]))
        row = (vec_entry[1] if vec_entry else txt_entry[1]).copy()
        row["rrf_score"] = rrf_score
        if vec_entry:
            row["similarity"] = vec_entry[1].get("similarity", 0)
        if txt_entry:
            row["text_rank"] = txt_entry[1].get("text_rank", 0)
        fused.append(row)

    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused


class RAGService:
    """Retrieve relevant document context for a user query."""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        db: DatabaseService,
    ):
        self.settings = settings
        self.embedding = embedding_service
        self.db = db
        self.enabled = settings.rag_enabled
        self.match_count = settings.rag_match_count
        self.match_threshold = settings.rag_match_threshold
        self.max_context_chars = settings.rag_max_context_chars
        self.hybrid_enabled = settings.rag_hybrid_enabled
        self.vector_weight = settings.rag_vector_weight
        self.text_weight = settings.rag_text_weight
        self.rrf_k = settings.rag_rrf_k

    async def user_has_documents(self, user_id: str) -> bool:
        """
        Check if the user has any document chunks (completed RAG documents).
        Skips embedding load when user has no documents.
        """
        if not self.enabled:
            return False
        try:
            return await self.db.user_has_document_chunks(user_id)
        except Exception as exc:
            print(f"[rag] user_has_documents check failed: {type(exc).__name__}: {exc}")
            return False

    async def _search_chunks(
        self,
        user_id: str,
        query_embedding: List[float],
        count: int,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Run a single vector similarity search against document_chunks."""
        try:
            return await self.db.vector_search_chunks(
                user_id, query_embedding, threshold, count
            )
        except Exception as exc:
            print(f"[rag] Similarity search failed: {type(exc).__name__}: {exc}")
            return []

    async def _hybrid_search(
        self,
        user_id: str,
        query_embedding: List[float],
        query_text: str,
        count: int,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Run hybrid search (vector + full-text) with RRF fusion."""
        try:
            vector_results, text_results = await asyncio.gather(
                self.db.vector_search_chunks(user_id, query_embedding, threshold, count),
                self.db.fts_search_chunks(user_id, query_text, count),
            )
            return _rrf_fuse(
                vector_results,
                text_results,
                k=self.rrf_k,
                vector_weight=self.vector_weight,
                text_weight=self.text_weight,
            )[:count]
        except Exception as exc:
            print(f"[rag] Hybrid search failed: {type(exc).__name__}: {exc}")
            print("[rag] Falling back to vector-only search")
            return await self._search_chunks(user_id, query_embedding, count, threshold)

    async def _rewrite_query(
        self,
        query: str,
        conversation_context: List[Dict[str, str]],
    ) -> str:
        """Rewrite the query using conversation context for better retrieval.

        Falls back to original query on timeout or error.
        """
        context_lines = []
        for msg in conversation_context[-6:]:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            content = content[:300]
            context_lines.append(f"{role}: {content}")
        context_text = "\n".join(context_lines)

        prompt = (
            "Given this conversation context and the latest user query, "
            "rewrite the query as a standalone search query that would "
            "retrieve the most relevant documents. Include key terms and "
            "context that might be implied by the conversation.\n\n"
            f"CONVERSATION:\n{context_text}\n\n"
            f"LATEST QUERY: {query}\n\n"
            "Rewritten search query (one line, no explanation):"
        )

        try:
            from app.services.inference import get_inference_service
            inference = get_inference_service()

            result = await asyncio.wait_for(
                inference.generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    mode=self.settings.rag_query_rewrite_llm_mode,
                    max_tokens=self.settings.rag_query_rewrite_max_tokens,
                    temperature=0.0,
                ),
                timeout=self.settings.rag_query_rewrite_timeout,
            )

            rewritten = result.get("content", "").strip()
            if "</think>" in rewritten:
                rewritten = rewritten.split("</think>", 1)[-1].strip()
            rewritten = rewritten.strip("\"'")

            if rewritten and len(rewritten) > 3:
                print(f"[rag] Query rewritten: '{query[:60]}' -> '{rewritten[:60]}'")
                return rewritten
            else:
                print("[rag] Rewrite produced empty/short result, using original")
                return query

        except asyncio.TimeoutError:
            print(f"[rag] Query rewrite timed out ({self.settings.rag_query_rewrite_timeout}s), using original")
            return query
        except Exception as exc:
            print(f"[rag] Query rewrite failed ({type(exc).__name__}: {exc}), using original")
            return query

    async def retrieve_context(
        self,
        user_id: str,
        query: str,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed the query and search for matching document chunks.

        Pipeline: (rewrite) -> embed -> hybrid/vector search -> rerank -> return
        When hybrid search is enabled, uses RRF fusion of vector + FTS.
        Otherwise falls back to pure vector search.

        Returns list of dicts with keys:
            chunk_id, document_id, filename, content, chunk_index, metadata,
            similarity, and optionally rrf_score, text_rank
        Also returns rag_timings dict for metrics.
        """
        if not self.enabled:
            return []

        count = limit or self.match_count
        thresh = threshold or self.match_threshold
        timings: Dict[str, float] = {}

        # Step 0: Query rewriting (contextual expansion)
        search_query = query
        if (self.settings.rag_query_rewrite_enabled
                and conversation_context
                and len(conversation_context) >= 2):
            t_rw = time.monotonic()
            search_query = await self._rewrite_query(query, conversation_context)
            timings["t_rewrite_ms"] = round((time.monotonic() - t_rw) * 1000, 1)
        else:
            timings["t_rewrite_ms"] = 0

        # Step 1: Embed query
        t0 = time.monotonic()
        print(f"[rag] Embedding query ({len(search_query)} chars)...")
        query_embedding = self.embedding.embed(search_query)
        timings["t_embed_ms"] = round((time.monotonic() - t0) * 1000, 1)

        # Step 2: Search
        t1 = time.monotonic()
        if self.hybrid_enabled:
            results = await self._hybrid_search(
                user_id, query_embedding, search_query, count, thresh
            )
        else:
            results = await self._search_chunks(
                user_id, query_embedding, count, thresh
            )
        timings["t_search_ms"] = round((time.monotonic() - t1) * 1000, 1)

        if not results:
            print(f"[rag] No matching chunks (threshold={thresh})")
            return []

        top_score = results[0].get("rrf_score") or results[0].get("similarity", 0)
        print(f"[rag] Found {len(results)} chunks (top score: {top_score:.4f})")

        # Step 3: Rerank (if enabled)
        t2 = time.monotonic()
        if self.settings.rag_reranker_enabled:
            try:
                from app.services.reranker import get_reranker
                reranker = get_reranker()
                results = reranker.rerank(
                    query, results, top_k=self.settings.rag_rerank_top_k
                )
                timings["t_rerank_ms"] = round((time.monotonic() - t2) * 1000, 1)
                print(f"[rag] Reranked to top-{len(results)} (took {timings['t_rerank_ms']:.0f}ms)")
            except Exception as exc:
                print(f"[rag] Reranking failed (using search order): {type(exc).__name__}: {exc}")
                timings["t_rerank_ms"] = 0
        else:
            timings["t_rerank_ms"] = 0

        timings["t_total_ms"] = round(
            timings.get("t_rewrite_ms", 0) + timings["t_embed_ms"] + timings["t_search_ms"] + timings["t_rerank_ms"], 1
        )

        # Attach timings to results for metrics
        if results:
            results[0]["_rag_timings"] = timings

        if self.settings.rag_metrics_log_console:
            print(
                f"[rag] Timings: embed={timings['t_embed_ms']:.0f}ms, "
                f"search={timings['t_search_ms']:.0f}ms, "
                f"rerank={timings['t_rerank_ms']:.0f}ms, "
                f"total={timings['t_total_ms']:.0f}ms"
            )

        return results

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a text block for the system message.

        Each chunk is labelled with a quality tier (HIGH / MODERATE / WEAK)
        so the model can calibrate trust accordingly. Headings from metadata
        are shown when available.
        """
        if not chunks:
            return ""

        lines = []
        total_chars = 0
        for i, c in enumerate(chunks, 1):
            filename = c.get("filename", "unknown")
            content = c.get("content", "")
            similarity = c.get("similarity", 0)
            rrf_score = c.get("rrf_score")
            label = _quality_label(similarity)

            # Extract heading from metadata
            metadata = c.get("metadata") or {}
            heading = metadata.get("heading")
            source = f"{filename} > {heading}" if heading else filename

            # Use rrf_score for display when available
            if rrf_score is not None:
                entry = f"[{i}] {source} ({label}, relevance: {similarity:.2f}, rrf: {rrf_score:.4f})\n{content}"
            else:
                entry = f"[{i}] {source} ({label}, relevance: {similarity:.2f})\n{content}"

            if total_chars + len(entry) > self.max_context_chars:
                break
            lines.append(entry)
            total_chars += len(entry)

        return "\n\n".join(lines)

    def get_metrics(self, chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract RAG metrics from retrieval results."""
        if not chunks or not self.settings.rag_metrics_enabled:
            return None

        timings = chunks[0].get("_rag_timings", {}) if chunks else {}
        similarities = [c.get("similarity", 0) for c in chunks]
        rrf_scores = [c.get("rrf_score", 0) for c in chunks if c.get("rrf_score") is not None]
        rerank_scores = [c.get("rerank_score", 0) for c in chunks if c.get("rerank_score") is not None]
        unique_docs = len(set(c.get("document_id", "") for c in chunks))

        return {
            "num_candidates": len(chunks),
            "unique_documents": unique_docs,
            "top_similarity": max(similarities) if similarities else 0,
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0,
            "top_rrf_score": max(rrf_scores) if rrf_scores else None,
            "top_rerank_score": max(rerank_scores) if rerank_scores else None,
            "score_spread": max(similarities) - min(similarities) if len(similarities) > 1 else 0,
            "hybrid_enabled": self.hybrid_enabled,
            "reranker_enabled": self.settings.rag_reranker_enabled,
            **timings,
        }


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get cached RAGService instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(get_settings(), get_embedding_service(), get_database_service())
    return _rag_service
