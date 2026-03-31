"""
Database service using SQLite with sqlite-vec and FTS5.

Replaces the Supabase REST client. All methods are async and enforce
user isolation via WHERE user_id = ? in every query.
"""

import json
import struct
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional

import aiosqlite

from app.config import Settings, get_settings
from app.schema import get_connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


def _row_to_dict(row: aiosqlite.Row) -> Dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


def _serialize_embedding(embedding: List[float]) -> bytes:
    """Pack a float list into bytes for sqlite-vec."""
    return struct.pack(f"{len(embedding)}f", *embedding)


class DatabaseService:
    """Async SQLite database service for all CRUD and search operations."""

    def __init__(self, settings: Settings):
        self.data_dir = getattr(settings, "data_dir", "data")
        self._conn: Optional[aiosqlite.Connection] = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await get_connection(self.data_dir)
        return self._conn

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ==================== Sessions ====================

    async def create_session(
        self, user_id: str, title: str = "New Chat", **_kwargs
    ) -> Optional[Dict[str, Any]]:
        db = await self._get_conn()
        now = _now()
        sid = _uuid()
        await db.execute(
            "INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, user_id, title, now, now),
        )
        await db.commit()
        row = await db.execute_fetchall(
            "SELECT * FROM chat_sessions WHERE id = ?", (sid,)
        )
        return _row_to_dict(row[0]) if row else None

    async def get_session(
        self, session_id: str, user_id: str, **_kwargs
    ) -> Optional[Dict[str, Any]]:
        db = await self._get_conn()
        rows = await db.execute_fetchall(
            "SELECT * FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        return _row_to_dict(rows[0]) if rows else None

    async def list_sessions(
        self, user_id: str, limit: int = 50, **_kwargs
    ) -> List[Dict[str, Any]]:
        db = await self._get_conn()
        rows = await db.execute_fetchall(
            "SELECT * FROM chat_sessions WHERE user_id = ? AND is_archived = 0 "
            "ORDER BY is_pinned DESC, updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        return [_row_to_dict(r) for r in rows]

    async def pin_session(
        self, session_id: str, user_id: str, is_pinned: bool, **_kwargs
    ) -> bool:
        db = await self._get_conn()
        await db.execute(
            "UPDATE chat_sessions SET is_pinned = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (int(is_pinned), _now(), session_id, user_id),
        )
        await db.commit()
        return True

    async def update_session_title(
        self, session_id: str, user_id: str, title: str, **_kwargs
    ) -> bool:
        db = await self._get_conn()
        await db.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (title, _now(), session_id, user_id),
        )
        await db.commit()
        return True

    async def delete_session(
        self, session_id: str, user_id: str, **_kwargs
    ) -> bool:
        db = await self._get_conn()
        await db.execute(
            "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        await db.commit()
        return True

    async def update_session_summary(
        self,
        session_id: str,
        user_id: str,
        summary: str,
        summary_message_count: int,
        **_kwargs,
    ) -> None:
        db = await self._get_conn()
        await db.execute(
            "UPDATE chat_sessions SET summary = ?, summary_message_count = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (summary, summary_message_count, _now(), session_id, user_id),
        )
        await db.commit()

    # ==================== Messages ====================

    async def create_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        mode_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
        reasoning_content: Optional[str] = None,
        **_kwargs,
    ) -> Optional[Dict[str, Any]]:
        db = await self._get_conn()
        now = _now()
        mid = _uuid()
        await db.execute(
            "INSERT INTO chat_messages "
            "(id, session_id, user_id, role, content, mode_used, tokens_used, reasoning_content, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mid, session_id, user_id, role, content, mode_used, tokens_used, reasoning_content, now),
        )
        # Update session updated_at
        await db.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        await db.commit()
        row = await db.execute_fetchall(
            "SELECT * FROM chat_messages WHERE id = ?", (mid,)
        )
        return _row_to_dict(row[0]) if row else None

    async def get_session_messages(
        self,
        session_id: str,
        user_id: str,
        limit: int = 100,
        **_kwargs,
    ) -> List[Dict[str, Any]]:
        db = await self._get_conn()
        rows = await db.execute_fetchall(
            "SELECT * FROM chat_messages WHERE session_id = ? AND user_id = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (session_id, user_id, limit),
        )
        return [_row_to_dict(r) for r in rows]

    # ==================== Message Images ====================

    async def create_message_image(
        self,
        message_id: str,
        user_id: str,
        storage_path: str,
        filename: Optional[str] = None,
        mime_type: str = "image/jpeg",
        file_size: int = 0,
        **_kwargs,
    ) -> Optional[Dict[str, Any]]:
        db = await self._get_conn()
        iid = _uuid()
        now = _now()
        await db.execute(
            "INSERT INTO message_images "
            "(id, message_id, user_id, storage_path, filename, mime_type, file_size, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (iid, message_id, user_id, storage_path, filename, mime_type, file_size, now),
        )
        await db.commit()
        row = await db.execute_fetchall(
            "SELECT * FROM message_images WHERE id = ?", (iid,)
        )
        return _row_to_dict(row[0]) if row else None

    async def get_message_images(
        self,
        message_ids: List[str],
        user_id: str,
        **_kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        if not message_ids:
            return {}
        db = await self._get_conn()
        placeholders = ",".join("?" * len(message_ids))
        rows = await db.execute_fetchall(
            f"SELECT id, message_id, storage_path, filename, mime_type, file_size "
            f"FROM message_images WHERE message_id IN ({placeholders}) AND user_id = ?",
            (*message_ids, user_id),
        )
        return {r["message_id"]: _row_to_dict(r) for r in rows}

    # ==================== Message Sources (Web Search) ====================

    async def create_message_sources(
        self,
        message_id: str,
        sources: List[Dict[str, str]],
    ) -> None:
        """Batch-insert web search source links for a message."""
        if not sources:
            return
        db = await self._get_conn()
        now = _now()
        for src in sources:
            await db.execute(
                "INSERT INTO message_sources (id, message_id, url, title, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (_uuid(), message_id, src.get("url", ""), src.get("title", ""), now),
            )
        await db.commit()

    async def get_message_sources(
        self,
        message_ids: List[str],
    ) -> Dict[str, List[Dict[str, str]]]:
        """Fetch source links for multiple messages. Returns {message_id: [{title, url}, ...]}."""
        if not message_ids:
            return {}
        db = await self._get_conn()
        placeholders = ",".join("?" * len(message_ids))
        rows = await db.execute_fetchall(
            f"SELECT message_id, url, title FROM message_sources "
            f"WHERE message_id IN ({placeholders}) ORDER BY created_at ASC",
            tuple(message_ids),
        )
        result: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            mid = r["message_id"]
            if mid not in result:
                result[mid] = []
            result[mid].append({"title": r["title"], "url": r["url"]})
        return result

    async def get_image_metadata(
        self, image_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        db = await self._get_conn()
        rows = await db.execute_fetchall(
            "SELECT storage_path, mime_type FROM message_images "
            "WHERE id = ? AND user_id = ?",
            (image_id, user_id),
        )
        return _row_to_dict(rows[0]) if rows else None

    # ==================== User Profile ====================

    async def get_or_create_profile(
        self, user_id: str, email: Optional[str] = None, **_kwargs
    ) -> Optional[Dict[str, Any]]:
        db = await self._get_conn()
        rows = await db.execute_fetchall(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        )
        if rows:
            return _row_to_dict(rows[0])
        if email:
            now = _now()
            await db.execute(
                "INSERT OR IGNORE INTO users (id, email, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, email, now, now),
            )
            await db.commit()
            rows = await db.execute_fetchall(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            )
            return _row_to_dict(rows[0]) if rows else None
        return None

    # ==================== Documents ====================

    async def insert_document(
        self,
        user_id: str,
        filename: str,
        storage_path: str,
        file_type: str,
        file_size: int,
        doc_status: str = "pending",
    ) -> Dict[str, Any]:
        db = await self._get_conn()
        did = _uuid()
        now = _now()
        await db.execute(
            "INSERT INTO documents "
            "(id, user_id, filename, storage_path, file_type, file_size, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (did, user_id, filename, storage_path, file_type, file_size, doc_status, now),
        )
        await db.commit()
        rows = await db.execute_fetchall("SELECT * FROM documents WHERE id = ?", (did,))
        return _row_to_dict(rows[0])

    async def update_document_status(
        self, doc_id: str, doc_status: str, processed_at: Optional[str] = None
    ) -> None:
        db = await self._get_conn()
        if processed_at:
            await db.execute(
                "UPDATE documents SET status = ?, processed_at = ? WHERE id = ?",
                (doc_status, processed_at, doc_id),
            )
        else:
            await db.execute(
                "UPDATE documents SET status = ? WHERE id = ?",
                (doc_status, doc_id),
            )
        await db.commit()

    async def get_document(
        self, doc_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        db = await self._get_conn()
        rows = await db.execute_fetchall(
            "SELECT * FROM documents WHERE id = ? AND user_id = ?",
            (doc_id, user_id),
        )
        return _row_to_dict(rows[0]) if rows else None

    async def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        db = await self._get_conn()
        rows = await db.execute_fetchall(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        return [_row_to_dict(r) for r in rows]

    async def delete_document(self, doc_id: str, user_id: str) -> None:
        db = await self._get_conn()
        # Delete chunks first (and their vec/fts entries via triggers)
        chunk_rows = await db.execute_fetchall(
            "SELECT rowid FROM document_chunks WHERE document_id = ? AND user_id = ?",
            (doc_id, user_id),
        )
        if chunk_rows:
            rowids = [r["rowid"] for r in chunk_rows]
            placeholders = ",".join("?" * len(rowids))
            await db.execute(
                f"DELETE FROM document_chunks_vec WHERE rowid IN ({placeholders})",
                rowids,
            )
        await db.execute(
            "DELETE FROM document_chunks WHERE document_id = ? AND user_id = ?",
            (doc_id, user_id),
        )
        await db.execute(
            "DELETE FROM documents WHERE id = ? AND user_id = ?",
            (doc_id, user_id),
        )
        await db.commit()

    # ==================== Document Chunks ====================

    async def insert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Batch insert chunks into document_chunks, FTS5, and vec tables."""
        if not chunks:
            return
        db = await self._get_conn()
        for chunk in chunks:
            cid = _uuid()
            now = _now()
            metadata_json = json.dumps(chunk.get("metadata")) if chunk.get("metadata") else None
            await db.execute(
                "INSERT INTO document_chunks "
                "(id, document_id, user_id, content, content_contextualized, chunk_index, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    chunk["document_id"],
                    chunk["user_id"],
                    chunk["content"],
                    chunk.get("content_contextualized"),
                    chunk.get("chunk_index", 0),
                    metadata_json,
                    now,
                ),
            )
            # Get the rowid for vec insertion
            cursor = await db.execute("SELECT last_insert_rowid()")
            row = await cursor.fetchone()
            rowid = row[0]

            # Insert embedding into vec table
            embedding = chunk.get("embedding")
            if embedding:
                blob = _serialize_embedding(embedding)
                await db.execute(
                    "INSERT INTO document_chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (rowid, blob),
                )
        await db.commit()

    async def user_has_document_chunks(self, user_id: str) -> bool:
        db = await self._get_conn()
        rows = await db.execute_fetchall(
            "SELECT 1 FROM document_chunks WHERE user_id = ? LIMIT 1",
            (user_id,),
        )
        return len(rows) > 0

    # ==================== Session Documents ====================

    async def insert_session_document(
        self,
        session_id: str,
        user_id: str,
        filename: str,
        content: str,
        file_size: int,
    ) -> Dict[str, Any]:
        db = await self._get_conn()
        sid = _uuid()
        now = _now()
        await db.execute(
            "INSERT INTO session_documents "
            "(id, session_id, user_id, filename, content, file_size, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sid, session_id, user_id, filename, content, file_size, now),
        )
        await db.commit()
        rows = await db.execute_fetchall(
            "SELECT * FROM session_documents WHERE id = ?", (sid,)
        )
        return _row_to_dict(rows[0])

    async def list_session_documents(
        self, session_id: str, user_id: str
    ) -> List[Dict[str, Any]]:
        db = await self._get_conn()
        rows = await db.execute_fetchall(
            "SELECT * FROM session_documents "
            "WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC",
            (session_id, user_id),
        )
        return [_row_to_dict(r) for r in rows]

    async def delete_session_document(
        self, attachment_id: str, session_id: str, user_id: str
    ) -> None:
        db = await self._get_conn()
        await db.execute(
            "DELETE FROM session_documents WHERE id = ? AND session_id = ? AND user_id = ?",
            (attachment_id, session_id, user_id),
        )
        await db.commit()

    async def fetch_session_attachments_text(
        self, session_id: str, user_id: str
    ) -> str:
        """Fetch session documents formatted for system prompt injection."""
        rows = await self.list_session_documents(session_id, user_id)
        if not rows:
            return ""
        parts = ["ATTACHED DOCUMENTS (read by user request — use this as primary context):"]
        for idx, row in enumerate(rows, 1):
            content = row.get("content", "")
            filename = row.get("filename", "document")
            parts.append(f"\n[{idx}] {filename} ({len(content):,} chars)")
            parts.append(content)
        return "\n".join(parts)

    # ==================== Vector Search ====================

    async def vector_search_chunks(
        self,
        user_id: str,
        query_embedding: List[float],
        threshold: float = 0.3,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search document chunks by vector similarity using sqlite-vec."""
        db = await self._get_conn()
        blob = _serialize_embedding(query_embedding)
        rows = await db.execute_fetchall(
            """
            SELECT
                dc.id AS chunk_id,
                dc.document_id,
                d.filename,
                dc.content,
                dc.chunk_index,
                dc.metadata,
                v.distance
            FROM document_chunks_vec AS v
            JOIN document_chunks AS dc ON dc.rowid = v.rowid
            JOIN documents AS d ON d.id = dc.document_id
            WHERE v.embedding MATCH ?
              AND k = ?
              AND dc.user_id = ?
              AND d.status = 'completed'
            ORDER BY v.distance
            """,
            (blob, limit, user_id),
        )
        results = []
        for r in rows:
            d = _row_to_dict(r)
            similarity = 1.0 - d.pop("distance", 0.0)
            if similarity < threshold:
                continue
            d["similarity"] = similarity
            if d.get("metadata") and isinstance(d["metadata"], str):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        return results

    async def fts_search_chunks(
        self,
        user_id: str,
        query_text: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search document chunks by full-text (BM25) using FTS5."""
        db = await self._get_conn()
        # Escape FTS5 special chars in query
        safe_query = query_text.replace('"', '""')
        rows = await db.execute_fetchall(
            """
            SELECT
                dc.id AS chunk_id,
                dc.document_id,
                d.filename,
                dc.content,
                dc.chunk_index,
                dc.metadata,
                fts.rank AS text_rank
            FROM document_chunks_fts AS fts
            JOIN document_chunks AS dc ON dc.rowid = fts.rowid
            JOIN documents AS d ON d.id = dc.document_id
            WHERE document_chunks_fts MATCH ?
              AND dc.user_id = ?
              AND d.status = 'completed'
            ORDER BY fts.rank
            LIMIT ?
            """,
            (safe_query, user_id, limit),
        )
        results = []
        for r in rows:
            d = _row_to_dict(r)
            # FTS5 rank is negative (lower = better), negate for consistency
            d["text_rank"] = -d.get("text_rank", 0.0)
            if d.get("metadata") and isinstance(d["metadata"], str):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        return results

    # ==================== User Memories ====================

    async def store_memory(
        self,
        user_id: str,
        content: str,
        category: str,
        embedding: List[float],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = await self._get_conn()
        mid = _uuid()
        now = _now()
        await db.execute(
            "INSERT INTO user_memories "
            "(id, user_id, content, category, source_session_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mid, user_id, content, category, session_id, now, now),
        )
        # Get the rowid for vec insertion
        cursor = await db.execute("SELECT last_insert_rowid()")
        row = await cursor.fetchone()
        rowid = row[0]

        blob = _serialize_embedding(embedding)
        await db.execute(
            "INSERT INTO user_memories_vec (rowid, embedding) VALUES (?, ?)",
            (rowid, blob),
        )
        await db.commit()
        rows = await db.execute_fetchall(
            "SELECT * FROM user_memories WHERE id = ?", (mid,)
        )
        return _row_to_dict(rows[0])

    async def update_memory(
        self,
        memory_id: str,
        content: str,
        embedding: List[float],
    ) -> None:
        db = await self._get_conn()
        now = _now()
        await db.execute(
            "UPDATE user_memories SET content = ?, updated_at = ? WHERE id = ?",
            (content, now, memory_id),
        )
        # Update vec table: get rowid from user_memories
        rows = await db.execute_fetchall(
            "SELECT rowid FROM user_memories WHERE id = ?", (memory_id,)
        )
        if rows:
            rowid = rows[0]["rowid"]
            blob = _serialize_embedding(embedding)
            await db.execute(
                "UPDATE user_memories_vec SET embedding = ? WHERE rowid = ?",
                (blob, rowid),
            )
        await db.commit()

    async def vector_search_memories(
        self,
        user_id: str,
        query_embedding: List[float],
        threshold: float = 0.5,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search user memories by vector similarity."""
        db = await self._get_conn()
        blob = _serialize_embedding(query_embedding)
        rows = await db.execute_fetchall(
            """
            SELECT
                um.id AS memory_id,
                um.content,
                um.category,
                um.created_at,
                v.distance
            FROM user_memories_vec AS v
            JOIN user_memories AS um ON um.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
              AND um.user_id = ?
            ORDER BY v.distance
            """,
            (blob, limit, user_id),
        )
        results = []
        for r in rows:
            d = _row_to_dict(r)
            similarity = 1.0 - d.pop("distance", 0.0)
            if similarity < threshold:
                continue
            d["similarity"] = similarity
            results.append(d)

        # Bump access stats for returned memories
        if results:
            now = _now()
            for m in results:
                await db.execute(
                    "UPDATE user_memories SET access_count = access_count + 1, "
                    "last_accessed_at = ? WHERE id = ?",
                    (now, m["memory_id"]),
                )
            await db.commit()

        return results

    # ==================== RAG Metrics ====================

    async def insert_rag_metrics(
        self, user_id: str, metrics: Dict[str, Any]
    ) -> None:
        db = await self._get_conn()
        mid = _uuid()
        now = _now()
        await db.execute(
            "INSERT INTO rag_query_metrics "
            "(id, user_id, query_text, num_candidates, unique_documents, "
            "top_similarity, avg_similarity, top_rrf_score, top_rerank_score, "
            "score_spread, hybrid_enabled, reranker_enabled, "
            "t_embed_ms, t_search_ms, t_rerank_ms, t_total_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mid,
                user_id,
                metrics.get("query_text"),
                metrics.get("num_candidates"),
                metrics.get("unique_documents"),
                metrics.get("top_similarity"),
                metrics.get("avg_similarity"),
                metrics.get("top_rrf_score"),
                metrics.get("top_rerank_score"),
                metrics.get("score_spread"),
                int(metrics.get("hybrid_enabled", False)),
                int(metrics.get("reranker_enabled", False)),
                metrics.get("t_embed_ms"),
                metrics.get("t_search_ms"),
                metrics.get("t_rerank_ms"),
                metrics.get("t_total_ms"),
                now,
            ),
        )
        await db.commit()


_db_service: Optional[DatabaseService] = None


@lru_cache
def get_database_service() -> DatabaseService:
    """Get cached database service instance."""
    return DatabaseService(get_settings())
