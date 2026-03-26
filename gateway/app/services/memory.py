"""
Structured memory service for cross-session user facts.

Extracts categorised facts from conversations using the LLM,
deduplicates against existing memories via cosine similarity,
and stores them with vector embeddings for retrieval. At query
time, relevant memories are injected into the system message.
"""

import asyncio
import json
import re
from typing import Optional, Dict, Any, List

import httpx

from app.config import Settings, get_settings

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_EXTRACT_PROMPT = """\
Extract important facts about the user from this conversation snippet.
Focus on information that would be useful in future conversations.

CONVERSATION:
{messages_text}

Extract facts in these categories:
- preference: User preferences, likes, dislikes, style choices
- background: Personal background, job, expertise, location
- project: Current projects, goals, what they're working on
- instruction: How they want to be communicated with, recurring requests

Return a JSON array of objects. Each object has "content" (the fact as a concise sentence) and "category" (one of the four above).
Return an empty array [] if no meaningful user facts are present.
Return ONLY the JSON array, no other text.

Examples:
[{{"content": "Prefers Python over JavaScript", "category": "preference"}},
 {{"content": "Works as a machine learning engineer", "category": "background"}}]"""


class MemoryService:
    """Extract, store, and retrieve cross-session user memories."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.memory_enabled
        self.extract_interval = settings.memory_extract_interval
        self.max_facts = settings.memory_max_facts_per_extraction
        self.dedup_threshold = settings.memory_dedup_threshold
        self.retrieval_threshold = settings.memory_retrieval_threshold
        self.retrieval_top_k = settings.memory_retrieval_top_k
        self.llm_mode = settings.memory_llm_mode
        self.extract_max_tokens = settings.memory_extract_max_tokens

    def should_extract(self, total_message_count: int) -> bool:
        """Check if memory extraction should trigger (every N messages)."""
        if not self.enabled:
            return False
        return total_message_count > 0 and total_message_count % self.extract_interval == 0

    async def extract_and_store(
        self,
        session_id: str,
        user_id: str,
        user_token: str,
        recent_messages: List[Dict[str, Any]],
    ) -> None:
        """Background task: extract facts from recent messages, dedup, store.

        Runs via asyncio.create_task — must not raise into the caller.
        """
        try:
            await self._do_extract_and_store(
                session_id, user_id, user_token, recent_messages
            )
        except Exception as exc:
            print(f"[memory] Error extracting for user {user_id}: {type(exc).__name__}: {exc}")

    async def _do_extract_and_store(
        self,
        session_id: str,
        user_id: str,
        user_token: str,
        recent_messages: List[Dict[str, Any]],
    ) -> None:
        """Core extraction and storage logic."""
        # 1. Format recent messages for LLM
        messages_text = self._format_messages(recent_messages[-self.extract_interval:])

        # 2. Call LLM to extract facts
        from app.services.inference import get_inference_service
        inference = get_inference_service()

        result = await inference.generate_response(
            messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(
                messages_text=messages_text
            )}],
            mode=self.llm_mode,
            max_tokens=self.extract_max_tokens,
            temperature=0.0,
        )

        raw_content = result.get("content", "").strip()
        # Strip think tags
        raw_content = _THINK_RE.sub("", raw_content).strip()
        close_idx = raw_content.find("</think>")
        if close_idx != -1:
            raw_content = raw_content[close_idx + len("</think>"):].strip()

        # 3. Parse JSON response
        facts = self._parse_facts(raw_content)
        if not facts:
            print(f"[memory] No facts extracted from {len(recent_messages)} messages")
            return

        # 4. Embed each fact
        from app.services.embedding import get_embedding_service
        embedding_service = get_embedding_service()

        fact_texts = [f["content"] for f in facts]
        embeddings = await asyncio.to_thread(
            embedding_service.embed_batch, fact_texts
        )

        # 5. For each fact, check for duplicates and store/update
        stored = 0
        updated = 0
        for fact, embedding in zip(facts, embeddings):
            action = await self._dedup_and_store(
                user_id=user_id,
                user_token=user_token,
                content=fact["content"],
                category=fact["category"],
                embedding=embedding,
                session_id=session_id,
            )
            if action == "stored":
                stored += 1
            elif action == "updated":
                updated += 1

        print(
            f"[memory] User {user_id}: extracted {len(facts)} facts, "
            f"stored {stored}, updated {updated}, "
            f"skipped {len(facts) - stored - updated} (duplicates)"
        )

    async def _dedup_and_store(
        self,
        user_id: str,
        user_token: str,
        content: str,
        category: str,
        embedding: List[float],
        session_id: str,
    ) -> str:
        """Check for duplicate via cosine similarity, store or update.

        Returns 'stored', 'updated', or 'skipped'.
        """
        similar = await self._search_similar(
            user_token, embedding, threshold=self.dedup_threshold, limit=1
        )

        if similar:
            existing = similar[0]
            if existing["content"].strip().lower() != content.strip().lower():
                await asyncio.to_thread(
                    self._update_memory, existing["memory_id"], content, embedding, user_token
                )
                return "updated"
            return "skipped"

        await asyncio.to_thread(
            self._store_memory, user_id, content, category, embedding, session_id, user_token
        )
        return "stored"

    async def _search_similar(
        self,
        user_token: str,
        embedding: List[float],
        threshold: float,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Search for similar memories using the match_memories RPC."""
        rpc_url = f"{self.settings.supabase_url}/rest/v1/rpc/match_memories"
        headers = {
            "apikey": self.settings.supabase_anon_key,
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "query_embedding": embedding,
            "match_threshold": threshold,
            "match_count": limit,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(rpc_url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            print(f"[memory] Similar search failed: {type(exc).__name__}: {exc}")
            return []

    def _store_memory(
        self,
        user_id: str,
        content: str,
        category: str,
        embedding: List[float],
        session_id: str,
        user_token: str,
    ) -> None:
        """Store a new memory via REST API (synchronous)."""
        from app.services.database import get_database_service
        db = get_database_service()
        db._request(
            "POST",
            "user_memories",
            user_token=user_token,
            json={
                "user_id": user_id,
                "content": content,
                "category": category,
                "embedding": embedding,
                "source_session_id": session_id,
            },
        )

    def _update_memory(
        self,
        memory_id: str,
        content: str,
        embedding: List[float],
        user_token: str,
    ) -> None:
        """Update an existing memory's content and embedding (synchronous)."""
        from app.services.database import get_database_service
        db = get_database_service()
        db._request(
            "PATCH",
            f"user_memories?id=eq.{memory_id}",
            user_token=user_token,
            json={
                "content": content,
                "embedding": embedding,
            },
        )

    async def retrieve_relevant_memories(
        self,
        user_token: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to the current query.

        Called at query time (inline, not background) to inject context.
        """
        if not self.enabled:
            return []

        from app.services.embedding import get_embedding_service
        embedding_service = get_embedding_service()

        query_embedding = await asyncio.to_thread(
            embedding_service.embed, query
        )

        memories = await self._search_similar(
            user_token, query_embedding,
            threshold=self.retrieval_threshold,
            limit=self.retrieval_top_k,
        )

        if memories:
            print(
                f"[memory] Retrieved {len(memories)} relevant memories "
                f"(top sim: {memories[0].get('similarity', 0):.3f})"
            )

        return memories

    def format_memories_for_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories into a system message block."""
        if not memories:
            return ""

        lines = ["USER CONTEXT (remembered from previous conversations):"]
        for m in memories:
            cat = m.get("category", "").upper()
            content = m.get("content", "")
            lines.append(f"  [{cat}] {content}")

        return "\n".join(lines)

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for the extraction prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    def _parse_facts(self, raw: str) -> List[Dict[str, str]]:
        """Parse LLM JSON response into a list of fact dicts."""
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)

            facts = json.loads(cleaned)
            if not isinstance(facts, list):
                return []

            valid_categories = {"preference", "background", "project", "instruction"}
            validated = []
            for f in facts[:self.max_facts]:
                if isinstance(f, dict) and "content" in f and "category" in f:
                    cat = f["category"].lower()
                    if cat in valid_categories:
                        validated.append({
                            "content": str(f["content"]).strip(),
                            "category": cat,
                        })
            return validated

        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[memory] Failed to parse LLM response as JSON: {exc}")
            print(f"[memory] Raw response: {raw[:200]}")
            return []


_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get cached MemoryService instance."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService(get_settings())
    return _memory_service
