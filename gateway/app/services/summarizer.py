"""
Conversation summarisation service.

Generates rolling summaries of long conversations to keep context windows
compact. Runs as a non-blocking background task after each assistant
message. When a summary exists, the chat route reduces history from 20
to 10 recent messages and prepends the summary to the system message.
"""

import asyncio
import re
from typing import Optional, Dict, Any, List

from app.config import Settings, get_settings

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_SUMMARIZE_PROMPT = """\
You are a conversation summariser. Produce a concise summary of the conversation below.

{existing_summary_block}
CONVERSATION TO SUMMARISE:
{messages_text}

INSTRUCTIONS:
- Capture the key topics discussed, decisions made, important facts shared, and any open questions.
- If an existing summary was provided, build on it — integrate new information, don't repeat verbatim.
- Write in third person ("The user asked about...", "The assistant explained...").
- Be concise but complete. Aim for 2-4 paragraphs.
- Output ONLY the summary, nothing else."""


class SummariserService:
    """Generate and update rolling conversation summaries."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.summary_enabled
        self.trigger_threshold = settings.summary_trigger_threshold
        self.recent_messages = settings.summary_recent_messages
        self.max_input_chars = settings.summary_max_input_chars
        self.max_output_tokens = settings.summary_max_output_tokens
        self.llm_mode = settings.summary_llm_mode

    def should_summarise(self, total_message_count: int, summary_message_count: int) -> bool:
        """Check if summarisation should trigger."""
        if not self.enabled:
            return False
        unsummarised = total_message_count - summary_message_count
        return unsummarised >= self.trigger_threshold

    async def summarise_session(
        self,
        session_id: str,
        user_id: str,
        user_token: str,
    ) -> None:
        """Background task: fetch messages, generate summary, store it.

        Runs via asyncio.create_task — must not raise into the caller.
        """
        try:
            await self._do_summarise(session_id, user_id, user_token)
        except Exception as exc:
            print(f"[summariser] Error for session {session_id}: {type(exc).__name__}: {exc}")

    async def _do_summarise(
        self,
        session_id: str,
        user_id: str,
        user_token: str,
    ) -> None:
        """Core summarisation logic."""
        from app.services.database import get_database_service
        db = get_database_service()

        # 1. Get session to read existing summary state
        session = await asyncio.to_thread(
            db.get_session, session_id, user_id, user_token
        )
        if not session:
            print(f"[summariser] Session {session_id} not found")
            return

        existing_summary = session.get("summary") or ""
        summary_msg_count = session.get("summary_message_count", 0)

        # 2. Fetch ALL messages for this session
        all_messages = await asyncio.to_thread(
            db.get_session_messages, session_id, user_id, limit=500, user_token=user_token
        )
        total_count = len(all_messages)

        if not self.should_summarise(total_count, summary_msg_count):
            return

        # 3. Determine which messages to summarise:
        #    From summary_msg_count to (total - recent_messages)
        end_idx = max(0, total_count - self.recent_messages)
        start_idx = summary_msg_count
        messages_to_summarise = all_messages[start_idx:end_idx]

        if not messages_to_summarise:
            return

        # 4. Format messages for LLM, respecting char limit
        messages_text = self._format_messages(messages_to_summarise)

        # 5. Build prompt
        existing_block = ""
        if existing_summary:
            existing_block = f"EXISTING SUMMARY (build on this, integrate new info):\n{existing_summary}\n\n"

        prompt = _SUMMARIZE_PROMPT.format(
            existing_summary_block=existing_block,
            messages_text=messages_text,
        )

        # 6. Call LLM
        from app.services.inference import get_inference_service
        inference = get_inference_service()

        result = await inference.generate_response(
            messages=[{"role": "user", "content": prompt}],
            mode=self.llm_mode,
            max_tokens=self.max_output_tokens,
            temperature=0.3,
        )

        summary_text = result.get("content", "").strip()
        if not summary_text:
            print("[summariser] LLM returned empty summary")
            return

        # Strip any <think> tags
        summary_text = _THINK_RE.sub("", summary_text).strip()
        close_idx = summary_text.find("</think>")
        if close_idx != -1:
            summary_text = summary_text[close_idx + len("</think>"):].strip()

        if not summary_text:
            print("[summariser] Summary empty after stripping think tags")
            return

        # 7. Update session in DB
        new_summary_count = end_idx
        await asyncio.to_thread(
            self._update_session_summary,
            session_id, user_id, summary_text, new_summary_count, user_token,
        )

        print(
            f"[summariser] Session {session_id}: summarised messages "
            f"{start_idx + 1}-{end_idx} ({len(messages_to_summarise)} msgs), "
            f"summary={len(summary_text)} chars"
        )

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages into text for the LLM prompt, respecting char limit."""
        lines = []
        total_chars = 0
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            line = f"{role}: {content}"
            if total_chars + len(line) > self.max_input_chars:
                remaining = self.max_input_chars - total_chars
                if remaining > 100:
                    lines.append(line[:remaining] + "...")
                lines.append("[...earlier messages truncated for length...]")
                break
            lines.append(line)
            total_chars += len(line)
        return "\n\n".join(lines)

    def _update_session_summary(
        self,
        session_id: str,
        user_id: str,
        summary: str,
        summary_message_count: int,
        user_token: str,
    ) -> None:
        """Synchronous DB update for the session summary fields."""
        from app.services.database import get_database_service
        db = get_database_service()
        db._request(
            "PATCH",
            f"chat_sessions?id=eq.{session_id}&user_id=eq.{user_id}",
            user_token=user_token,
            json={
                "summary": summary,
                "summary_message_count": summary_message_count,
            },
        )


_summariser_service: Optional[SummariserService] = None


def get_summariser_service() -> SummariserService:
    """Get cached SummariserService instance."""
    global _summariser_service
    if _summariser_service is None:
        _summariser_service = SummariserService(get_settings())
    return _summariser_service
