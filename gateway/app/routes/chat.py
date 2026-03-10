"""
Chat endpoints for the Gateway API.
"""

import re
from datetime import date
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.middleware.auth import JWTPayload, get_current_user
from app.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    ChatMode,
    InferenceMetadata,
    SessionInfo,
    SessionRenameRequest,
    SessionListResponse,
)
from app.services.database import DatabaseService, get_database_service
from app.services.inference import InferenceService, get_inference_service


SYSTEM_PROMPT_TEMPLATE = """\
Your name is Alfred.

Role: Stoic Engineer-Philosopher.
Tone: Calm, rational, ordinary words. Short-to-medium sentences. First-principles thinking. Zero fluff.

Constraints:
- Never complain, posture, or show frustration.
- Speak the truth. Admit uncertainty clearly.
- Value long-term thinking and intellectual honesty.
- Do not mention named people unless asked.
- Response length: Match depth to the question. Simple questions get direct answers. Complex questions get thorough explanations. Never pad, never truncate prematurely. Never sacrifice correctness for brevity.

Problem Solving:
1. Break to first principles.
2. Question every implicit assumption.
3. Rebuild reasoning upward.
4. Propose simplest solution that works long-term.

Code Review:
- Clean, readable, maintainable code first.
- Performance matters, but never at cost of correctness.
- Prefer boring & correct over clever.

Response Structure:
1. Core Insight / Most important sentence.
2. Reasoning Chain (Short, logical steps).
3. Practical Implication or Next Action.

Do not lecture about philosophy unless directly asked.

Current date: {current_date}\
"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _build_system_message() -> dict:
    return {
        "role": "system",
        "content": SYSTEM_PROMPT_TEMPLATE.format(current_date=date.today().strftime("%B %Y")),
    }


def _extract_thinking(content: str) -> Optional[str]:
    """Extract thinking block text from raw content, if present."""
    m = _THINK_RE.search(content)
    if m:
        inner = content[m.start() + len("<think>"):m.end() - len("</think>")].strip()
        return inner or None
    close_idx = content.find("</think>")
    if close_idx != -1:
        inner = content[:close_idx].strip()
        return inner or None
    return None


def _strip_thinking(content: str) -> str:
    """Remove thinking blocks so DB history stays clean.

    Handles two formats:
      - ``<think>...</think>`` — standard tags
      - ``...</think>`` — mlx_vlm strips ``<think>`` to empty string
    """
    result = _THINK_RE.sub("", content)
    if result != content:
        return result.strip()
    close_idx = content.find("</think>")
    if close_idx != -1:
        return content[close_idx + len("</think>"):].strip()
    return content.strip()


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
    inference: InferenceService = Depends(get_inference_service),
) -> ChatResponse:
    """
    Send a message and get an AI response.
    
    - Creates a new session if session_id is not provided
    - Stores user message and AI response in database
    - Returns the AI response with session info
    """
    user_id = user.user_id
    token = user.raw_token
    
    # Get or create session
    if request.session_id:
        session = db.get_session(request.session_id, user_id, user_token=token)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
    else:
        title = request.message[:50] + "..." if len(request.message) > 50 else request.message
        session = db.create_session(user_id, title, user_token=token)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session"
            )
    
    session_id = session["id"]
    
    # Store user message
    user_msg = db.create_message(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=request.message,
        user_token=token,
    )
    if not user_msg:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store message"
        )
    
    # Get conversation history for context
    history = db.get_session_messages(session_id, user_id, limit=20, user_token=token)
    messages = [_build_system_message()] + [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]
    
    # Generate AI response
    inference_result = await inference.generate_response(
        messages=messages,
        mode=request.mode.value,
    )
    
    # Store assistant response (strip thinking blocks, persist reasoning separately)
    raw = inference_result["content"]
    reasoning = inference_result.get("reasoning_content") or _extract_thinking(raw)
    clean_content = _strip_thinking(raw)
    assistant_msg = db.create_message(
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=clean_content,
        mode_used=inference_result["mode_used"],
        tokens_used=inference_result.get("tokens_used"),
        reasoning_content=reasoning,
        user_token=token,
    )
    if not assistant_msg:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store response"
        )
    
    # Build inference metadata for the response
    inference_meta = InferenceMetadata(
        mode_used=inference_result["mode_used"],
        model=inference_result.get("model"),
        fallback_used=inference_result.get("fallback_used", False),
        original_mode=inference_result.get("original_mode"),
        latency_ms=inference_result.get("latency_ms"),
        tokens_used=inference_result.get("tokens_used"),
        prompt_tokens=inference_result.get("prompt_tokens"),
        completion_tokens=inference_result.get("completion_tokens"),
        finish_reason=inference_result.get("finish_reason"),
    )
    
    return ChatResponse(
        session_id=session_id,
        message=ChatMessage(
            id=assistant_msg["id"],
            role="assistant",
            content=assistant_msg["content"],
            mode_used=assistant_msg.get("mode_used"),
            model_used=inference_result.get("model"),
            created_at=assistant_msg["created_at"],
        ),
        session_title=session.get("title"),
        inference=inference_meta,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
) -> SessionListResponse:
    """List all chat sessions for the current user."""
    sessions = db.list_sessions(user.user_id, user_token=user.raw_token)
    return SessionListResponse(
        sessions=[
            SessionInfo(
                id=s["id"],
                title=s["title"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                is_archived=s.get("is_archived", False),
            )
            for s in sessions
        ]
    )


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(
    session_id: str,
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
) -> List[ChatMessage]:
    """Get all messages in a session."""
    token = user.raw_token
    session = db.get_session(session_id, user.user_id, user_token=token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    messages = db.get_session_messages(session_id, user.user_id, user_token=token)
    return [
        ChatMessage(
            id=m["id"],
            role=m["role"],
            content=m["content"],
            mode_used=m.get("mode_used"),
            model_used=m.get("model_used"),
            reasoning_content=m.get("reasoning_content"),
            created_at=m["created_at"],
        )
        for m in messages
    ]


@router.patch("/sessions/{session_id}", response_model=SessionInfo)
async def rename_session(
    session_id: str,
    request: SessionRenameRequest,
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
) -> SessionInfo:
    """Rename a chat session."""
    token = user.raw_token
    session = db.get_session(session_id, user.user_id, user_token=token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    db.update_session_title(session_id, user.user_id, request.title, user_token=token)
    updated = db.get_session(session_id, user.user_id, user_token=token)
    return SessionInfo(
        id=updated["id"],
        title=updated["title"],
        created_at=updated["created_at"],
        updated_at=updated["updated_at"],
        is_archived=updated.get("is_archived", False),
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
):
    """Delete a chat session and all its messages."""
    success = db.delete_session(session_id, user.user_id, user_token=user.raw_token)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )


@router.post("/stream")
async def send_message_stream(
    request: ChatRequest,
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
    inference: InferenceService = Depends(get_inference_service),
):
    """
    Send a message and stream the AI response via Server-Sent Events (SSE).
    
    - Same session/message logic as POST /chat
    - Returns SSE stream with OpenAI-compatible chunks
    - Final event includes full content for database storage
    """
    import json
    
    user_id = user.user_id
    token = user.raw_token
    
    # Get or create session
    if request.session_id:
        session = db.get_session(request.session_id, user_id, user_token=token)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
    else:
        title = request.message[:50] + "..." if len(request.message) > 50 else request.message
        session = db.create_session(user_id, title, user_token=token)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session"
            )
    
    session_id = session["id"]
    
    # Store user message
    user_msg = db.create_message(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=request.message,
        user_token=token,
    )
    if not user_msg:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store message"
        )
    
    # Get conversation history
    history = db.get_session_messages(session_id, user_id, limit=20, user_token=token)
    messages = [_build_system_message()] + [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]
    
    async def event_generator():
        """Generate SSE events from inference stream."""
        full_content = []
        full_reasoning = []
        
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        
        async for chunk in inference.generate_response_stream(
            messages=messages,
            mode=request.mode.value,
        ):
            try:
                if chunk.startswith("data: ") and not chunk.strip().endswith("[DONE]"):
                    data = json.loads(chunk[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        full_content.append(delta["content"])
                    if "reasoning_content" in delta:
                        full_reasoning.append(delta["reasoning_content"])
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
            
            yield chunk
        
        # Build final content for DB (strip thinking tags, persist reasoning separately)
        raw_content = "".join(full_content)
        content = _strip_thinking(raw_content)

        reasoning = "".join(full_reasoning) if full_reasoning else _extract_thinking(raw_content)
        if not reasoning:
            reasoning = None

        if content:
            db.create_message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=content,
                mode_used=request.mode.value,
                reasoning_content=reasoning,
                user_token=token,
            )
        elif not content and reasoning:
            db.create_message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content="(no visible response)",
                mode_used=request.mode.value,
                reasoning_content=reasoning,
                user_token=token,
            )
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
        },
    )
