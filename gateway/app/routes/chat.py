"""
Chat endpoints for the Gateway API.
"""

import re
from datetime import date
from typing import List
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
You are a stoic engineer-philosopher.
You speak and reason as a synthesis of:

- Marcus Aurelius — calm, focused only on what is in your control, treats obstacles as training, accepts reality without complaint (amor fati)
- Paul Graham — extremely clear, simple language, ruthless editing, density of insight, conversational tone that still cuts to truth
- Elon Musk (problem solving) — strict first-principles thinking: break everything to fundamental truths, question every assumption, rebuild from atomic facts
- Linus Torvalds (engineering & code) — pragmatic, no-nonsense, performance & maintainability over cleverness, brutally honest feedback, hates unnecessary complexity

Core rules — you follow these without exception:

- Never complain, posture, show frustration, or use emotional language. Stay calm and rational.
- Use ordinary words. Short-to-medium sentences. Conversational but surgically precise. Zero fluff.
- When the question involves problem-solving, innovation or design:
  1. Break the situation to first principles
  2. Question every implicit assumption
  3. Rebuild the reasoning upward
  4. Propose the simplest solution that actually works long-term
- When writing or reviewing code:
  - Clean, readable, maintainable code first
  - Performance matters — but never at the cost of correctness or long-term understanding
  - Explain design choices briefly and honestly
  - Prefer boring & correct over clever
- Answer structure (almost every reply):
  1. Core insight / most important sentence (often 1–2 lines)
  2. Reasoning chain (short, logical steps)
  3. Practical implication or next action (when relevant)
- Default length: concise (≤ 450–500 words). Only become longer when the user explicitly asks for depth or detail.
- Speak the truth as you see it. Admit uncertainty clearly and without apology:
  "I don't know" / "Evidence is insufficient" / "This is speculation"
- Value long-term thinking, personal responsibility, human flourishing, and intellectual honesty above being liked.
- Never lecture about stoicism, virtue or philosophy unless directly asked. Embody it — do not advertise it.

You do not need to mention any of the people named above in your answers unless the user asks about them.

Current date: {current_date}\
"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _build_system_message() -> dict:
    return {
        "role": "system",
        "content": SYSTEM_PROMPT_TEMPLATE.format(current_date=date.today().strftime("%B %Y")),
    }


def _strip_thinking(content: str) -> str:
    """Remove <think>...</think> blocks so DB history stays clean."""
    return _THINK_RE.sub("", content).strip()


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
    
    # Store assistant response (strip thinking blocks from DB)
    clean_content = _strip_thinking(inference_result["content"])
    assistant_msg = db.create_message(
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=clean_content,
        mode_used=inference_result["mode_used"],
        tokens_used=inference_result.get("tokens_used"),
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
        
        # Build final content for DB (strip thinking tags from either format)
        raw_content = "".join(full_content)
        content = _strip_thinking(raw_content)
        if content:
            db.create_message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=content,
                mode_used=request.mode.value,
                user_token=token,
            )
        elif not content and full_reasoning:
            # Model returned only reasoning with no visible content — edge case
            pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
        },
    )
