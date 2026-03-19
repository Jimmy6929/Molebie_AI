"""
Chat endpoints for the Gateway API.
"""

import asyncio
import re
from datetime import date
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse

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
    TTSRequest,
)
from app.services.database import DatabaseService, get_database_service
from app.services.inference import InferenceService, get_inference_service
from app.services.web_search import WebSearchService, get_web_search_service
from app.services.rag import RAGService, get_rag_service


SYSTEM_PROMPT_TEMPLATE = """\
Your name is Alfred like in batman movie.

Role: Stoic Engineer.
Tone: Calm, rational, ordinary words. First-principles thinking. Zero fluff.

PRIORITY:
• Understand the problem deeply before answering.
• Break problems into clear steps.
• Show structured thinking.

PROCESS:
1. Restate the problem briefly.
2. Identify possible causes or approaches.
3. Choose the best option with reasoning.
4. Provide a clear solution.

CODING & DEBUGGING:
• Focus on root cause, not quick patches.
• Prefer minimal and safe changes.
• Consider edge cases and risks.
• Suggest simple ways to test the solution.

STYLE:
• Structured, logical, and precise.
• Use sections and bullet points.
• Avoid unnecessary verbosity.

EVIDENCE HANDLING:
Follow these steps when sources (web results or documents) are provided:
1. ASSESS — Review each source. Note its type (official docs, forum, news, general web) and relevance to the question.
2. ANSWER — Base your answer primarily on the provided sources. Clearly distinguish between:
   - Facts from sources: cite them inline using [Source Title](url) for web results, or (filename) for documents.
   - Your inference: label as "Based on this, I infer..." or "This suggests..."
   - General knowledge (not verified by sources): label as "From general knowledge (not verified against current sources)..."
3. SIGNAL CONFIDENCE — After answering, indicate your confidence level:
   - High: multiple authoritative sources agree.
   - Moderate: some sources support the answer but coverage is incomplete.
   - Low: limited or weak sources; answer may be outdated or incomplete.
   - If sources conflict, present both positions and note the disagreement.

WHEN YOU DON'T KNOW:
• If the provided sources do not adequately answer the question, say so explicitly.
• Use phrases like: "The available sources don't cover this", "I don't have enough information to confirm", "This is uncertain based on what I found".
• It is better to say "I'm not sure" than to guess confidently.
• NEVER fabricate citations, URLs, or source references. Only cite sources actually provided to you.

SOURCE PRIORITY:
• Official documentation and .gov/.edu sites > news outlets > forums > general web pages.
• When sources conflict, prefer more authoritative sources but note the disagreement.
• Treat HIGH MATCH document chunks with more confidence than WEAK MATCH ones.

Current date: {current_date}\
"""

SYSTEM_PROMPT_VOICE_TEMPLATE = """\
Your name is Alfred.

Role: Stoic engineer — calm, competent, quietly dry British humour when it fits naturally.
Tone: Measured, rational, ordinary precise English. First-principles thinking. Minimal fluff — only light courtesy / dry wit when it actually helps clarity or the interaction.

You may use short Alfred-style acknowledgements ("Very good, sir." / "If I may, sir…" / "Noted.") when they serve utility or match the moment. Do not force them.

PRIORITY
• Understand what is actually being asked before answering.
• When the question is technical / analytical / problem-solving → show structured reasoning.
• When the question is casual, social or "introduce yourself" → give a short, in-character reply and move on. Do not philosophically deconstruct identity / self / AI nature unless explicitly asked to analyse that topic.

RESPONSE STRUCTURE (use when the topic is non-trivial / problem-oriented)
1. Restate the core request or problem in 1–2 clear sentences.
2. List main approaches / causes / constraints worth considering.
3. Select the path you judge best + brief reasoning.
4. Deliver the concrete answer / solution / next step(s).

When the question is light / social / one-shot ("hi", "who are you", "tell me a joke") → skip the full structure. Reply briefly, stay in character, keep momentum.

EVIDENCE HANDLING (for spoken delivery)
When sources are provided, follow these steps:
1. ASSESS — Note which sources are most relevant and authoritative.
2. ANSWER — Use the sources. Distinguish between:
   - Facts from sources: cite naturally, e.g. "According to the Python documentation..." or "Based on a Reuters report..."
   - Your inference: say "Based on this, I'd say..." or "This suggests..."
   - General knowledge not backed by sources: say "From general knowledge, though I can't verify this against current sources..."
3. SIGNAL CONFIDENCE — Briefly indicate how confident you are:
   - "I'm fairly confident based on multiple sources" / "This is well-supported"
   - "I found some information but it's not comprehensive"
   - "I'm not entirely sure — the sources are limited on this"
   - If sources conflict: present both sides briefly.

WHEN YOU DON'T KNOW
• If sources don't adequately answer the question, say so: "I don't have strong sources on that" or "The information I found doesn't fully cover this."
• Better to say "I'm not certain" than to guess confidently.
• Never fabricate source references.

DOCUMENT MEMORY
• When document context is provided, use it to answer the user's question.
• Reference the source document filename naturally in conversation when citing specific information.
• If the user's question is not related to any provided document context, answer normally.

GENERAL RULES
• Never lecture the user about what AI "really" is unless directly asked to explain it.
• Do not refuse small-talk by claiming you have no self / no emotions / no past — simply answer in character and proceed.
• If unsure what the user wants, ask one calm clarifying question instead of over-analysing.

Current date: {current_date}\
"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _build_evidence_summary(
    search_results: List[Dict[str, Any]],
    rag_chunks: List[Dict[str, Any]],
) -> str:
    """Generate an evidence meta-block for the system message.

    Tells the model what evidence is available so it can self-calibrate
    confidence before answering.
    """
    parts = []
    if search_results:
        domains = {r.get("domain", "") for r in search_results if r.get("domain")}
        parts.append(f"{len(search_results)} web results from {len(domains)} domain(s)")
    if rag_chunks:
        top_sim = max(c.get("similarity", 0) for c in rag_chunks)
        parts.append(f"{len(rag_chunks)} document chunks (best match: {top_sim:.2f})")

    if parts:
        return f"EVIDENCE AVAILABLE: {'; '.join(parts)}.\nUse this to calibrate your confidence."
    return "EVIDENCE AVAILABLE: None. Answer from general knowledge only. Signal low confidence."


def _rag_quality_header(chunks: List[Dict[str, Any]]) -> str:
    """Build a descriptive header for the RAG context section."""
    if not chunks:
        return "DOCUMENT CONTEXT:"
    top_sim = max(c.get("similarity", 0) for c in chunks)
    if top_sim > 0.75:
        quality = "high relevance"
    elif top_sim > 0.6:
        quality = "moderate relevance"
    else:
        quality = "low relevance"
    return f"DOCUMENT CONTEXT ({len(chunks)} chunks, {quality}, top match: {top_sim:.2f}):"


def _validate_response_sources(
    response_text: str,
    search_results: List[Dict[str, Any]],
    rag_chunks: List[Dict[str, Any]],
) -> None:
    """Log whether the response referenced any of the provided sources.

    This is a lightweight monitoring check — no model calls, just substring matching.
    """
    if not search_results and not rag_chunks:
        return

    referenced = False
    lower_response = response_text.lower()

    for r in search_results:
        title = (r.get("title") or "").lower()
        domain = (r.get("domain") or "").lower()
        if (title and title in lower_response) or (domain and domain in lower_response):
            referenced = True
            break

    if not referenced:
        for c in rag_chunks:
            filename = (c.get("filename") or "").lower()
            if filename and filename in lower_response:
                referenced = True
                break

    tag = "sources_referenced" if referenced else "sources_not_referenced"
    source_count = len(search_results) + len(rag_chunks)
    print(f"[chat] Response validation: {tag} (provided: {source_count} sources)")


def _build_system_message(conversation_mode: bool = False) -> dict:
    prompt_template = SYSTEM_PROMPT_VOICE_TEMPLATE if conversation_mode else SYSTEM_PROMPT_TEMPLATE
    return {
        "role": "system",
        "content": prompt_template.format(current_date=date.today().strftime("%B %Y")),
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
    web_search: WebSearchService = Depends(get_web_search_service),
    rag: RAGService = Depends(get_rag_service),
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
    messages = [_build_system_message(request.conversation_mode)] + [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]
    
    # Web search: inject real-time results into context
    search_results = []
    if web_search.should_search(request.message):
        search_results = await web_search.search(request.message)
        if search_results:
            context_text = web_search.format_results_for_context(search_results)
            messages[0]["content"] += f"\n\nWEB SEARCH RESULTS:\n{context_text}"

    # RAG: inject relevant document chunks (skip in conversation mode to avoid embedding load)
    # Skip RAG entirely when user has no documents to avoid unnecessary embedding model load
    rag_chunks = []
    if not request.conversation_mode:
        if await rag.user_has_documents(token):
            rag_chunks = await rag.retrieve_context(token, request.message)
            if rag_chunks:
                rag_text = rag.format_context(rag_chunks)
                header = _rag_quality_header(rag_chunks)
                messages[0]["content"] += f"\n\n{header}\n{rag_text}"

    # Inject evidence summary so the model can self-calibrate confidence
    evidence_summary = _build_evidence_summary(search_results, rag_chunks)
    messages[0]["content"] += f"\n\n{evidence_summary}"

    total_chars = sum(len(m.get("content", "")) for m in messages)
    print(f"[chat] Sending {len(messages)} messages ({total_chars} chars) to inference")

    # Voice conversation always uses instant tier (Qwen 3.5 4B); config via INFERENCE_INSTANT_*
    inference_mode = "instant" if request.conversation_mode else request.mode.value

    # Generate AI response
    inference_result = await inference.generate_response(
        messages=messages,
        mode=inference_mode,
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

    # Log whether the response referenced provided sources
    _validate_response_sources(clean_content, search_results, rag_chunks)

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
            inference_model=inference_result.get("model"),
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
            inference_model=m.get("model_used"),
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
    http_request: Request,
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
    inference: InferenceService = Depends(get_inference_service),
    web_search: WebSearchService = Depends(get_web_search_service),
    rag: RAGService = Depends(get_rag_service),
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
    messages = [_build_system_message(request.conversation_mode)] + [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]

    # Web search: run before streaming so results are ready
    search_results = []
    if web_search.should_search(request.message):
        search_results = await web_search.search(request.message)
        if search_results:
            context_text = web_search.format_results_for_context(search_results)
            messages[0]["content"] += f"\n\nWEB SEARCH RESULTS:\n{context_text}"

    # RAG: inject relevant document chunks (skip in conversation mode to avoid embedding load)
    # Skip RAG entirely when user has no documents to avoid unnecessary embedding model load
    rag_chunks = []
    if not request.conversation_mode:
        if await rag.user_has_documents(token):
            rag_chunks = await rag.retrieve_context(token, request.message)
            if rag_chunks:
                rag_text = rag.format_context(rag_chunks)
                header = _rag_quality_header(rag_chunks)
                messages[0]["content"] += f"\n\n{header}\n{rag_text}"

    # Inject evidence summary so the model can self-calibrate confidence
    evidence_summary = _build_evidence_summary(search_results, rag_chunks)
    messages[0]["content"] += f"\n\n{evidence_summary}"

    total_chars = sum(len(m.get("content", "")) for m in messages)
    print(f"[chat] Sending {len(messages)} messages ({total_chars} chars) to inference")

    # Voice conversation always uses instant tier (Qwen 3.5 4B); config via INFERENCE_INSTANT_*
    inference_mode = "instant" if request.conversation_mode else request.mode.value

    async def event_generator():
        """Generate SSE events from inference stream."""
        full_content = []
        full_reasoning = []
        client_disconnected = False
        
        if await http_request.is_disconnected():
            return

        try:
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        except (BrokenPipeError, ConnectionResetError, RuntimeError, asyncio.CancelledError):
            return

        if search_results:
            sources = [{"title": r["title"], "url": r["url"]} for r in search_results]
            try:
                yield f"data: {json.dumps({'type': 'search_done', 'sources': sources})}\n\n"
            except (BrokenPipeError, ConnectionResetError, RuntimeError, asyncio.CancelledError):
                return
        
        async for chunk in inference.generate_response_stream(
            messages=messages,
            mode=inference_mode,
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

            if await http_request.is_disconnected():
                client_disconnected = True
                break

            try:
                yield chunk
            except (BrokenPipeError, ConnectionResetError, RuntimeError, asyncio.CancelledError):
                client_disconnected = True
                break

        if client_disconnected:
            return
        
        # Build final content for DB (strip thinking tags, persist reasoning separately)
        raw_content = "".join(full_content)
        content = _strip_thinking(raw_content)

        reasoning = "".join(full_reasoning) if full_reasoning else _extract_thinking(raw_content)
        if not reasoning:
            reasoning = None

        save_content = content if content else ("(no visible response)" if reasoning else None)
        if save_content:
            try:
                db.create_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=save_content,
                    mode_used=inference_mode,
                    reasoning_content=reasoning,
                    user_token=token,
                )
                _validate_response_sources(save_content, search_results, rag_chunks)
            except Exception as db_err:
                print(f"[chat] Failed to save assistant message to DB: {db_err}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
        },
    )


@router.post("/transcribe")
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    verify_speaker: bool = Form(False),
    user: JWTPayload = Depends(get_current_user),
) -> Dict[str, Any]:
    """Transcribe audio to text. Optionally verify the speaker matches enrolled voice."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    from app.services.transcription import transcribe_audio

    try:
        text = await transcribe_audio(audio_bytes, file.filename or "audio.webm")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

    result: Dict[str, Any] = {"text": text}

    if verify_speaker:
        from app.services.speaker import verify_speaker as do_verify

        try:
            verified, confidence = await do_verify(audio_bytes, user.sub, file.filename or "audio.webm")
            result["speaker_verified"] = verified
            result["speaker_confidence"] = round(confidence, 4)
        except Exception:
            result["speaker_verified"] = True
            result["speaker_confidence"] = 1.0

    return result


@router.post("/voice-enroll")
async def enroll_voice(
    file: UploadFile = File(...),
    user: JWTPayload = Depends(get_current_user),
) -> Dict[str, Any]:
    """Add a voice sample for speaker enrollment."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    from app.services.speaker import enroll_voice_sample

    try:
        return await enroll_voice_sample(audio_bytes, user.sub, file.filename or "audio.webm")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {exc}") from exc


@router.get("/voice-profile")
async def voice_profile_status(
    user: JWTPayload = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get voice enrollment status for the current user."""
    from app.services.speaker import get_voice_profile_status

    return await get_voice_profile_status(user.sub)


@router.delete("/voice-profile")
async def delete_voice_profile_endpoint(
    user: JWTPayload = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete the enrolled voice profile."""
    from app.services.speaker import delete_voice_profile

    deleted = await delete_voice_profile(user.sub)
    return {"deleted": deleted}


@router.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    user: JWTPayload = Depends(get_current_user),
):
    """Synthesize speech from text using Kokoro TTS."""
    import httpx
    from app.config import get_settings

    settings = get_settings()
    kokoro_url = settings.kokoro_tts_url

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{kokoro_url}/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": request.text,
                    "voice": request.voice,
                    "speed": request.speed,
                    "response_format": "mp3",
                },
            )
            resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Kokoro TTS service not reachable. Is it running?")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Kokoro TTS error: {exc.response.status_code}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}")

    return Response(content=resp.content, media_type="audio/mpeg")
