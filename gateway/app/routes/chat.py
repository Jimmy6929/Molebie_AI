"""
Chat endpoints for the Gateway API.
"""

import asyncio
import base64
import re
import uuid
from datetime import date
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
import httpx

from app.middleware.auth import JWTPayload, get_current_user
from app.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    ChatMode,
    InferenceMetadata,
    SessionInfo,
    SessionRenameRequest,
    SessionPinRequest,
    SessionListResponse,
    TTSRequest,
)
from app.config import get_settings
from app.services.database import DatabaseService, get_database_service
from app.services.inference import InferenceService, get_inference_service
from app.services.web_search import WebSearchService, get_web_search_service
from app.services.rag import RAGService, get_rag_service


SYSTEM_PROMPT_TEMPLATE = """\
Your name is Alfred — as in Alfred Pennyworth. You are the gentleman's gentleman: quietly brilliant, practically competent, and never above a dry observation when the moment calls for it.

PERSONALITY:
• Warm under British reserve. You care about giving a genuinely useful answer.
• Dry wit when it fits — never forced. "I believe that would be inadvisable, sir."
• Firm when needed — if the user is heading in the wrong direction, say so with tact.
• Practical above all. You solve problems, you don't lecture about methodology.
• Use "sir" or "ma'am" sparingly and naturally — once or twice per conversation, not every sentence.

THINKING:
• Understand the problem before answering. Break complex questions into parts.
• First-principles reasoning. Identify root causes, not symptoms.
• When debugging: focus on the actual cause, prefer minimal safe changes.
• Give a clear, decisive answer. Hedging without substance helps no one.

WHEN SOURCES ARE PROVIDED (web results or documents):
Use provided sources as your primary evidence. Your job is to READ them, SYNTHESIZE them, and give a clear answer — not to mechanically list what each source says.

Three principles:
1. USE EVERYTHING AVAILABLE. Read the full content of each source. Combine information across sources to build a complete picture. If sources give partial data, piece it together. If indirect evidence exists (related figures, comparable data, historical context), use it to reason toward an answer.

2. BE HONEST ABOUT WHAT'S FROM WHERE. When citing source data, reference the source naturally: "According to [Source]..." or "The Python docs state...". When adding your own knowledge beyond the sources, signal it: "From what I know..." or "Generally speaking..." — don't pretend it came from a source, but don't refuse to share it either.

3. GIVE A USEFUL ANSWER. Always end with a clear, actionable conclusion. If sources fully answer the question — great, cite and answer. If sources partially answer it — synthesize what they have, fill gaps with your knowledge (labeled), and give your best answer. If sources don't help at all — say so briefly, then answer from your knowledge with appropriate caveats.

WHAT NOT TO DO:
• Don't invent specific numbers, statistics, or measurements and attribute them to sources. If a source says "approximately 500" you can cite that. If no source gives a number, say "I don't have a specific figure from these sources" and offer your best estimate labeled as such.
• Don't fabricate citations, URLs, or source references.
• Don't confuse "not found in these search results" with "this information doesn't exist." If the search didn't find it, say "I didn't find this in the current results" — not "there is no answer."
• Don't be so cautious that you fail to answer the question. A hedged non-answer is worse than a clearly-labeled best estimate.

CONFIDENCE — signal naturally, not with labels:
• Strong evidence → answer directly and cite
• Mixed or partial evidence → give your answer, note the gaps
• Weak evidence → give your best assessment, explain your uncertainty
• No relevant evidence → say so, then still try to help from general knowledge

Current date: {current_date}\
"""

SYSTEM_PROMPT_VOICE_TEMPLATE = """\
Your name is Alfred — as in Alfred Pennyworth. You are the gentleman's gentleman: quietly brilliant, dry wit when the moment calls for it, and always practical.

• Warm under the reserve. Genuinely helpful, not just correct.
• Dry humour when it fits — "I believe that would be inadvisable, sir."
• Firm when needed. If someone's wrong, say so with tact.
• "Sir" or "ma'am" sparingly and naturally.

When the question is casual or social — reply briefly in character and move on. No deconstructing identity or AI nature unless asked.

When the question is substantive — think clearly, then answer clearly:
1. Understand what's really being asked.
2. If sources are provided — read them, combine what they say, and give a direct answer. Cite sources naturally ("According to the Python docs..." or "Reuters reports...").
3. If your own knowledge fills gaps — share it honestly: "From what I know..." or "Generally speaking..."
4. Always finish with a clear, useful answer. Don't just list caveats.

Keep it honest:
• Don't invent numbers and attribute them to sources.
• Don't fabricate citations.
• "Not found in this search" is different from "no answer exists" — say which you mean.
• Don't be so cautious you forget to actually answer the question.

When document context is provided, use it naturally. Reference filenames when citing specific information.

Current date: {current_date}\
"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _build_evidence_summary(
    search_results: List[Dict[str, Any]],
    rag_chunks: List[Dict[str, Any]],
) -> str:
    """Generate an evidence meta-block for the system message.

    Tells the model what evidence is available — including content depth,
    source types, and retrieval completeness — so it can self-calibrate
    confidence before answering.
    """
    if not search_results and not rag_chunks:
        return (
            "EVIDENCE SUMMARY:\n"
            "• No web results. No document matches.\n"
            "• No sources available. Answer from your own knowledge — be upfront about it."
        )

    lines = ["EVIDENCE SUMMARY:"]

    # ── Web results ──
    if search_results:
        domains = {r.get("domain", "") for r in search_results if r.get("domain")}
        full_page = sum(1 for r in search_results if r.get("content_source") == "full_page")
        snippet_only = len(search_results) - full_page

        type_counts: Dict[str, int] = {}
        for r in search_results:
            st = r.get("source_type", "web")
            type_counts[st] = type_counts.get(st, 0) + 1
        type_summary = ", ".join(f"{v} {k}" for k, v in type_counts.items())

        content_desc = []
        if full_page:
            content_desc.append(f"{full_page} full-page")
        if snippet_only:
            content_desc.append(f"{snippet_only} snippet-only")
        content_str = ", ".join(content_desc) if content_desc else str(len(search_results))

        lines.append(
            f"• Web: {len(search_results)} results ({content_str}) from {len(domains)} domain(s)\n"
            f"  Types: {type_summary}"
        )

    # ── RAG chunks ──
    if rag_chunks:
        top_sim = max(c.get("similarity", 0) for c in rag_chunks)
        quality_counts: Dict[str, int] = {"HIGH": 0, "MODERATE": 0, "WEAK": 0}
        for c in rag_chunks:
            sim = c.get("similarity", 0)
            if sim > 0.75:
                quality_counts["HIGH"] += 1
            elif sim > 0.6:
                quality_counts["MODERATE"] += 1
            else:
                quality_counts["WEAK"] += 1
        qual_parts = [f"{v} {k}" for k, v in quality_counts.items() if v > 0]
        lines.append(
            f"• Documents: {len(rag_chunks)} chunks ({', '.join(qual_parts)}, best: {top_sim:.2f})"
        )

    # ── Retrieval completeness ──
    has_full = any(r.get("content_source") == "full_page" for r in search_results) if search_results else False
    has_high_rag = any(c.get("similarity", 0) > 0.75 for c in rag_chunks) if rag_chunks else False

    if has_full and has_high_rag:
        completeness = "FULL"
    elif has_full or has_high_rag:
        completeness = "PARTIAL"
    elif search_results or rag_chunks:
        completeness = "MINIMAL"
    else:
        completeness = "NONE"

    lines.append(
        f"• Retrieval: {completeness}\n"
        "  Use all available evidence to give the best possible answer."
    )

    return "\n".join(lines)


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
        "content": prompt_template.format(current_date=date.today().strftime("%A, %B %d, %Y")),
    }


def _validate_image(data_uri: str, settings) -> tuple:
    """Validate a base64 image data URI. Returns (mime_type, raw_bytes) or raises HTTPException."""
    if not data_uri.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="Image must be a data URI (data:image/...;base64,...)")
    try:
        header, b64_data = data_uri.split(",", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Malformed data URI")

    # Extract MIME type
    mime_type = header.split(":")[1].split(";")[0]
    allowed = [t.strip() for t in settings.vision_allowed_types.split(",")]
    if mime_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Image type {mime_type} not allowed. Allowed: {allowed}")

    try:
        raw_bytes = base64.b64decode(b64_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

    if len(raw_bytes) > settings.vision_max_image_size:
        max_mb = settings.vision_max_image_size / (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"Image too large (max {max_mb:.0f} MB)")

    return mime_type, raw_bytes


def _upload_image_to_storage(settings, user_id: str, raw_bytes: bytes, mime_type: str, filename: str = "image") -> str:
    """Upload image bytes to Supabase Storage chat-images bucket. Returns storage_path."""
    ext = mime_type.split("/")[-1]
    if ext == "jpeg":
        ext = "jpg"
    storage_path = f"{user_id}/{uuid.uuid4().hex}_{filename}.{ext}"
    url = f"{settings.supabase_url}/storage/v1/object/chat-images/{storage_path}"

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            url,
            content=raw_bytes,
            headers={
                "apikey": settings.supabase_service_role_key,
                "Authorization": f"Bearer {settings.supabase_service_role_key}",
                "Content-Type": mime_type,
            },
        )
        if not resp.is_success:
            print(f"[chat] Image storage upload failed: {resp.status_code} {resp.text[:300]}")
            raise HTTPException(status_code=502, detail="Failed to store image")

    return storage_path


def _download_image_from_storage(settings, storage_path: str) -> bytes:
    """Download image bytes from Supabase Storage."""
    url = f"{settings.supabase_url}/storage/v1/object/chat-images/{storage_path}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, headers={
            "apikey": settings.supabase_service_role_key,
            "Authorization": f"Bearer {settings.supabase_service_role_key}",
        })
        resp.raise_for_status()
        return resp.content


def _fetch_session_attachments(session_id: str, user_id: str, token: str) -> str:
    """Fetch session_documents and format them for system prompt injection.

    Returns an empty string if no attachments exist.
    """
    import httpx

    settings = get_settings()
    url = (
        f"{settings.supabase_url}/rest/v1/session_documents"
        f"?session_id=eq.{session_id}&user_id=eq.{user_id}"
        f"&select=filename,content"
        f"&order=created_at.asc"
    )
    with httpx.Client() as client:
        resp = client.get(url, headers={
            "apikey": settings.supabase_anon_key,
            "Authorization": f"Bearer {token}",
        })
        if not resp.is_success:
            print(f"[chat] Failed to fetch session attachments: {resp.status_code}")
            return ""
        rows = resp.json()

    if not rows:
        return ""

    parts = ["ATTACHED DOCUMENTS (read by user request — use this as primary context):"]
    for idx, row in enumerate(rows, 1):
        content = row.get("content", "")
        filename = row.get("filename", "document")
        parts.append(f"\n[{idx}] {filename} ({len(content):,} chars)")
        parts.append(content)

    return "\n".join(parts)


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


@router.post("/sessions/create", response_model=SessionInfo)
async def create_session(
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
) -> SessionInfo:
    """Create an empty chat session (for attaching docs before first message)."""
    session = db.create_session(user.user_id, "New Chat", user_token=user.raw_token)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to create session")
    return SessionInfo(
        id=session["id"],
        title=session["title"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        is_archived=session.get("is_archived", False),
    )


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

    # Auto-rename "New Chat" sessions on first message
    if session.get("title") == "New Chat":
        title = request.message[:50] + ("..." if len(request.message) > 50 else "")
        db.update_session_title(session_id, user_id, title, user_token=token)

    # Validate image if provided
    settings = get_settings()
    image_mime = None
    image_bytes = None
    if request.image:
        image_mime, image_bytes = _validate_image(request.image, settings)

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

    # Store image in Supabase Storage if provided
    if image_bytes and image_mime:
        storage_path = _upload_image_to_storage(settings, user_id, image_bytes, image_mime)
        db.create_message_image(
            message_id=user_msg["id"],
            user_id=user_id,
            storage_path=storage_path,
            filename="image",
            mime_type=image_mime,
            file_size=len(image_bytes),
            user_token=token,
        )

    # Get conversation history for context
    history = db.get_session_messages(session_id, user_id, limit=20, user_token=token)

    # Build messages — for past messages with images, use placeholder text
    image_map = db.get_message_images(
        [m["id"] for m in history if m["role"] == "user"], user_id, user_token=token
    )
    hist_messages = []
    for msg in history:
        if msg["id"] == user_msg["id"] and request.image:
            # Current message with image — build multimodal format
            hist_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": request.message},
                    {"type": "image_url", "image_url": {"url": request.image}},
                ],
            })
        elif msg["id"] in image_map:
            # Past message that had an image — placeholder
            hist_messages.append({
                "role": msg["role"],
                "content": f"[Image was attached]\n{msg['content']}",
            })
        else:
            hist_messages.append({"role": msg["role"], "content": msg["content"]})

    messages = [_build_system_message(request.conversation_mode)] + hist_messages

    # Session attachments: inject full document text (highest priority context)
    attach_text = _fetch_session_attachments(session_id, user_id, token)
    if attach_text:
        messages[0]["content"] += f"\n\n{attach_text}"

    # Web search: inject real-time results into context
    search_results = []
    if await web_search.should_search(request.message):
        search_results = await web_search.search(request.message)
        if search_results:
            search_results = await web_search.enrich_with_full_content(search_results)
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

    total_chars = sum(
        len(m["content"]) if isinstance(m.get("content"), str) else sum(
            len(p.get("text", "")) for p in m.get("content", []) if isinstance(p, dict)
        )
        for m in messages
    )
    print(f"[chat] Sending {len(messages)} messages ({total_chars} chars) to inference")

    # Force thinking tier when image is attached (instant tier is text-only)
    # Voice conversation always uses instant tier (Qwen 3.5 4B); config via INFERENCE_INSTANT_*
    if request.image:
        inference_mode = "thinking"
        print("[chat] Image attached — forcing thinking tier for vision")
    elif request.conversation_mode:
        inference_mode = "instant"
    else:
        inference_mode = request.mode.value

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
    rag_metrics = rag.get_metrics(rag_chunks) if rag_chunks else None
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
        rag_metrics=rag_metrics,
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
                is_pinned=s.get("is_pinned", False),
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

    # Fetch image metadata for user messages
    user_msg_ids = [m["id"] for m in messages if m["role"] == "user"]
    image_map = db.get_message_images(user_msg_ids, user.user_id, user_token=token) if user_msg_ids else {}

    return [
        ChatMessage(
            id=m["id"],
            role=m["role"],
            content=m["content"],
            mode_used=m.get("mode_used"),
            inference_model=m.get("model_used"),
            reasoning_content=m.get("reasoning_content"),
            image_id=image_map[m["id"]]["id"] if m["id"] in image_map else None,
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
        is_pinned=updated.get("is_pinned", False),
    )


@router.patch("/sessions/{session_id}/pin", response_model=SessionInfo)
async def pin_session(
    session_id: str,
    request: SessionPinRequest,
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
) -> SessionInfo:
    """Pin or unpin a chat session."""
    token = user.raw_token
    session = db.get_session(session_id, user.user_id, user_token=token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    db.pin_session(session_id, user.user_id, request.is_pinned, user_token=token)
    updated = db.get_session(session_id, user.user_id, user_token=token)
    return SessionInfo(
        id=updated["id"],
        title=updated["title"],
        created_at=updated["created_at"],
        updated_at=updated["updated_at"],
        is_archived=updated.get("is_archived", False),
        is_pinned=updated.get("is_pinned", False),
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
    settings = get_settings()

    # Validate image if provided
    image_mime = None
    image_bytes = None
    if request.image:
        image_mime, image_bytes = _validate_image(request.image, settings)

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

    # Auto-rename "New Chat" sessions on first message
    if session.get("title") == "New Chat":
        title = request.message[:50] + ("..." if len(request.message) > 50 else "")
        db.update_session_title(session_id, user_id, title, user_token=token)

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

    # Store image in Supabase Storage if provided
    if image_bytes and image_mime:
        storage_path = _upload_image_to_storage(settings, user_id, image_bytes, image_mime)
        db.create_message_image(
            message_id=user_msg["id"],
            user_id=user_id,
            storage_path=storage_path,
            filename="image",
            mime_type=image_mime,
            file_size=len(image_bytes),
            user_token=token,
        )

    # Get conversation history
    history = db.get_session_messages(session_id, user_id, limit=20, user_token=token)

    # Build messages — for past messages with images, use placeholder text
    image_map = db.get_message_images(
        [m["id"] for m in history if m["role"] == "user"], user_id, user_token=token
    )
    hist_messages = []
    for msg in history:
        if msg["id"] == user_msg["id"] and request.image:
            # Current message with image — build multimodal format
            hist_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": request.message},
                    {"type": "image_url", "image_url": {"url": request.image}},
                ],
            })
        elif msg["id"] in image_map:
            # Past message that had an image — placeholder
            hist_messages.append({
                "role": msg["role"],
                "content": f"[Image was attached]\n{msg['content']}",
            })
        else:
            hist_messages.append({"role": msg["role"], "content": msg["content"]})

    messages = [_build_system_message(request.conversation_mode)] + hist_messages

    # Session attachments: inject full document text (highest priority context)
    attach_text = _fetch_session_attachments(session_id, user_id, token)
    if attach_text:
        messages[0]["content"] += f"\n\n{attach_text}"

    # Web search: run before streaming so results are ready
    search_results = []
    if await web_search.should_search(request.message):
        search_results = await web_search.search(request.message)
        if search_results:
            search_results = await web_search.enrich_with_full_content(search_results)
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

    total_chars = sum(
        len(m["content"]) if isinstance(m.get("content"), str) else sum(
            len(p.get("text", "")) for p in m.get("content", []) if isinstance(p, dict)
        )
        for m in messages
    )
    print(f"[chat] Sending {len(messages)} messages ({total_chars} chars) to inference")

    # Force thinking tier when image is attached (instant tier is text-only)
    # Voice conversation always uses instant tier (Qwen 3.5 4B); config via INFERENCE_INSTANT_*
    if request.image:
        inference_mode = "thinking"
        print("[chat] Image attached — forcing thinking tier for vision")
    elif request.conversation_mode:
        inference_mode = "instant"
    else:
        inference_mode = request.mode.value

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


# ── Image endpoints ──────────────────────────────────────────────────

@router.get("/images/{image_id}")
async def get_image(
    image_id: str,
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
):
    """Serve a chat image from Supabase Storage."""
    settings = get_settings()
    # Fetch image metadata (RLS ensures user can only see their own)
    result = db._request(
        "GET",
        f"message_images?id=eq.{image_id}&user_id=eq.{user.user_id}&select=storage_path,mime_type",
        user_token=user.raw_token,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Image not found")

    row = result[0]
    image_bytes = _download_image_from_storage(settings, row["storage_path"])
    return Response(content=image_bytes, media_type=row["mime_type"])


@router.get("/sessions/{session_id}/images")
async def get_session_images(
    session_id: str,
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
):
    """Get all image metadata for messages in a session."""
    token = user.raw_token
    session = db.get_session(session_id, user.user_id, user_token=token)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.get_session_messages(session_id, user.user_id, user_token=token)
    msg_ids = [m["id"] for m in messages if m["role"] == "user"]
    if not msg_ids:
        return {}

    images = db.get_message_images(msg_ids, user.user_id, user_token=token)
    return images
