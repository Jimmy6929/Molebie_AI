"""
Chat endpoints for the Gateway API.
"""

import asyncio
import base64
import json
import re
import secrets
import time
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import Response, StreamingResponse

from app.config import get_settings
from app.middleware.auth import JWTPayload, get_current_user
from app.models.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    InferenceMetadata,
    SessionInfo,
    SessionListResponse,
    SessionPinRequest,
    SessionRenameRequest,
    TTSRequest,
)
from app.services.consistency import is_verifiable_query, vote_with_self_consistency
from app.services.database import DatabaseService, get_database_service
from app.services.inference import InferenceService, get_inference_service
from app.services.judge import get_grounding_judge
from app.services.memory import get_memory_service
from app.services.metrics_registry import RequestRecord, get_metrics_registry
from app.services.rag import RAGService, compute_retrieval_confidence, get_rag_service
from app.services.selfcheck import get_selfcheck_service
from app.services.sse_split import split_oversized_sse_delta
from app.services.storage import get_storage_service
from app.services.streaming_think_filter import ThinkBlockFilter
from app.services.summarizer import get_summariser_service
from app.services.tools import TOOL_SCHEMAS, ToolExecutor
from app.services.verification import get_chain_of_verification
from app.services.web_search import (
    WebSearchService,
    get_web_search_service,
    looks_like_search_query,
)


async def _run_tracked_task(name: str, coro):
    """Wrap a fire-and-forget coroutine with mark_task lifecycle so the
    Tasks panel in the live monitor reflects what's happening."""
    registry = get_metrics_registry()
    await registry.mark_task(name, "start")
    try:
        await coro
        await registry.mark_task(name, "done")
    except Exception as exc:
        print(f"[chat] Background task {name} failed: {type(exc).__name__}: {exc}")
        await registry.mark_task(name, "fail")


@lru_cache(maxsize=4)
def _load_prompt_template(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    settings = get_settings()
    prompt_path = Path(__file__).parent.parent.parent / settings.prompt_dir / f"{name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "You are a helpful AI assistant.\n\nCurrent date: {current_date}"


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# High-precision patterns that indicate the user explicitly wants notes-lookup
# behavior — overrides the confidence-based template choice into strict-RAG
# mode even when retrieval confidence is NONE or weak (LOW). Kept narrow on
# purpose: this is *not* a generic intent classifier, only an escape hatch
# for queries that literally say "from my notes" / "what did I write".
_EXPLICIT_LOOKUP_PATTERNS = [
    re.compile(r"\bin my notes\b", re.IGNORECASE),
    re.compile(r"\bfrom my notes\b", re.IGNORECASE),
    re.compile(r"\bwhat did i write\b", re.IGNORECASE),
    re.compile(r"\bdid i (write|mention|note|say)\b", re.IGNORECASE),
    re.compile(r"\bfind (the )?notes?\b", re.IGNORECASE),
]


def _explicit_lookup_phrasing(message: str) -> bool:
    """True iff the user message uses one of the high-precision lookup
    phrasings. Used to override into strict notes-lookup mode regardless of
    retrieval confidence.
    """
    if not message:
        return False
    return any(p.search(message) for p in _EXPLICIT_LOOKUP_PATTERNS)


def _routing_mode(confidence: str, explicit_lookup: bool) -> str:
    """Pick the prompt template for a (confidence, explicit-lookup) tuple.

    HIGH/MODERATE retrieval → ``lookup`` (system_rag.txt, strict grounding).
    LOW/NONE retrieval → ``generative`` (system_generative.txt, permissive).
    ``explicit_lookup=True`` overrides into lookup mode so the user's
    explicit "from my notes" intent wins regardless of confidence.

    Why LOW falls through to generative: forcing strict citation mode on
    weak retrievals made the model either refuse unnecessarily or cite
    irrelevant chunks just to satisfy the prompt's RULES. Sending LOW to
    the generative template lets the model use the chunks as background
    when they help and answer from general knowledge otherwise — see the
    LOW+generative directive in ``_confidence_directive``.
    """
    if explicit_lookup:
        return "lookup"
    if confidence in ("HIGH", "MODERATE"):
        return "lookup"
    return "generative"   # LOW or NONE


def _confidence_directive(
    confidence: str,
    chunks: list[dict[str, Any]],
    mode: str = "lookup",
    use_tool_calling: bool = False,
    factual_query: bool = False,
) -> str | None:
    """Return a directive to append to the system prompt, or None.

    Two cases need a directive on top of the base template:

    1. Generative mode + NONE confidence + factual query + tools enabled:
       nudge the model to offer a web search before answering. (Without
       this nudge a 7B local model often defaults to "I cannot access X"
       even when web_search is in its tool list.) For non-factual / casual
       turns, no directive — the system prompt already handles "answer
       directly, no preamble".

    2. Lookup mode + LOW confidence: remind the model that the retrieval
       was weak — cite [S#] only when actually used, and say so plainly
       if the evidence doesn't answer the question.

    Crucially, the literal word "REFUSE" never appears in any directive —
    the model was previously mimicking that token from its own prompt.
    """
    if mode == "generative" and confidence == "NONE":
        if use_tool_calling and factual_query:
            return (
                "The user's notes don't cover this. If you're confident, "
                "answer from general knowledge. If you're uncertain on "
                "specifics, offer to search the web before answering "
                "(e.g. \"Want me to look this up online?\")."
            )
        return None
    if confidence == "LOW" and mode == "lookup":
        # Reachable only via the explicit-lookup override (user literally
        # asked "from my notes"). Strict grounding stays on; just remind
        # the model the evidence is thin so it doesn't fabricate citations.
        return (
            "The retrieved evidence has weak relevance. Cite [S#] only for "
            "claims you actually drew from it; if it doesn't answer the "
            "question, say so plainly and offer alternatives."
        )
    if confidence == "LOW" and mode == "generative":
        # New default LOW path (post-routing-refinement): use chunks as
        # background context, fall back to general knowledge when they
        # don't fit, cite only for facts that actually came from a note.
        return (
            "Some related notes were found but may not directly answer "
            "this question. Use them as background if helpful, but rely "
            "on general knowledge when they don't fit. Cite [S#] only "
            "when a specific fact actually came from a note."
        )
    return None


def _build_evidence_summary(
    search_results: list[dict[str, Any]],
    rag_chunks: list[dict[str, Any]],
) -> str:
    """Generate an evidence meta-block for the system message.

    Reports what evidence is available — content depth, source types,
    retrieval completeness — so the model can self-calibrate confidence.
    Does NOT instruct the model how to use that evidence; that's the
    template's job. Mixing the two layers caused the documented conflict
    between "Use ONLY context" (template) and "Use all available evidence"
    (this block) — the model would burn thinking tokens reconciling them.

    Returns an empty string when there is nothing to report — callers
    must guard the append. Previously this returned a "No web... No
    matches..." block that primed the model to think in [S#]/[W#] tags
    even on casual turns with no retrieval.
    """
    if not search_results and not rag_chunks:
        return ""

    lines = ["AVAILABLE EVIDENCE (informational — use only if it answers the question):"]

    # ── Web results ──
    if search_results:
        domains = {r.get("domain", "") for r in search_results if r.get("domain")}
        full_page = sum(1 for r in search_results if r.get("content_source") == "full_page")
        snippet_only = len(search_results) - full_page

        type_counts: dict[str, int] = {}
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
        quality_counts: dict[str, int] = {"HIGH": 0, "MODERATE": 0, "WEAK": 0}
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

    lines.append(f"• Retrieval: {completeness}")

    return "\n".join(lines)


_CITATION_RE = re.compile(r"\[S(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Patterns flagging a "specific factual claim" — numbers, URLs, dates.
# Conservative on purpose: false-positives generate annoying footnotes.
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_FOOTNOTE_TEXT = (
    "\n\n*Note: Some claims in this response could not be verified against "
    "your notes.*"
)


def _has_specific_claim(sentence: str) -> bool:
    """True when the sentence contains a number, URL, or date — the kinds
    of facts a small model is most likely to fabricate."""
    return bool(
        _NUMBER_RE.search(sentence)
        or _URL_RE.search(sentence)
        or _DATE_RE.search(sentence)
    )


def _claim_supported_by_chunk(sentence: str, chunk_content: str) -> bool:
    """Cheap fuzzy match: do enough content tokens from the sentence appear
    in the chunk? We strip the [S#] marker and stop-words-ish short tokens
    via the TOKEN_RE filter (length >= 3, alphanumeric).

    This is intentionally lightweight — anything that needs NLI-grade
    grounding belongs in Phase 3 (CoVe / SelfCheckGPT), not here.
    """
    sentence_clean = _CITATION_RE.sub("", sentence)
    sent_tokens = {t.lower() for t in _TOKEN_RE.findall(sentence_clean)}
    if not sent_tokens:
        return True   # nothing to verify — give the benefit of the doubt
    chunk_tokens = {t.lower() for t in _TOKEN_RE.findall(chunk_content or "")}
    if not chunk_tokens:
        return False
    overlap = sent_tokens & chunk_tokens
    # ≥40% of the sentence's content tokens must appear in the cited chunk.
    # Tuned for short chunks (512 chars) — relax to 0.3 if false-negatives
    # become noisy in practice.
    return len(overlap) / max(1, len(sent_tokens)) >= 0.4


def _validate_citations(
    response_text: str,
    num_sources: int,
    rag_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Parse ``[S#]`` citations and check them against the available source
    count + (optionally) the cited chunks themselves.

    Phase 1.9 Stage A added the index-validity check. Phase 2.4 adds:
      * per-cited-claim fuzzy match (does the claim's content actually
        appear in the chunk it cites?)
      * unsupported-claim flagging (sentence has a number/URL/date but
        no [S#] tag → the model probably invented it)

    Returns a dict suitable for logging / inference-metadata.
    """
    cited_indices = {int(m) for m in _CITATION_RE.findall(response_text)}
    valid = set(range(1, num_sources + 1))
    invalid = cited_indices - valid

    chunks_by_index: dict[int, dict[str, Any]] = {}
    if rag_chunks:
        for i, c in enumerate(rag_chunks, 1):
            chunks_by_index[i] = c

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(response_text) if s.strip()]
    weak_citations: list[dict[str, Any]] = []
    unsupported_claims: list[str] = []

    for sentence in sentences:
        sentence_citations = [int(m) for m in _CITATION_RE.findall(sentence)]
        if sentence_citations:
            # If the sentence cites at least one valid source, treat it as
            # supported when ANY cited chunk overlaps. Catch the case where
            # the model cites the wrong chunk for a real claim.
            any_supported = False
            for idx in sentence_citations:
                chunk = chunks_by_index.get(idx)
                if chunk and _claim_supported_by_chunk(sentence, chunk.get("content", "")):
                    any_supported = True
                    break
            if not any_supported:
                weak_citations.append(
                    {"sentence": sentence[:160], "cited": sentence_citations}
                )
        elif _has_specific_claim(sentence):
            unsupported_claims.append(sentence[:160])

    return {
        "num_sources": num_sources,
        "cited_count": len(cited_indices),
        "cited_indices": sorted(cited_indices),
        "invalid_indices": sorted(invalid),
        "no_citations": num_sources > 0 and not cited_indices,
        "weak_citations": weak_citations,
        "unsupported_claims": unsupported_claims,
        # Heuristic — append the user-visible footnote when there are
        # un-cited specific facts. Wrong citations are logged but not
        # surfaced to the user since the fix is for the operator to read.
        "needs_footnote": bool(unsupported_claims),
    }


def _validate_response_sources(
    response_text: str,
    search_results: list[dict[str, Any]],
    rag_chunks: list[dict[str, Any]],
    routing_mode: str = "lookup",
) -> dict[str, Any] | None:
    """Log whether the response referenced any of the provided sources, and —
    when RAG was used in lookup mode — validate ``[S#]`` citations against the
    number of available sources.

    Returns the citation-validation dict (or None when no RAG was used or we're
    in generative mode) so callers can attach it to inference metadata.

    Generative-mode skip rationale: in generative mode we explicitly tell the
    model that notes-citation isn't required (notes are background, general
    knowledge is permitted). Running ``_validate_citations`` against [S#] tags
    would mis-flag legitimate uncited claims as "unsupported" and append a
    misleading "could not be verified against your notes" footnote.
    """
    if not search_results and not rag_chunks:
        return None
    if routing_mode == "generative":
        # Still log whether sources were referenced, but skip [S#] validation.
        return None

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

    citation_report: dict[str, Any] | None = None
    if rag_chunks:
        citation_report = _validate_citations(
            response_text, len(rag_chunks), rag_chunks=rag_chunks,
        )
        if citation_report["invalid_indices"]:
            print(
                f"[chat] Citation warning: invalid [S#] references "
                f"{citation_report['invalid_indices']} "
                f"(only {citation_report['num_sources']} sources available)"
            )
        elif citation_report["no_citations"]:
            print(
                f"[chat] Citation warning: response cites none of the "
                f"{citation_report['num_sources']} provided sources"
            )
        else:
            print(
                f"[chat] Citations OK: {citation_report['cited_count']} of "
                f"{citation_report['num_sources']} sources referenced"
            )
        if citation_report["weak_citations"]:
            print(
                f"[chat] Citation weak: {len(citation_report['weak_citations'])} "
                "sentences cite a chunk that doesn't share their key terms"
            )
        if citation_report["unsupported_claims"]:
            print(
                f"[chat] Citation: {len(citation_report['unsupported_claims'])} "
                "sentences contain numbers/URLs/dates without a [S#] tag"
            )
    return citation_report


async def _run_tool_loop(
    inference: Any,
    executor: ToolExecutor,
    messages: list[dict[str, Any]],
    mode: str,
    enable_thinking: bool | None,
    max_iterations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tool calling: model emits ``tool_calls`` → we execute → feed results
    back as ``tool`` role messages → call inference again. Loops until the
    model returns a plain text response or we hit ``max_iterations``.

    Returns ``(final_inference_result, tool_call_log)``. The log captures
    every (name, args, result) triple so the response payload can surface
    what the model actually did.
    """
    tool_call_log: list[dict[str, Any]] = []
    working_messages = list(messages)

    for iteration in range(max_iterations):
        result = await inference.generate_response(
            messages=working_messages,
            mode=mode,
            enable_thinking=enable_thinking,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        tool_calls = result.get("tool_calls") or []
        if not tool_calls:
            return result, tool_call_log

        # Append the model's assistant turn (which contains tool_calls)
        # so the backend's chat template threads tool results correctly.
        # Sanitize content via _finalize_assistant_text so any <think>
        # blocks (closed or truncated-open) don't leak into the next
        # iteration's prompt.
        working_messages.append(
            {
                "role": "assistant",
                "content": _finalize_assistant_text(result.get("content") or ""),
                "tool_calls": tool_calls,
            }
        )

        # Execute each tool call, append result as a tool-role message.
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError as exc:
                args = {}
                tool_result = {"ok": False, "error": f"arguments JSON invalid: {exc}"}
            else:
                tool_result = await executor.execute(name, args)

            tool_call_log.append(
                {"name": name, "args": args, "result": tool_result, "iteration": iteration}
            )
            print(f"[chat] Tool call #{iteration}: {name}({args}) → ok={tool_result.get('ok')}")

            working_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "content": json.dumps(tool_result),
                }
            )

    # Hit iteration cap — return whatever the last call produced, plus a
    # sentinel so the caller can flag this in the response metadata.
    print(f"[chat] Tool loop hit max_iterations={max_iterations}; "
          "returning last result.")
    return result, tool_call_log


def _maybe_append_footnote(
    response_text: str,
    citation_report: dict[str, Any] | None,
) -> str:
    """When the citation report flags un-cited factual claims, append the
    footnote. Idempotent — won't duplicate if the footnote text is already
    present (defensive against background-task reprocessing)."""
    if not citation_report or not citation_report.get("needs_footnote"):
        return response_text
    if _FOOTNOTE_TEXT.strip() in response_text:
        return response_text
    return response_text + _FOOTNOTE_TEXT


_TOOLS_HINT = (
    "TOOLS AVAILABLE: web_search (live web), search_notes (your saved notes), "
    "calculate (arithmetic), get_current_time (current date/time). "
    "Call them when needed; do not claim you cannot search the web — you can."
)


def _build_system_message(
    conversation_mode: bool = False,
    summary: str | None = None,
    memories_text: str | None = None,
    evidence: str | None = None,
    confidence: str = "NONE",
    explicit_lookup: bool = False,
    use_tool_calling: bool = False,
) -> dict:
    """Build the system message with confidence-driven prompt routing.

    Routing rule (see ``_routing_mode``):
      - HIGH/MODERATE/LOW retrieval → ``system_rag.txt`` (strict, cite-when-used)
      - NONE retrieval → ``system_generative.txt`` (permissive, general
        knowledge + web allowed)
      - ``explicit_lookup=True`` overrides into strict mode regardless of
        confidence so the user's explicit "from my notes" intent wins.

    When ``use_tool_calling`` is True (and not voice mode), a one-line
    "TOOLS AVAILABLE" hint is appended. Local Qwen-class models otherwise
    have no prior that the tool list shipped over the wire is real and
    will refuse with "I cannot access X".

    Conversation (voice) mode bypasses routing/tools entirely and uses
    ``system_voice.txt``.
    """
    today_str = date.today().strftime("%A, %B %d, %Y")
    evidence_text = evidence if evidence else "(no retrieved evidence — none of the user's notes match this query closely)"

    if conversation_mode:
        prompt_template = _load_prompt_template("system_voice")
        content = prompt_template.format(current_date=today_str)
    else:
        mode = _routing_mode(confidence, explicit_lookup)
        template_name = "system_rag" if mode == "lookup" else "system_generative"
        prompt_template = _load_prompt_template(template_name)
        content = prompt_template.format(
            current_date=today_str,
            evidence=evidence_text,
        )
        if use_tool_calling:
            content += f"\n\n{_TOOLS_HINT}"

    if summary:
        content += f"\n\nCONVERSATION SUMMARY (earlier messages condensed):\n{summary}"

    if memories_text:
        content += f"\n\n{memories_text}"

    return {"role": "system", "content": content}


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


def _extract_thinking(content: str) -> str | None:
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


def _finalize_assistant_text(raw: str) -> str:
    """Single chokepoint for any text leaving inference toward DB / UI / next-turn.

    1. Closes orphan ``<think>`` openers (truncation at max_tokens leaves these
       — without a closing tag the regex in ``_strip_thinking`` can't match
       and the raw reasoning would leak to the user).
    2. Strips ``<think>...</think>`` blocks.
    3. Trims.

    Idempotent — safe to call twice. Use this at every site where assistant
    content crosses into a place that mustn't see reasoning: user-visible
    output, DB persistence, or the prior-turn context fed to the next
    inference call (notably the tool loop's working_messages).
    """
    if not raw:
        return ""
    if "<think>" in raw and "</think>" not in raw:
        raw = raw + "</think>"
    return _strip_thinking(raw)


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/sessions/create", response_model=SessionInfo)
async def create_session(
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
) -> SessionInfo:
    """Create an empty chat session (for attaching docs before first message)."""
    session = await db.create_session(user.user_id, "New Chat")
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

    # Get or create session
    if request.session_id:
        session = await db.get_session(request.session_id, user_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
    else:
        title = request.message[:50] + "..." if len(request.message) > 50 else request.message
        session = await db.create_session(user_id, title)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session"
            )

    session_id = session["id"]
    session_summary = session.get("summary")
    summary_msg_count = session.get("summary_message_count", 0)

    # Auto-rename "New Chat" sessions on first message
    if session.get("title") == "New Chat":
        title = request.message[:50] + ("..." if len(request.message) > 50 else "")
        await db.update_session_title(session_id, user_id, title)

    # Validate image if provided
    settings = get_settings()
    image_mime = None
    image_bytes = None
    if request.image:
        image_mime, image_bytes = _validate_image(request.image, settings)

    # Store user message
    user_msg = await db.create_message(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=request.message,
    )
    if not user_msg:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store message"
        )

    # Store image locally if provided
    if image_bytes and image_mime:
        storage_path = get_storage_service().upload_image(user_id, image_bytes, image_mime)
        await db.create_message_image(
            message_id=user_msg["id"],
            user_id=user_id,
            storage_path=storage_path,
            filename="image",
            mime_type=image_mime,
            file_size=len(image_bytes),
        )

    # Get conversation history (reduced window when summary exists)
    history_limit = settings.summary_recent_messages if session_summary else 20
    history = await db.get_session_messages(session_id, user_id, limit=history_limit)

    # Build messages — for past messages with images, use placeholder text
    image_map = await db.get_message_images(
        [m["id"] for m in history if m["role"] == "user"], user_id
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

    # Retrieve relevant user memories for context injection
    memory_service = get_memory_service()
    memories_text = ""
    if memory_service.enabled and not request.conversation_mode:
        try:
            memories = await memory_service.retrieve_relevant_memories(user_id, request.message)
            memories_text = memory_service.format_memories_for_context(memories)
        except Exception as exc:
            print(f"[chat] Memory retrieval failed: {exc}")

    # RAG retrieval runs BEFORE system-message build so chunks can be slotted
    # into the strict-grounding template (system_rag.txt) instead of being
    # appended after a general prompt — this gives the model unambiguous
    # citation rules and a defined refusal path. RAG failures degrade to
    # no-context retrieval rather than failing the whole request — the
    # model can still answer from memory + chat history.
    rag_chunks = []
    if not request.conversation_mode:
        if not rag.enabled:
            print("[chat] RAG disabled — skipping document retrieval")
        elif await rag.user_has_documents(user_id):
            try:
                rag_chunks = await rag.retrieve_context(
                    user_id, request.message, conversation_context=hist_messages,
                )
            except Exception as exc:
                print(f"[chat] RAG retrieval failed — continuing without context: "
                      f"{type(exc).__name__}: {exc}")
                rag_chunks = []

    rag_context_text = rag.format_context(rag_chunks) if rag_chunks else None
    rag_confidence = compute_retrieval_confidence(rag_chunks) if rag_chunks else "NONE"
    explicit_lookup = _explicit_lookup_phrasing(request.message)
    routing_mode = _routing_mode(rag_confidence, explicit_lookup)
    if rag_chunks or explicit_lookup:
        print(f"[chat] Retrieval confidence: {rag_confidence} → mode: {routing_mode}"
              + (" (explicit lookup phrasing)" if explicit_lookup else ""))

    # Compute tool-calling availability up-front so it can be advertised in
    # the system prompt (otherwise the model has no prior that web_search /
    # search_notes / etc. are real and refuses with "I cannot access X").
    use_tool_calling = (
        settings.tool_calling_enabled
        and not request.conversation_mode
        and not request.image
    )

    messages = [_build_system_message(
        request.conversation_mode,
        summary=session_summary,
        memories_text=memories_text,
        evidence=rag_context_text,
        confidence=rag_confidence,
        explicit_lookup=explicit_lookup,
        use_tool_calling=use_tool_calling,
    )] + hist_messages

    # Confidence-driven directives: in generative mode on a factual gap with
    # tools available, nudge the model to offer a web search; in lookup mode
    # with LOW confidence, remind it to cite-when-used and admit gaps. HIGH/
    # MODERATE/casual turns rely on the base template alone.
    directive = _confidence_directive(
        rag_confidence,
        rag_chunks,
        mode=routing_mode,
        use_tool_calling=use_tool_calling,
        factual_query=looks_like_search_query(request.message),
    )
    if directive:
        messages[0]["content"] += f"\n\n{directive}"

    # Session attachments: inject full document text (highest priority context)
    attach_text = await db.fetch_session_attachments_text(session_id, user_id)
    if attach_text:
        messages[0]["content"] += f"\n\n{attach_text}"

    # Web search: inject real-time results into context (continues the EVIDENCE
    # block under the [W#] namespace — see web_search.format_results_for_context).
    search_results = []
    if request.web_search:
        search_results = await web_search.search(request.message)
        if search_results:
            search_results = await web_search.enrich_with_full_content(search_results)
            context_text = web_search.format_results_for_context(search_results)
            messages[0]["content"] += f"\n\n{context_text}"

    # Inject evidence summary so the model can self-calibrate confidence
    evidence_summary = _build_evidence_summary(search_results, rag_chunks)
    if evidence_summary:
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

    # Force CoT off when RAG context is present on the thinking tier — small
    # Qwen models burn thousands of thinking tokens "in circles" on retrieval
    # Q&A and the strict-grounding template doesn't need reasoning to follow.
    enable_thinking_override: bool | None = None
    if (rag_chunks
            and inference_mode in ("thinking", "thinking_harder")
            and settings.inference_thinking_auto_disable_for_rag):
        enable_thinking_override = False
        print(f"[chat] RAG present on {inference_mode} tier — disabling CoT")

    # Decide which inference path to take. Tool calling and self-consistency
    # are mutually exclusive in this first cut — tool calling produces a
    # deterministic answer (delegating to a tool) so the vote-and-pick
    # approach doesn't add value on top, and the cost of stacking both
    # (N samples × M tool iterations) isn't worth it without measured gain.
    # ``use_tool_calling`` is computed earlier (before the system message
    # build) so the TOOLS AVAILABLE hint can be advertised to the model.
    use_self_consistency = (
        not use_tool_calling
        and settings.self_consistency_enabled
        and not request.conversation_mode
        and not request.image
        and is_verifiable_query(request.message)
    )

    # Generate AI response — instrumented for /metrics/live
    # Non-streaming path: TTFT is approximated as total_ms because we don't
    # see individual tokens. Monitor reads the series with that caveat.
    registry = get_metrics_registry()
    await registry.mark_active(inference_mode)
    t_start = time.perf_counter()
    _metrics_error: str | None = None
    try:
        if use_tool_calling:
            print(f"[chat] Tool calling enabled — running tool loop "
                  f"(max_iterations={settings.tool_calling_max_iterations})")
            executor = ToolExecutor(
                user_id=user_id,
                rag_service=rag,
                web_search_service=web_search,
                settings=settings,
            )
            inference_result, tool_call_log = await _run_tool_loop(
                inference=inference,
                executor=executor,
                messages=messages,
                mode=inference_mode,
                enable_thinking=enable_thinking_override,
                max_iterations=settings.tool_calling_max_iterations,
            )
            inference_result["tool_calls_made"] = tool_call_log
        elif use_self_consistency:
            print(f"[chat] Self-consistency: voting across "
                  f"{settings.self_consistency_samples} samples")
            vote = await vote_with_self_consistency(
                inference=inference,
                messages=messages,
                mode=inference_mode,
                settings=settings,
                enable_thinking=enable_thinking_override,
            )
            inference_result = {
                "content": vote["content"],
                "reasoning_content": None,
                "tokens_used": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "mode_used": inference_mode,
                "model": settings.get_model_for_mode(inference_mode),
                "finish_reason": "stop",
                "fallback_used": False,
                "self_consistency": {
                    "samples": len(vote["samples"]),
                    "vote_counts": vote["vote_counts"],
                    "confident": vote["confident"],
                    "early_stopped": vote["early_stopped"],
                },
            }
        else:
            inference_result = await inference.generate_response(
                messages=messages,
                mode=inference_mode,
                enable_thinking=enable_thinking_override,
            )
    except Exception as exc:
        _metrics_error = type(exc).__name__
        raise
    finally:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        await registry.mark_inactive(inference_mode)
        ok = _metrics_error is None
        result = locals().get("inference_result") if ok else None
        await registry.record(RequestRecord(
            ended_at=time.time(),
            tier=inference_mode,
            model=(result.get("model") if result else None),
            streaming=False,
            ok=ok,
            ttft_ms=total_ms,              # approximation for non-streaming
            tpot_ms=None,
            total_ms=total_ms,
            completion_tokens=(result.get("completion_tokens") if result else None),
            prompt_tokens=(result.get("prompt_tokens") if result else None),
            finish_reason=(result.get("finish_reason") if result else None),
            fallback=bool(result.get("fallback_used")) if result else False,
            error_type=_metrics_error,
        ))

    # Store assistant response (strip thinking blocks, persist reasoning separately)
    raw = inference_result["content"]
    reasoning = inference_result.get("reasoning_content") or _extract_thinking(raw)
    clean_content = _finalize_assistant_text(raw)
    assistant_msg = await db.create_message(
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=clean_content,
        mode_used=inference_result["mode_used"],
        tokens_used=inference_result.get("tokens_used"),
        reasoning_content=reasoning,
    )
    if not assistant_msg:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store response"
        )

    # Persist web search source links
    if search_results:
        await db.create_message_sources(
            assistant_msg["id"],
            [{"title": r["title"], "url": r["url"]} for r in search_results],
        )

    # Log whether the response referenced provided sources, and validate
    # any [S#] citations against the available source count + per-claim
    # fuzzy match (task 2.4).
    citation_report = _validate_response_sources(
        clean_content, search_results, rag_chunks, routing_mode=routing_mode
    )

    # Phase 3.2 — Grounding judge (reranker yes/no head). Cheap gate that
    # runs before CoVe: ~one reranker forward pass per claim, no LLM
    # calls. Catches off-topic fabrications fast; CoVe still handles the
    # harder numeric-substitution cases. Default off.
    _post_metrics = get_metrics_registry()
    judge_report = None
    if settings.judge_enabled and rag_chunks:
        try:
            from app.services.reranker import get_reranker
            judge = get_grounding_judge(get_reranker(), settings)
            t_judge = time.perf_counter()
            judge_result = judge.judge(response=clean_content, rag_chunks=rag_chunks)
            await _post_metrics.record_subsystem(
                "verify.judge",
                latency_ms=(time.perf_counter() - t_judge) * 1000.0,
                ok=True,
                note=f"{judge_result.flagged_count}/{judge_result.scored_count} flagged"
                     if judge_result.applied else "skipped",
            )
            if judge_result.applied:
                print(
                    f"[chat] Judge: {judge_result.flagged_count}/"
                    f"{judge_result.scored_count} claims below threshold "
                    f"{judge_result.threshold:.2f}"
                )
                clean_content = judge_result.revised_response
                if judge_result.flagged_count and citation_report is not None:
                    citation_report["needs_footnote"] = True
                    citation_report["judge_flagged"] = judge_result.flagged_count
            else:
                print(f"[chat] Judge skipped: {judge_result.skipped_reason}")
            judge_report = {
                "applied": judge_result.applied,
                "scored_count": judge_result.scored_count,
                "flagged_count": judge_result.flagged_count,
                "threshold": judge_result.threshold,
                "skipped_reason": judge_result.skipped_reason,
            }
        except Exception as exc:
            # Judge is best-effort — never let it break the chat path.
            print(f"[chat] Judge raised {type(exc).__name__}: {exc}")
            judge_report = {"applied": False, "error": f"{type(exc).__name__}"}

    # Phase 3.1 — Chain-of-Verification post-processor. Decomposes long
    # factual responses into atomic claims, verifies each in a fresh
    # inference context against the cited chunk, and inserts a [?] marker
    # next to unsupported claims. Default off (cove_enabled=False) — flips
    # to ~5–9× extra inference calls on applicable responses, so wait for
    # eval signal before turning on.
    cove_report = None
    if settings.cove_enabled and rag_chunks:
        cove = get_chain_of_verification(inference, settings)
        cove_result = await cove.maybe_verify(
            response=clean_content,
            query=request.message,
            rag_chunks=rag_chunks,
            mode=inference_result.get("mode_used", "instant"),
        )
        if cove_result.applied:
            print(
                f"[chat] CoVe: {cove_result.unsupported_count}/"
                f"{cove_result.claims_checked} claims flagged unsupported"
                + (f", {cove_result.verify_json_failures} JSON failures"
                   if cove_result.verify_json_failures else "")
                + (" (rule-based decompose fallback)"
                   if cove_result.decompose_fallback else "")
            )
            clean_content = cove_result.revised_response
            # Promote unsupported claims to the existing footnote pathway
            # so the user sees one consolidated note rather than two.
            if cove_result.unsupported_count and citation_report is not None:
                citation_report["needs_footnote"] = True
                citation_report["cove_unsupported"] = cove_result.unsupported_count
        else:
            print(f"[chat] CoVe skipped: {cove_result.skipped_reason}")
        cove_report = {
            "applied": cove_result.applied,
            "claims_checked": cove_result.claims_checked,
            "unsupported_count": cove_result.unsupported_count,
            "skipped_reason": cove_result.skipped_reason,
            "decompose_fallback": cove_result.decompose_fallback,
            "verify_json_failures": cove_result.verify_json_failures,
        }

    # Phase 3.3 — SelfCheckGPT (reference-free). Fires only when there's
    # NO RAG context: with chunks, CoVe + Judge are strictly better
    # because they have ground truth to compare against. Default off —
    # N extra full-response generations is expensive.
    selfcheck_report = None
    if settings.selfcheck_enabled and not rag_chunks:
        try:
            selfcheck = get_selfcheck_service(inference, settings)
            async with _post_metrics.subsystem_timer("verify.selfcheck"):
                sc_result = await selfcheck.maybe_check(
                    response=clean_content,
                    messages_for_resample=messages,
                    rag_chunks=rag_chunks,
                    mode=inference_result.get("mode_used", "instant"),
                )
            if sc_result.applied:
                print(
                    f"[chat] SelfCheck ({sc_result.backend}): "
                    f"{sc_result.flagged_count}/{sc_result.sentences_checked} "
                    f"sentences inconsistent across {sc_result.samples_used} samples"
                )
                clean_content = sc_result.revised_response
                if sc_result.flagged_count and citation_report is None:
                    # No citation report exists for non-RAG paths. Build a
                    # minimal one so _maybe_append_footnote fires.
                    citation_report = {
                        "needs_footnote": True,
                        "selfcheck_flagged": sc_result.flagged_count,
                    }
                elif sc_result.flagged_count and citation_report is not None:
                    citation_report["needs_footnote"] = True
                    citation_report["selfcheck_flagged"] = sc_result.flagged_count
            else:
                print(f"[chat] SelfCheck skipped: {sc_result.skipped_reason}")
            selfcheck_report = {
                "applied": sc_result.applied,
                "samples_used": sc_result.samples_used,
                "sentences_checked": sc_result.sentences_checked,
                "flagged_count": sc_result.flagged_count,
                "threshold": sc_result.threshold,
                "backend": sc_result.backend,
                "skipped_reason": sc_result.skipped_reason,
            }
        except Exception as exc:
            print(f"[chat] SelfCheck raised {type(exc).__name__}: {exc}")
            selfcheck_report = {"applied": False, "error": f"{type(exc).__name__}"}

    # Surface un-cited factual claims as a user-visible footnote. Mutates
    # both the in-memory message and the persisted DB content so the next
    # turn's history reflects what the user actually saw.
    final_content = _maybe_append_footnote(clean_content, citation_report)
    if final_content != clean_content:
        await db.update_message_content(assistant_msg["id"], user_id, final_content)
        assistant_msg["content"] = final_content
        clean_content = final_content

    # Background tasks: summarisation and memory extraction (non-blocking)
    total_msg_count = len(history) + 1  # +1 for the assistant message just saved
    summariser = get_summariser_service()
    if summariser.should_summarise(total_msg_count + summary_msg_count, summary_msg_count):
        asyncio.create_task(_run_tracked_task(
            "task.summary",
            summariser.summarise_session(session_id, user_id),
        ))

    if memory_service.should_extract(total_msg_count):
        recent_for_extraction = hist_messages + [
            {"role": "assistant", "content": clean_content}
        ]
        asyncio.create_task(_run_tracked_task(
            "task.memory_extract",
            memory_service.extract_and_store(
                session_id, user_id, recent_for_extraction
            ),
        ))

    # Build inference metadata for the response. Merge the citation report
    # (task 1.9 Stage A) into rag_metrics — it's a free-form dict, so no
    # schema/model migration is required to surface this to the frontend.
    rag_metrics = rag.get_metrics(rag_chunks) if rag_chunks else None
    if citation_report is not None:
        rag_metrics = {**(rag_metrics or {}), "citations": citation_report}
    if cove_report is not None:
        rag_metrics = {**(rag_metrics or {}), "verification": cove_report}
    if judge_report is not None:
        rag_metrics = {**(rag_metrics or {}), "judge": judge_report}
    if selfcheck_report is not None:
        rag_metrics = {**(rag_metrics or {}), "selfcheck": selfcheck_report}
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

    sources_list = [{"title": r["title"], "url": r["url"]} for r in search_results] if search_results else None

    return ChatResponse(
        session_id=session_id,
        message=ChatMessage(
            id=assistant_msg["id"],
            role="assistant",
            content=assistant_msg["content"],
            mode_used=assistant_msg.get("mode_used"),
            inference_model=inference_result.get("model"),
            sources=sources_list,
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
    sessions = await db.list_sessions(user.user_id)
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


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessage])
async def get_session_messages(
    session_id: str,
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
) -> list[ChatMessage]:
    """Get all messages in a session."""
    session = await db.get_session(session_id, user.user_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    messages = await db.get_session_messages(session_id, user.user_id)

    # Fetch image metadata for user messages
    user_msg_ids = [m["id"] for m in messages if m["role"] == "user"]
    image_map = await db.get_message_images(user_msg_ids, user.user_id) if user_msg_ids else {}

    # Fetch web search sources for assistant messages
    assistant_msg_ids = [m["id"] for m in messages if m["role"] == "assistant"]
    sources_map = await db.get_message_sources(assistant_msg_ids) if assistant_msg_ids else {}

    return [
        ChatMessage(
            id=m["id"],
            role=m["role"],
            content=m["content"],
            mode_used=m.get("mode_used"),
            inference_model=m.get("model_used"),
            reasoning_content=m.get("reasoning_content"),
            image_id=image_map[m["id"]]["id"] if m["id"] in image_map else None,
            sources=sources_map.get(m["id"]) or None,
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
    session = await db.get_session(session_id, user.user_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    await db.update_session_title(session_id, user.user_id, request.title)
    updated = await db.get_session(session_id, user.user_id)
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
    session = await db.get_session(session_id, user.user_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    await db.pin_session(session_id, user.user_id, request.is_pinned)
    updated = await db.get_session(session_id, user.user_id)
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
    success = await db.delete_session(session_id, user.user_id)
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
    settings = get_settings()

    # Validate image if provided
    image_mime = None
    image_bytes = None
    if request.image:
        image_mime, image_bytes = _validate_image(request.image, settings)

    # Get or create session
    if request.session_id:
        session = await db.get_session(request.session_id, user_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
    else:
        title = request.message[:50] + "..." if len(request.message) > 50 else request.message
        session = await db.create_session(user_id, title)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session"
            )

    session_id = session["id"]
    session_summary = session.get("summary")
    summary_msg_count = session.get("summary_message_count", 0)

    # Auto-rename "New Chat" sessions on first message
    if session.get("title") == "New Chat":
        title = request.message[:50] + ("..." if len(request.message) > 50 else "")
        await db.update_session_title(session_id, user_id, title)

    # Store user message
    user_msg = await db.create_message(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=request.message,
    )
    if not user_msg:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store message"
        )

    # Store image locally if provided
    if image_bytes and image_mime:
        storage_path = get_storage_service().upload_image(user_id, image_bytes, image_mime)
        await db.create_message_image(
            message_id=user_msg["id"],
            user_id=user_id,
            storage_path=storage_path,
            filename="image",
            mime_type=image_mime,
            file_size=len(image_bytes),
        )

    # Get conversation history (reduced window when summary exists)
    history_limit = settings.summary_recent_messages if session_summary else 20
    history = await db.get_session_messages(session_id, user_id, limit=history_limit)

    # Build messages — for past messages with images, use placeholder text
    image_map = await db.get_message_images(
        [m["id"] for m in history if m["role"] == "user"], user_id
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

    # Retrieve relevant user memories for context injection
    memory_service = get_memory_service()
    memories_text = ""
    if memory_service.enabled and not request.conversation_mode:
        try:
            memories = await memory_service.retrieve_relevant_memories(user_id, request.message)
            memories_text = memory_service.format_memories_for_context(memories)
        except Exception as exc:
            print(f"[chat] Memory retrieval failed: {exc}")

    # Per-request id for the live pipeline panel (so the monitor can show
    # one timeline per chat turn). Short hex — 8 chars is plenty for the
    # bounded 50-event window we keep in the registry.
    req_id = secrets.token_hex(4)
    pipeline_metrics = get_metrics_registry()
    await pipeline_metrics.pipeline_event(req_id, "request.start", status="ok",
                                           note="streaming")

    # RAG retrieval runs BEFORE system-message build so chunks slot into the
    # strict-grounding template — see send_message() for rationale. RAG
    # failures (bad embedding model, missing reranker, etc.) must NEVER
    # break the chat stream — degrade to no-context retrieval and let the
    # model answer without RAG. The pipeline event records the failure so
    # the operator can see what broke without losing the user's response.
    rag_chunks = []
    if not request.conversation_mode:
        if not rag.enabled:
            print("[chat] RAG disabled — skipping document retrieval")
        elif await rag.user_has_documents(user_id):
            t_rag = time.perf_counter()
            try:
                rag_chunks = await rag.retrieve_context(
                    user_id, request.message, conversation_context=hist_messages,
                )
                _ms = (time.perf_counter() - t_rag) * 1000.0
                await pipeline_metrics.pipeline_event(
                    req_id, "rag.retrieve", ms=_ms, status="ok",
                    note=f"{len(rag_chunks)} chunks",
                )
            except Exception as exc:
                _ms = (time.perf_counter() - t_rag) * 1000.0
                print(f"[chat] RAG retrieval failed — continuing without context: "
                      f"{type(exc).__name__}: {exc}")
                await pipeline_metrics.pipeline_event(
                    req_id, "rag.retrieve", ms=_ms, status="fail",
                    note=f"{type(exc).__name__}: {str(exc)[:80]}",
                )
                rag_chunks = []

    rag_context_text = rag.format_context(rag_chunks) if rag_chunks else None
    rag_confidence = compute_retrieval_confidence(rag_chunks) if rag_chunks else "NONE"
    explicit_lookup = _explicit_lookup_phrasing(request.message)
    routing_mode = _routing_mode(rag_confidence, explicit_lookup)
    if rag_chunks or explicit_lookup:
        print(f"[chat] Retrieval confidence: {rag_confidence} → mode: {routing_mode}"
              + (" (explicit lookup phrasing)" if explicit_lookup else ""))

    # Streaming path doesn't use tool_calling (no tool loop on stream), so the
    # TOOLS AVAILABLE hint is suppressed here — advertising tools the model
    # can't call would be worse than staying silent. The tool-loop
    # (non-streaming) path advertises and uses them together.
    messages = [_build_system_message(
        request.conversation_mode,
        summary=session_summary,
        memories_text=memories_text,
        evidence=rag_context_text,
        confidence=rag_confidence,
        explicit_lookup=explicit_lookup,
        use_tool_calling=False,
    )] + hist_messages

    # Confidence-driven directives — see send_message() for rationale.
    directive = _confidence_directive(
        rag_confidence,
        rag_chunks,
        mode=routing_mode,
        use_tool_calling=False,
        factual_query=looks_like_search_query(request.message),
    )
    if directive:
        messages[0]["content"] += f"\n\n{directive}"

    # Session attachments: inject full document text (highest priority context)
    attach_text = await db.fetch_session_attachments_text(session_id, user_id)
    if attach_text:
        messages[0]["content"] += f"\n\n{attach_text}"

    # Web search: user controls via toggle button
    will_search = request.web_search

    # Force thinking tier when image is attached (instant tier is text-only)
    # Voice conversation always uses instant tier (Qwen 3.5 4B); config via INFERENCE_INSTANT_*
    if request.image:
        inference_mode = "thinking"
        print("[chat] Image attached — forcing thinking tier for vision")
    elif request.conversation_mode:
        inference_mode = "instant"
    else:
        inference_mode = request.mode.value

    # Force CoT off when RAG context is present on the thinking tier — see
    # send_message() for rationale.
    enable_thinking_override: bool | None = None
    if (rag_chunks
            and inference_mode in ("thinking", "thinking_harder")
            and settings.inference_thinking_auto_disable_for_rag):
        enable_thinking_override = False
        print(f"[chat] RAG present on {inference_mode} tier — disabling CoT")

    async def event_generator():
        """Generate SSE events from inference stream."""
        full_content = []
        full_reasoning = []
        # Strip <think>...</think> from delta.content as it streams so the
        # user never sees raw reasoning tags. Captured inner text is fed into
        # full_reasoning so the DB row still has the model's chain of thought.
        think_filter = ThinkBlockFilter()
        client_disconnected = False

        # ── Live-monitor instrumentation ─────────────────────────────
        # Stamp entry; stamp first-token when we see the first content delta;
        # in the finally-block compute ttft/tpot and record. The try/finally
        # guarantees the active-count can never leak on exception or early
        # return — critical, because a stuck counter poisons the dashboard.
        registry = get_metrics_registry()
        t_req_start = time.perf_counter()
        t_req_start_wall = time.time()
        t_first_token: float | None = None
        n_content_deltas = 0
        _metrics_error: str | None = None
        _stream_ok = False
        await registry.mark_active(inference_mode)

        try:
            if await http_request.is_disconnected():
                return

            try:
                yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            except (BrokenPipeError, ConnectionResetError, RuntimeError, asyncio.CancelledError):
                return

            # Web search: run inside generator so we can send real-time SSE events
            search_results = []
            if will_search:
                try:
                    yield f"data: {json.dumps({'type': 'search_start'})}\n\n"
                except (BrokenPipeError, ConnectionResetError, RuntimeError, asyncio.CancelledError):
                    return

                t_ws = time.perf_counter()
                search_results = await web_search.search(request.message)
                if search_results:
                    search_results = await web_search.enrich_with_full_content(search_results)
                    context_text = web_search.format_results_for_context(search_results)
                    messages[0]["content"] += f"\n\n{context_text}"
                _ms_ws = (time.perf_counter() - t_ws) * 1000.0
                await registry.pipeline_event(
                    req_id, "web.search", ms=_ms_ws, status="ok",
                    note=f"{len(search_results)} results",
                )

                sources = [{"title": r["title"], "url": r["url"]} for r in search_results]
                try:
                    yield f"data: {json.dumps({'type': 'search_done', 'sources': sources})}\n\n"
                except (BrokenPipeError, ConnectionResetError, RuntimeError, asyncio.CancelledError):
                    return

            # Inject evidence summary so the model can self-calibrate confidence
            evidence_summary = _build_evidence_summary(search_results, rag_chunks)
            if evidence_summary:
                messages[0]["content"] += f"\n\n{evidence_summary}"

            total_chars = sum(
                len(m["content"]) if isinstance(m.get("content"), str) else sum(
                    len(p.get("text", "")) for p in m.get("content", []) if isinstance(p, dict)
                )
                for m in messages
            )
            print(f"[chat] Sending {len(messages)} messages ({total_chars} chars) to inference")
            await registry.pipeline_event(
                req_id, f"inference.{inference_mode}", status="running",
                note=f"{total_chars} chars in",
            )
            await registry.set_in_flight(
                req_id=req_id,
                stage=f"inference.{inference_mode}",
                started_at=t_req_start_wall,
                tokens=0,
                note=f"{total_chars} chars in",
            )
            # Push a live token-count update at most every ~0.5s so the
            # in-flight panel stays current without thrashing the lock on
            # every delta during fast streams.
            _last_inflight_push = time.perf_counter()
            _INFLIGHT_PUSH_EVERY_SEC = 0.5

            async for chunk in inference.generate_response_stream(
                messages=messages,
                mode=inference_mode,
                enable_thinking=enable_thinking_override,
            ):
                rewritten_chunk = chunk
                try:
                    if chunk.startswith("data: ") and not chunk.strip().endswith("[DONE]"):
                        data = json.loads(chunk[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            visible, captured_reasoning = think_filter.feed(delta["content"])
                            if captured_reasoning:
                                full_reasoning.append(captured_reasoning)
                            if visible:
                                # Stamp the first-token time on the first VISIBLE
                                # delta — TTFT shouldn't include time spent inside
                                # an opening <think> block.
                                if t_first_token is None:
                                    t_first_token = time.perf_counter()
                                n_content_deltas += 1
                                full_content.append(visible)
                            if visible != delta["content"]:
                                delta["content"] = visible
                                rewritten_chunk = "data: " + json.dumps(data) + "\n\n"
                        if "reasoning_content" in delta:
                            full_reasoning.append(delta["reasoning_content"])
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass

                # Periodic in-flight refresh so the live panel shows a rising
                # token counter during long thinking-mode streams.
                _now = time.perf_counter()
                if _now - _last_inflight_push >= _INFLIGHT_PUSH_EVERY_SEC:
                    _last_inflight_push = _now
                    _ttft = ((t_first_token - t_req_start) * 1000.0) if t_first_token else None
                    try:
                        await registry.set_in_flight(
                            req_id=req_id,
                            stage=f"inference.{inference_mode}",
                            started_at=t_req_start_wall,
                            tokens=n_content_deltas,
                            ttft_ms=_ttft,
                            note=f"{n_content_deltas} deltas",
                        )
                    except Exception:
                        pass

                if await http_request.is_disconnected():
                    client_disconnected = True
                    break

                # Backend-agnostic shape normalization: fan-out split any
                # oversized upstream chunk into smaller SSE deltas so the
                # frontend typewriter sees a fine-grained stream regardless
                # of how the backend chunked. The bookkeeping above
                # (TTFT stamp, content/reasoning accumulators, in-flight
                # refresh) intentionally runs ONCE per upstream chunk —
                # the split is purely a wire-shape transformation.
                pieces = split_oversized_sse_delta(
                    rewritten_chunk, settings.streaming_max_chunk_chars,
                )
                for piece in pieces:
                    try:
                        yield piece
                    except (BrokenPipeError, ConnectionResetError, RuntimeError, asyncio.CancelledError):
                        client_disconnected = True
                        break
                if client_disconnected:
                    break

            # Drain any partial-tag tail buffered by the think filter.
            # Visible tail = held-back chars that turned out NOT to be a tag
            # opener (emit them). Reasoning tail = orphan-think content from
            # a stream that ended without </think> (persisted only).
            visible_tail, reasoning_tail = think_filter.flush()
            if reasoning_tail:
                full_reasoning.append(reasoning_tail)
            if visible_tail:
                full_content.append(visible_tail)
                if not client_disconnected:
                    final_chunk = "data: " + json.dumps({
                        "choices": [{"index": 0, "delta": {"content": visible_tail}}]
                    }) + "\n\n"
                    for piece in split_oversized_sse_delta(
                        final_chunk, settings.streaming_max_chunk_chars,
                    ):
                        try:
                            yield piece
                        except (BrokenPipeError, ConnectionResetError, RuntimeError, asyncio.CancelledError):
                            client_disconnected = True
                            break

            if client_disconnected:
                await registry.pipeline_event(
                    req_id, f"inference.{inference_mode}", status="fail",
                    note="client disconnected",
                )
                return
            _inf_ms = (time.perf_counter() - t_req_start) * 1000.0
            await registry.pipeline_event(
                req_id, f"inference.{inference_mode}", ms=_inf_ms, status="ok",
                note=f"{n_content_deltas} deltas",
            )

            # Build final content for DB (strip thinking tags, persist reasoning separately)
            raw_content = "".join(full_content)
            content = _finalize_assistant_text(raw_content)

            reasoning = "".join(full_reasoning) if full_reasoning else _extract_thinking(raw_content)
            if not reasoning:
                reasoning = None

            save_content = content if content else ("(no visible response)" if reasoning else None)
            if save_content:
                try:
                    assistant_msg = await db.create_message(
                        session_id=session_id,
                        user_id=user_id,
                        role="assistant",
                        content=save_content,
                        mode_used=inference_mode,
                        reasoning_content=reasoning,
                    )

                    # Persist web search source links
                    if assistant_msg and search_results:
                        await db.create_message_sources(
                            assistant_msg["id"],
                            [{"title": r["title"], "url": r["url"]} for r in search_results],
                        )

                    _validate_response_sources(
                        save_content, search_results, rag_chunks, routing_mode=routing_mode
                    )
                    # NOTE: CoVe (Phase 3.1) is intentionally not wired into
                    # the streaming path — running it after a stream
                    # completes would mutate stored DB content vs what the
                    # user already saw. If we want streaming-CoVe later,
                    # emit a final SSE event with the verification report
                    # and let the frontend overlay [?] markers post-hoc.

                    # Background tasks: summarisation and memory extraction (non-blocking)
                    total_msg_count = len(history) + 1
                    summariser = get_summariser_service()
                    if summariser.should_summarise(total_msg_count + summary_msg_count, summary_msg_count):
                        asyncio.create_task(_run_tracked_task(
                            "task.summary",
                            summariser.summarise_session(session_id, user_id),
                        ))

                    if memory_service.should_extract(total_msg_count):
                        recent_for_extraction = hist_messages + [
                            {"role": "assistant", "content": save_content}
                        ]
                        asyncio.create_task(_run_tracked_task(
                            "task.memory_extract",
                            memory_service.extract_and_store(
                                session_id, user_id, recent_for_extraction
                            ),
                        ))
                except Exception as db_err:
                    print(f"[chat] Failed to save assistant message to DB: {db_err}")

            _stream_ok = True
        except Exception as exc:
            _metrics_error = type(exc).__name__
            raise
        finally:
            total_ms = (time.perf_counter() - t_req_start) * 1000.0
            ttft_ms = ((t_first_token - t_req_start) * 1000.0) if t_first_token else None
            tpot_ms = None
            if ttft_ms is not None and n_content_deltas > 1:
                tpot_ms = max(0.0, (total_ms - ttft_ms) / n_content_deltas)
            await registry.mark_inactive(inference_mode)
            await registry.clear_in_flight()
            await registry.record(RequestRecord(
                ended_at=time.time(),
                tier=inference_mode,
                model=None,
                streaming=True,
                ok=_stream_ok,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                total_ms=total_ms,
                completion_tokens=n_content_deltas or None,
                error_type=_metrics_error,
            ))

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
) -> dict[str, Any]:
    """Transcribe audio to text. Optionally verify the speaker matches enrolled voice."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    from app.services.transcription import transcribe_audio

    try:
        text = await transcribe_audio(audio_bytes, file.filename or "audio.webm")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

    result: dict[str, Any] = {"text": text}

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
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    """Get voice enrollment status for the current user."""
    from app.services.speaker import get_voice_profile_status

    return await get_voice_profile_status(user.sub)


@router.delete("/voice-profile")
async def delete_voice_profile_endpoint(
    user: JWTPayload = Depends(get_current_user),
) -> dict[str, Any]:
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
    registry = get_metrics_registry()

    try:
        async with registry.subsystem_timer(
            "tts.synth", note=f"voice={request.voice} {len(request.text)} chars",
        ):
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
        await registry.set_model_state("tts", url=kokoro_url, status="up")
    except httpx.ConnectError:
        await registry.set_model_state("tts", url=kokoro_url, status="down")
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
    """Serve a chat image from local storage."""
    row = await db.get_image_metadata(image_id, user.user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    image_bytes = get_storage_service().download_image(row["storage_path"])
    return Response(content=image_bytes, media_type=row["mime_type"])


@router.get("/sessions/{session_id}/images")
async def get_session_images(
    session_id: str,
    user: JWTPayload = Depends(get_current_user),
    db: DatabaseService = Depends(get_database_service),
):
    """Get all image metadata for messages in a session."""
    session = await db.get_session(session_id, user.user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await db.get_session_messages(session_id, user.user_id)
    msg_ids = [m["id"] for m in messages if m["role"] == "user"]
    if not msg_ids:
        return {}

    images = await db.get_message_images(msg_ids, user.user_id)
    return images
