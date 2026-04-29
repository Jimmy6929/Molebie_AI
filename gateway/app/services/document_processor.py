"""
Document processing pipeline for RAG.

Handles text extraction (PDF, DOCX, TXT, MD), chunking with overlap,
embedding generation, and storage in the document_chunks table.
Supports markdown-header-aware splitting for better chunk boundaries.
"""

import io
import re
from typing import Any

from app.config import Settings, get_settings
from app.services.embedding import EmbeddingService, get_embedding_service

# ── Text Extraction ──────────────────────────────────────────────────────

def extract_text_from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def extract_text_from_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(data: bytes, file_type: str) -> str:
    """Extract plain text from supported file types."""
    ft = file_type.lower()
    if ft in ("application/pdf", "pdf"):
        return extract_text_from_pdf(data)
    if ft in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"):
        return extract_text_from_docx(data)
    # TXT / MD — decode as UTF-8
    return data.decode("utf-8", errors="replace")


# ── Chunking ─────────────────────────────────────────────────────────────

# Build-plan separators: prefer markdown structure, then paragraphs, then
# sentences, then words. Note: the heading separators (\n## , \n### ) only
# trigger inside oversized sections — _split_by_headings already cuts on
# headings as a first pass.
_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```")


def _split_by_headings(
    text: str,
) -> list[tuple[tuple[str, ...], str]]:
    """Split text on markdown headings, tracking the full breadcrumb path.

    Returns list of ``(breadcrumb, section_text)`` tuples where ``breadcrumb``
    is a tuple of header strings from h1 down to the current level
    (e.g. ``("Setup", "Installation", "Dependencies")``). Non-headed
    preamble text gets ``breadcrumb=()``.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [((), text)]

    sections: list[tuple[tuple[str, ...], str]] = []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(((), preamble))

    # Maintain a stack of (level, heading) for h1/h2/h3 → breadcrumb at each section
    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        # Pop any same-or-deeper levels off the stack before pushing
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        breadcrumb = tuple(h for _, h in stack)

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((breadcrumb, body))

    return sections


def _merge_unbalanced_fences(chunks: list[str]) -> list[str]:
    """Merge consecutive chunks until triple-backtick fences are balanced.

    The recursive splitter doesn't know about code fences, so an oversized
    fenced block can land split across two chunks. Detect odd fence counts
    and merge forward — code blocks stay readable, slightly larger chunks
    are an acceptable tradeoff vs broken syntax.
    """
    if not chunks:
        return chunks
    merged: list[str] = []
    buffer: str | None = None
    for chunk in chunks:
        candidate = chunk if buffer is None else f"{buffer}\n{chunk}"
        fences = len(_CODE_FENCE_RE.findall(candidate))
        if fences % 2 == 1:
            buffer = candidate           # still unbalanced — keep accumulating
        else:
            merged.append(candidate)
            buffer = None
    if buffer is not None:               # trailing unbalanced fence — keep as-is
        merged.append(buffer)
    return merged


def chunk_text_structured(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict[str, Any]]:
    """Split text into overlapping chunks with header-breadcrumb metadata.

    For markdown text, splits on heading boundaries first (tracking the
    full h1>h2>h3 breadcrumb), then recursively splits oversized sections,
    then merges chunks that landed mid-code-fence so triple-backtick blocks
    aren't broken across chunks.

    Returns list of dicts:
      ``{"text": str, "chunk_index": int, "heading": str | None,
         "breadcrumb": list[str]}``

    ``heading`` is the deepest header (back-compat with existing callers);
    ``breadcrumb`` is the full path.
    """
    if not text.strip():
        return []

    sections = _split_by_headings(text)
    result: list[dict[str, Any]] = []
    chunk_index = 0

    for breadcrumb, section_text in sections:
        raw_chunks: list[str] = []
        _recursive_split(section_text, _SEPARATORS, chunk_size, chunk_overlap, raw_chunks)
        balanced_chunks = _merge_unbalanced_fences(raw_chunks)
        for chunk in balanced_chunks:
            stripped = chunk.strip()
            if not stripped:
                continue
            result.append({
                "text": stripped,
                "chunk_index": chunk_index,
                "heading": breadcrumb[-1] if breadcrumb else None,
                "breadcrumb": list(breadcrumb),
            })
            chunk_index += 1

    return result


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    """
    Split text into overlapping chunks using a recursive character splitter.

    Tries to split on paragraph boundaries first, then sentences, then words.
    chunk_size and chunk_overlap are in *characters* (not tokens) for simplicity.
    Backward-compatible wrapper around chunk_text_structured().
    """
    structured = chunk_text_structured(text, chunk_size, chunk_overlap)
    return [c["text"] for c in structured]


def _embedding_prefix(
    breadcrumb: list[str],
    note_title: str | None,
) -> str:
    """Build the contextual prefix prepended before embedding, per task 1.4.

    Format: ``{h1 > h2 > h3} | {note_title}``. Cheap contextual chunking —
    captures most of the lift of LLM-generated context without paying the
    per-chunk LLM cost. Returns empty string when no metadata exists.
    """
    parts: list[str] = []
    if breadcrumb:
        parts.append(" > ".join(breadcrumb))
    if note_title:
        parts.append(note_title)
    return " | ".join(parts)


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
    result: list[str],
):
    if len(text) <= chunk_size:
        result.append(text.strip())
        return

    sep = separators[0] if separators else ""
    next_seps = separators[1:] if len(separators) > 1 else [""]

    if sep == "":
        # Hard character split as last resort
        for i in range(0, len(text), chunk_size - chunk_overlap):
            piece = text[i : i + chunk_size]
            if piece.strip():
                result.append(piece.strip())
        return

    parts = text.split(sep)
    current: list[str] = []
    current_len = 0

    for part in parts:
        part_len = len(part) + len(sep)
        if current_len + part_len > chunk_size and current:
            merged = sep.join(current).strip()
            if len(merged) > chunk_size and next_seps:
                _recursive_split(merged, next_seps, chunk_size, chunk_overlap, result)
            elif merged:
                result.append(merged)

            # Keep overlap
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) + len(sep) > chunk_overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_len += len(p) + len(sep)
            current = overlap_parts
            current_len = overlap_len

        current.append(part)
        current_len += part_len

    if current:
        merged = sep.join(current).strip()
        if len(merged) > chunk_size and next_seps:
            _recursive_split(merged, next_seps, chunk_size, chunk_overlap, result)
        elif merged:
            result.append(merged)


# ── Processing Pipeline ──────────────────────────────────────────────────

class DocumentProcessor:
    """Orchestrates the full document processing pipeline."""

    def __init__(self, settings: Settings, embedding_service: EmbeddingService):
        self.settings = settings
        self.embedding = embedding_service
        self.chunk_size = settings.rag_chunk_size
        self.chunk_overlap = settings.rag_chunk_overlap

    def process(
        self,
        file_data: bytes,
        file_type: str,
        note_title: str | None = None,
    ) -> list[tuple[str, list[float], dict[str, Any]]]:
        """
        Extract text, chunk, and embed.

        ``note_title`` (typically the filename without extension) is prepended
        alongside the heading breadcrumb to the embedding-time text — cheap
        contextual chunking that improves retrieval without an LLM-per-chunk
        cost.

        Returns list of (chunk_text, embedding_vector, metadata) tuples.
        metadata includes ``chunk_index``, ``heading``, ``breadcrumb``.
        """
        text = extract_text(file_data, file_type)
        if not text.strip():
            raise ValueError("No text could be extracted from the document")

        structured_chunks = chunk_text_structured(text, self.chunk_size, self.chunk_overlap)
        if not structured_chunks:
            raise ValueError("Document produced no text chunks")

        # Build embedding-time texts: prefix each chunk with breadcrumb + title.
        # The original chunk text (without prefix) is what we store in
        # `content` and what the model sees at inference time.
        chunk_texts = [c["text"] for c in structured_chunks]
        embed_texts: list[str] = []
        for c in structured_chunks:
            prefix = _embedding_prefix(c.get("breadcrumb", []), note_title)
            embed_texts.append(f"{prefix}\n\n{c['text']}" if prefix else c["text"])

        print(f"[doc_processor] Extracted {len(text)} chars → {len(chunk_texts)} chunks")

        print(f"[doc_processor] Generating embeddings for {len(chunk_texts)} chunks...")
        try:
            embeddings = self.embedding.embed_batch(embed_texts)
        except Exception as exc:
            print(f"[doc_processor] Embedding failed: {type(exc).__name__}: {exc}")
            raise
        print(f"[doc_processor] Embeddings complete — {len(embeddings)} vectors of dim {len(embeddings[0])}")

        return [
            (
                c["text"],
                emb,
                {
                    "chunk_index": c["chunk_index"],
                    "heading": c["heading"],
                    "breadcrumb": c.get("breadcrumb", []),
                },
            )
            for c, emb in zip(structured_chunks, embeddings, strict=False)
        ]

    async def process_async(
        self,
        file_data: bytes,
        file_type: str,
        full_text: str | None = None,
        note_title: str | None = None,
    ) -> list[tuple[str, list[float], dict[str, Any]]]:
        """
        Async version of process() that supports contextual retrieval.

        Embedding-time text is built as ``{LLM context}\\n\\n{breadcrumb |
        note_title}\\n\\n{chunk_text}`` when contextual retrieval is on, or
        ``{breadcrumb | note_title}\\n\\n{chunk_text}`` otherwise. The
        breadcrumb prefix is the cheap fallback when LLM context generation
        is disabled or fails.

        Returns list of (chunk_text, embedding_vector, metadata) tuples.
        metadata includes ``chunk_index``, ``heading``, ``breadcrumb``, and
        optionally ``content_contextualized``.
        """
        text = full_text or extract_text(file_data, file_type)
        if not text.strip():
            raise ValueError("No text could be extracted from the document")

        structured_chunks = chunk_text_structured(text, self.chunk_size, self.chunk_overlap)
        if not structured_chunks:
            raise ValueError("Document produced no text chunks")

        chunk_texts = [c["text"] for c in structured_chunks]
        breadcrumb_prefixes = [
            _embedding_prefix(c.get("breadcrumb", []), note_title)
            for c in structured_chunks
        ]
        print(f"[doc_processor] Extracted {len(text)} chars → {len(chunk_texts)} chunks")

        # Contextual retrieval: LLM context prefix on top of breadcrumb prefix.
        contextualized_texts: list[str | None] | None = None
        contexts: list[str | None] = [None] * len(chunk_texts)

        if self.settings.rag_contextual_retrieval_enabled:
            try:
                from app.services.context_generator import generate_batch

                print(f"[doc_processor] Generating context prefixes for {len(chunk_texts)} chunks...")
                contexts = await generate_batch(text, chunk_texts, self.settings)
                contextualized_texts = []
                for chunk_text, context in zip(chunk_texts, contexts, strict=False):
                    if context:
                        contextualized_texts.append(f"{context}\n\n{chunk_text}")
                    else:
                        contextualized_texts.append(None)
                ctx_count = sum(1 for c in contextualized_texts if c)
                print(f"[doc_processor] Context generated for {ctx_count}/{len(chunk_texts)} chunks")
            except Exception as exc:
                print(f"[doc_processor] Context generation failed (degrading gracefully): {type(exc).__name__}: {exc}")
                contextualized_texts = None

        # Compose embedding inputs: LLM context (if any), breadcrumb, chunk.
        texts_to_embed: list[str] = []
        for i, chunk_text in enumerate(chunk_texts):
            parts: list[str] = []
            ctx = contexts[i] if i < len(contexts) else None
            if ctx:
                parts.append(ctx)
            if breadcrumb_prefixes[i]:
                parts.append(breadcrumb_prefixes[i])
            parts.append(chunk_text)
            texts_to_embed.append("\n\n".join(parts))

        print(f"[doc_processor] Generating embeddings for {len(texts_to_embed)} chunks...")
        try:
            embeddings = self.embedding.embed_batch(texts_to_embed)
        except Exception as exc:
            print(f"[doc_processor] Embedding failed: {type(exc).__name__}: {exc}")
            raise
        print(f"[doc_processor] Embeddings complete — {len(embeddings)} vectors of dim {len(embeddings[0])}")

        results = []
        for i, (c, emb) in enumerate(zip(structured_chunks, embeddings, strict=False)):
            meta: dict[str, Any] = {
                "chunk_index": c["chunk_index"],
                "heading": c["heading"],
                "breadcrumb": c.get("breadcrumb", []),
            }
            if contextualized_texts and contextualized_texts[i]:
                meta["content_contextualized"] = contextualized_texts[i]
            results.append((c["text"], emb, meta))

        return results


_processor: DocumentProcessor | None = None


def get_document_processor() -> DocumentProcessor:
    """Get cached DocumentProcessor instance."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor(get_settings(), get_embedding_service())
    return _processor
