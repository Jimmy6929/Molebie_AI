"""
Document processing pipeline for RAG.

Handles text extraction (PDF, DOCX, TXT, MD), chunking with overlap,
embedding generation, and storage in Supabase (document_chunks table).
Supports markdown-header-aware splitting for better chunk boundaries.
"""

import io
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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

_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", " ", ""]

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _split_by_headings(text: str) -> List[Tuple[Optional[str], str]]:
    """Split text on markdown headings (# / ## / ###).

    Returns list of (heading, section_text) tuples.
    Non-headed text at the start gets heading=None.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections: List[Tuple[Optional[str], str]] = []

    # Text before first heading
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))

    return sections


def chunk_text_structured(
    text: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks with heading metadata.

    For markdown text, splits on section boundaries first, then falls
    back to recursive splitting for oversized sections.

    Returns list of dicts: {"text": str, "chunk_index": int, "heading": Optional[str]}
    """
    if not text.strip():
        return []

    sections = _split_by_headings(text)
    result: List[Dict[str, Any]] = []
    chunk_index = 0

    for heading, section_text in sections:
        raw_chunks: List[str] = []
        _recursive_split(section_text, _SEPARATORS, chunk_size, chunk_overlap, raw_chunks)
        for chunk in raw_chunks:
            if chunk.strip():
                result.append({
                    "text": chunk.strip(),
                    "chunk_index": chunk_index,
                    "heading": heading,
                })
                chunk_index += 1

    return result


def chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> List[str]:
    """
    Split text into overlapping chunks using a recursive character splitter.

    Tries to split on paragraph boundaries first, then sentences, then words.
    chunk_size and chunk_overlap are in *characters* (not tokens) for simplicity.
    Backward-compatible wrapper around chunk_text_structured().
    """
    structured = chunk_text_structured(text, chunk_size, chunk_overlap)
    return [c["text"] for c in structured]


def _recursive_split(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
    result: List[str],
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
    current: List[str] = []
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
            overlap_parts: List[str] = []
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
        self, file_data: bytes, file_type: str
    ) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        """
        Extract text, chunk, and embed.

        Returns list of (chunk_text, embedding_vector, metadata) tuples.
        metadata includes: {"chunk_index": int, "heading": Optional[str]}
        """
        text = extract_text(file_data, file_type)
        if not text.strip():
            raise ValueError("No text could be extracted from the document")

        structured_chunks = chunk_text_structured(text, self.chunk_size, self.chunk_overlap)
        if not structured_chunks:
            raise ValueError("Document produced no text chunks")

        chunk_texts = [c["text"] for c in structured_chunks]
        print(f"[doc_processor] Extracted {len(text)} chars → {len(chunk_texts)} chunks")

        print(f"[doc_processor] Generating embeddings for {len(chunk_texts)} chunks...")
        try:
            embeddings = self.embedding.embed_batch(chunk_texts)
        except Exception as exc:
            print(f"[doc_processor] Embedding failed: {type(exc).__name__}: {exc}")
            raise
        print(f"[doc_processor] Embeddings complete — {len(embeddings)} vectors of dim {len(embeddings[0])}")

        return [
            (
                c["text"],
                emb,
                {"chunk_index": c["chunk_index"], "heading": c["heading"]},
            )
            for c, emb in zip(structured_chunks, embeddings)
        ]

    async def process_async(
        self,
        file_data: bytes,
        file_type: str,
        full_text: Optional[str] = None,
    ) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        """
        Async version of process() that supports contextual retrieval.

        When contextual retrieval is enabled, generates context prefixes
        for each chunk using the local LLM, then embeds the contextualized
        version for better retrieval quality.

        Returns list of (chunk_text, embedding_vector, metadata) tuples.
        metadata includes: chunk_index, heading, and optionally content_contextualized.
        """
        text = full_text or extract_text(file_data, file_type)
        if not text.strip():
            raise ValueError("No text could be extracted from the document")

        structured_chunks = chunk_text_structured(text, self.chunk_size, self.chunk_overlap)
        if not structured_chunks:
            raise ValueError("Document produced no text chunks")

        chunk_texts = [c["text"] for c in structured_chunks]
        print(f"[doc_processor] Extracted {len(text)} chars → {len(chunk_texts)} chunks")

        # Contextual retrieval: generate context prefixes
        texts_to_embed = chunk_texts
        contextualized_texts: Optional[List[Optional[str]]] = None

        if self.settings.rag_contextual_retrieval_enabled:
            try:
                from app.services.context_generator import generate_batch

                print(f"[doc_processor] Generating context prefixes for {len(chunk_texts)} chunks...")
                contexts = await generate_batch(text, chunk_texts, self.settings)
                contextualized_texts = []
                texts_to_embed = []
                for chunk_text, context in zip(chunk_texts, contexts):
                    if context:
                        ctx_text = f"{context}\n\n{chunk_text}"
                        contextualized_texts.append(ctx_text)
                        texts_to_embed.append(ctx_text)
                    else:
                        contextualized_texts.append(None)
                        texts_to_embed.append(chunk_text)
                ctx_count = sum(1 for c in contextualized_texts if c)
                print(f"[doc_processor] Context generated for {ctx_count}/{len(chunk_texts)} chunks")
            except Exception as exc:
                print(f"[doc_processor] Context generation failed (degrading gracefully): {type(exc).__name__}: {exc}")
                contextualized_texts = None
                texts_to_embed = chunk_texts

        print(f"[doc_processor] Generating embeddings for {len(texts_to_embed)} chunks...")
        try:
            embeddings = self.embedding.embed_batch(texts_to_embed)
        except Exception as exc:
            print(f"[doc_processor] Embedding failed: {type(exc).__name__}: {exc}")
            raise
        print(f"[doc_processor] Embeddings complete — {len(embeddings)} vectors of dim {len(embeddings[0])}")

        results = []
        for i, (c, emb) in enumerate(zip(structured_chunks, embeddings)):
            meta: Dict[str, Any] = {
                "chunk_index": c["chunk_index"],
                "heading": c["heading"],
            }
            if contextualized_texts and contextualized_texts[i]:
                meta["content_contextualized"] = contextualized_texts[i]
            results.append((c["text"], emb, meta))

        return results


_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """Get cached DocumentProcessor instance."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor(get_settings(), get_embedding_service())
    return _processor
