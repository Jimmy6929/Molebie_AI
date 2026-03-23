# Project TODO

## Completed

### RAG Quality ✅
- [x] Explore current retrieval pipeline end-to-end
- [x] Identify retrieval accuracy bottlenecks
- [x] Test document upload/processing flow
- [x] Debug known upload/processing issues
- [x] Improve chunking strategy (1024 chars, 128 overlap, markdown-aware)
- [x] Tune embedding search (hybrid vector+BM25, RRF fusion, cross-encoder reranking)
- [x] Validate RAG responses against source documents (rag_eval.py with hit rate, MRR)

### Document "Brain" / Attach-to-Chat ✅
- [x] Investigate known bugs in session-document attachment
- [x] Test per-session document context flow
- [x] Ensure AI reliably uses attached doc as primary context

### Image Understanding (Vision) ✅
- [x] Add image upload (file picker, clipboard paste, drag-and-drop)
- [x] Store images in Supabase Storage (chat-images bucket)
- [x] Send images to Qwen3.5-9B vision encoder (multimodal message format)
- [x] Auto-compress images to max 1024px (prevents OOM on 16GB)
- [x] Auto-route image messages to thinking tier
- [x] Display images in chat history (MessageBubble)
- [x] Image serving endpoint (GET /chat/images/{id})

### Multi-User (Partial) ✅
- [x] Verify RLS coverage across all tables (profiles, sessions, messages, images, documents, chunks, storage)

---

## Next Up: UX Polish (High Impact)

### Code & Message Quality
- [x] Copy button on code blocks (highest-friction gap vs ChatGPT/Claude)
- [x] Regenerate message button
- [x] Math/LaTeX rendering (KaTeX)
- [x] Inline source citations ([1], [2] in text, like Perplexity)

### Session Management
- [x] Search/filter sessions in sidebar
- [x] Pin/favorite sessions
- [x] Conversation export (Markdown)

### Distribution / Publish-Ready ✅
- [x] Root `.env.example` with comprehensive defaults (single-machine localhost)
- [x] `setup.sh` interactive installer (single + two-machine mode)
- [x] Unified `docker-compose.yml` (SearXNG + Kokoro TTS)
- [x] Configurable CORS via `CORS_ORIGINS` env var (no hardcoded Tailscale IPs)
- [x] Generalized README for any Apple Silicon user
- [x] `make setup` target
- [x] MIT License
- [x] `.gitignore` cleanup (tasks/, .claude/, .cursor/, *.plan.md)

---
