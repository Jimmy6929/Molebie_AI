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

### Auth & Distribution
- [ ] OAuth sign-in (GitHub, Google)
- [ ] Plan auth flow for additional users (invite system)
- [ ] One-command installer architecture

---

## Model Performance Improvement

### Tier 1: Immediate (no training needed)
- [ ] Better few-shot prompting in system prompt
- [ ] Inference-time self-consistency (generate multiple, pick best)
- [ ] Inference-time verification (model checks its own work)

### Tier 2: LoRA Fine-Tuning (1-2 days, on M4 Pro 48GB)
- [ ] Set up mlx-tune or mlx-lm LoRA on friend's M4 Pro
- [ ] Curate 500+ domain-specific training examples
- [ ] QLoRA fine-tune Qwen3.5-9B (~7GB VRAM, fits easily)
- [ ] Merge adapters and deploy back to M2 Pro

### Tier 3: GRPO Reasoning Training (1-2 weeks, on M4 Pro 48GB)
- [ ] Set up MLX-GRPO or mlx-tune GRPO pipeline
- [ ] Create verifiable reward functions (math, code execution)
- [ ] Bootstrap training data from GSM8K / code datasets
- [ ] GRPO train Qwen3.5-9B with QLoRA (~13-15GB VRAM)
- [ ] Evaluate reasoning improvement on benchmarks
- [ ] Export and deploy improved model

### Tier 4: Bonus
- [ ] Model merging (combine reasoning + domain adapters via mergekit)
- [ ] Explore Absolute Zero approach if cloud GPUs become available

---

## TTS Speed (Lower Priority)
- [ ] Profile Kokoro generation latency
- [ ] Investigate streaming TTS options
- [ ] Explore chunked generation approach
- [ ] Research smaller/faster voice models compatible with MLX
