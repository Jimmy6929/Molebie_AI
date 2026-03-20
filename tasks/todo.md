# Project TODO

## Next Up: RAG Quality (Top Priority)
- [ ] Explore current retrieval pipeline end-to-end
- [ ] Identify retrieval accuracy bottlenecks
- [ ] Test document upload/processing flow
- [ ] Debug known upload/processing issues
- [ ] Improve chunking strategy if needed
- [ ] Tune embedding search (similarity thresholds, reranking)
- [ ] Validate RAG responses against source documents

## Document "Brain" / Attach-to-Chat
- [ ] Investigate known bugs in session-document attachment
- [ ] Test per-session document context flow
- [ ] Ensure AI reliably uses attached doc as primary context

## TTS Speed
- [ ] Profile Kokoro generation latency
- [ ] Investigate streaming TTS options
- [ ] Explore chunked generation approach
- [ ] Research smaller/faster voice models compatible with MLX

## Distribution
- [ ] Design one-command installer architecture
- [ ] Script Docker/Supabase/model auto-setup
- [ ] Cross-platform support (macOS first)

## Multi-User
- [ ] Verify RLS coverage across all tables
- [ ] Plan auth flow for additional users
