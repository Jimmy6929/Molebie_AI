# Molebie RAG configuration

Notes on Molebie's retrieval-augmented-generation pipeline as configured
after Phase 1 of the hallucination-mitigation work.

## Chunking

- Chunk size: 512 tokens
- Chunk overlap: 64 tokens
- Headings are captured as a breadcrumb (h1 > h2 > h3) and prepended to
  every chunk's embedding text in the format `breadcrumb | note_title`.
- Code fences are merged so triple-backtick blocks never split mid-code.

## Embedding

- Embedding model: Qwen/Qwen3-Embedding-0.6B
- Vector dimension: 1024
- Embedding text format: `breadcrumb | note_title` followed by the chunk body.

## Hybrid retrieval + reranking

- Hybrid retrieval combines sqlite-vec dense search with FTS5 BM25 sparse
  search. Results are merged via Reciprocal Rank Fusion with k = 60.
- Top-20 hits feed into the reranker.
- Reranker model: Qwen/Qwen3-Reranker-0.6B (cross-encoder yes/no head).
- After reranking, top-5 chunks survive to the LLM.

## Match thresholds

- Match count: 30 (raw retrieval before reranking)
- Match threshold: 0.3 cosine similarity floor
- Maximum context: 12000 chars across the surviving chunks

## LongContextReorder

After reranking, chunks are reordered in a U-shape: the highest-scoring
chunk lands at the end of the context window, second-highest at the start,
and the weakest chunks in the middle. This matches LangChain's
LongContextReorder and exploits the lost-in-the-middle effect on
modern LLMs.
