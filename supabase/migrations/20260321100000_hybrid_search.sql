-- ============================================
-- Hybrid Search: tsvector + RRF Fusion
-- Phase 2 of RAG v2
-- ============================================
-- Adds full-text search (BM25-style) alongside vector search.
-- Uses native PostgreSQL tsvector — no new extensions.
-- RRF (Reciprocal Rank Fusion) combines both result sets.

-- 1. Add tsvector column for full-text search
ALTER TABLE public.document_chunks
  ADD COLUMN IF NOT EXISTS fts tsvector;

-- 2. Backfill existing rows
UPDATE public.document_chunks
  SET fts = to_tsvector('english', content)
  WHERE fts IS NULL;

-- 3. Auto-update trigger: keep fts in sync on INSERT/UPDATE
CREATE OR REPLACE FUNCTION public.document_chunks_fts_trigger()
RETURNS TRIGGER AS $$
BEGIN
  NEW.fts := to_tsvector('english', COALESCE(NEW.content, ''));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_document_chunks_fts ON public.document_chunks;
CREATE TRIGGER trg_document_chunks_fts
  BEFORE INSERT OR UPDATE OF content ON public.document_chunks
  FOR EACH ROW
  EXECUTE FUNCTION public.document_chunks_fts_trigger();

-- 4. GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_document_chunks_fts
  ON public.document_chunks USING gin (fts);

-- 5. Hybrid search function: vector + text via RRF
CREATE OR REPLACE FUNCTION public.hybrid_search(
  query_embedding vector(1536),
  query_text TEXT,
  match_count INT DEFAULT 20,
  match_threshold FLOAT DEFAULT 0.3,
  rrf_k INT DEFAULT 60,
  vector_weight FLOAT DEFAULT 0.7,
  text_weight FLOAT DEFAULT 0.3
)
RETURNS TABLE (
  chunk_id UUID,
  document_id UUID,
  filename TEXT,
  content TEXT,
  chunk_index INTEGER,
  metadata JSONB,
  similarity FLOAT,
  text_rank FLOAT,
  rrf_score FLOAT
)
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
  -- Increase ef_search for better vector recall
  SET LOCAL hnsw.ef_search = 100;

  RETURN QUERY
  WITH
  -- Vector search arm: top-N by cosine similarity
  vector_results AS (
    SELECT
      dc.id,
      dc.document_id,
      d.filename,
      dc.content,
      dc.chunk_index,
      dc.metadata,
      1 - (dc.embedding <=> query_embedding) AS sim,
      ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_embedding) AS vector_rank
    FROM public.document_chunks dc
    JOIN public.documents d ON d.id = dc.document_id
    WHERE dc.user_id = auth.uid()
      AND d.status = 'completed'
      AND 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count
  ),
  -- Text search arm: top-N by ts_rank_cd
  text_results AS (
    SELECT
      dc.id,
      dc.document_id,
      d.filename,
      dc.content,
      dc.chunk_index,
      dc.metadata,
      ts_rank_cd(dc.fts, websearch_to_tsquery('english', query_text)) AS txt_rank,
      ROW_NUMBER() OVER (
        ORDER BY ts_rank_cd(dc.fts, websearch_to_tsquery('english', query_text)) DESC
      ) AS text_rank_num
    FROM public.document_chunks dc
    JOIN public.documents d ON d.id = dc.document_id
    WHERE dc.user_id = auth.uid()
      AND d.status = 'completed'
      AND dc.fts @@ websearch_to_tsquery('english', query_text)
    ORDER BY txt_rank DESC
    LIMIT match_count
  ),
  -- RRF fusion: combine both arms
  combined AS (
    SELECT
      COALESCE(v.id, t.id) AS id,
      COALESCE(v.document_id, t.document_id) AS doc_id,
      COALESCE(v.filename, t.filename) AS fname,
      COALESCE(v.content, t.content) AS cont,
      COALESCE(v.chunk_index, t.chunk_index) AS cidx,
      COALESCE(v.metadata, t.metadata) AS meta,
      COALESCE(v.sim, 0.0) AS sim,
      COALESCE(t.txt_rank, 0.0) AS txt_rank,
      -- RRF: score = weight * 1/(k + rank)
      COALESCE(vector_weight * (1.0 / (rrf_k + v.vector_rank)), 0.0) +
      COALESCE(text_weight * (1.0 / (rrf_k + t.text_rank_num)), 0.0) AS rrf
    FROM vector_results v
    FULL OUTER JOIN text_results t ON v.id = t.id
  )
  SELECT
    combined.id AS chunk_id,
    combined.doc_id AS document_id,
    combined.fname AS filename,
    combined.cont AS content,
    combined.cidx AS chunk_index,
    combined.meta AS metadata,
    combined.sim::FLOAT AS similarity,
    combined.txt_rank::FLOAT AS text_rank,
    combined.rrf::FLOAT AS rrf_score
  FROM combined
  ORDER BY combined.rrf DESC
  LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION public.hybrid_search IS 'Hybrid vector + full-text search with Reciprocal Rank Fusion';
