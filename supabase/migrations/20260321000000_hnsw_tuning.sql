-- ============================================
-- HNSW Tuning + Metadata Column
-- Phase 1C of RAG v2
-- ============================================
-- Rebuilds HNSW index with higher ef_construction for better recall.
-- Adds metadata JSONB column for chunk heading and other metadata.

-- 1. Drop and rebuild HNSW index with better parameters
DROP INDEX IF EXISTS idx_document_chunks_embedding;

CREATE INDEX idx_document_chunks_embedding
  ON public.document_chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 128);

-- 2. Add metadata column
ALTER TABLE public.document_chunks
  ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

COMMENT ON COLUMN public.document_chunks.metadata IS 'Chunk metadata: heading, source section, etc.';

-- 3. Allow 'processing_context' as a document status
ALTER TABLE public.documents DROP CONSTRAINT IF EXISTS documents_status_check;
ALTER TABLE public.documents ADD CONSTRAINT documents_status_check
  CHECK (status IN ('pending', 'processing', 'processing_context', 'completed', 'failed'));

-- 4. Update match_documents_with_metadata to set ef_search and return metadata
-- Must drop first because return type changed (added metadata column)
DROP FUNCTION IF EXISTS public.match_documents_with_metadata(vector, double precision, integer);

CREATE OR REPLACE FUNCTION public.match_documents_with_metadata(
  query_embedding vector(1536),
  match_threshold FLOAT DEFAULT 0.3,
  match_count INT DEFAULT 20
)
RETURNS TABLE (
  chunk_id UUID,
  document_id UUID,
  filename TEXT,
  content TEXT,
  chunk_index INTEGER,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
  -- Increase ef_search for better recall at query time
  SET LOCAL hnsw.ef_search = 100;

  RETURN QUERY
  SELECT
    dc.id AS chunk_id,
    dc.document_id,
    d.filename,
    dc.content,
    dc.chunk_index,
    dc.metadata,
    1 - (dc.embedding <=> query_embedding) AS similarity
  FROM public.document_chunks dc
  JOIN public.documents d ON d.id = dc.document_id
  WHERE dc.user_id = auth.uid()
    AND d.status = 'completed'
    AND 1 - (dc.embedding <=> query_embedding) > match_threshold
  ORDER BY dc.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
