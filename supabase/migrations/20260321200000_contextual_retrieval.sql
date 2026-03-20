-- ============================================
-- Contextual Retrieval Support
-- Phase 3 of RAG v2
-- ============================================
-- Adds content_contextualized column for enriched chunk text.
-- Updates FTS trigger to prefer contextualized content for search.
-- Search uses contextualized text; display uses raw content.

-- 1. Add contextualized content column
ALTER TABLE public.document_chunks
  ADD COLUMN IF NOT EXISTS content_contextualized TEXT;

COMMENT ON COLUMN public.document_chunks.content_contextualized
  IS 'Chunk content with context prefix (for search). Raw content is in content column (for display).';

-- 2. Update FTS trigger to prefer contextualized content
CREATE OR REPLACE FUNCTION public.document_chunks_fts_trigger()
RETURNS TRIGGER AS $$
BEGIN
  NEW.fts := to_tsvector('english', COALESCE(NEW.content_contextualized, NEW.content, ''));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 3. Re-trigger FTS for existing rows that now have the updated function
-- (only needed if there are rows with content_contextualized already set)
-- For fresh installs this is a no-op.
UPDATE public.document_chunks
  SET fts = to_tsvector('english', COALESCE(content_contextualized, content, ''))
  WHERE fts IS NOT NULL;
