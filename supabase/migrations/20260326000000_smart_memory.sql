-- ============================================
-- Smart LLM Memory: Summarisation + Structured Memory
-- Phase 8 of AI Assistant
-- ============================================
-- 8a: Rolling conversation summaries (columns on chat_sessions)
-- 8b: Cross-session user memories with vector search (new table + RPC)
-- ============================================

-- ============================================
-- 8a: Conversation Summarisation
-- ============================================
-- Two new columns on chat_sessions to store rolling summary state.
-- summary_message_count tracks how many messages have been summarised,
-- so we know when the threshold is crossed again.

ALTER TABLE public.chat_sessions
  ADD COLUMN IF NOT EXISTS summary TEXT,
  ADD COLUMN IF NOT EXISTS summary_message_count INTEGER DEFAULT 0 NOT NULL;

COMMENT ON COLUMN public.chat_sessions.summary IS 'Rolling conversation summary, updated incrementally';
COMMENT ON COLUMN public.chat_sessions.summary_message_count IS 'Number of messages included in the summary so far';


-- ============================================
-- 8b: Structured Memory (Cross-Session User Facts)
-- ============================================

CREATE TABLE IF NOT EXISTS public.user_memories (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'preference'
    CHECK (category IN ('preference', 'background', 'project', 'instruction')),
  embedding vector(1536),
  source_session_id UUID REFERENCES public.chat_sessions(id) ON DELETE SET NULL,
  access_count INTEGER DEFAULT 0 NOT NULL,
  last_accessed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

COMMENT ON TABLE public.user_memories IS 'Cross-session user facts extracted by LLM for personalised context';

ALTER TABLE public.user_memories ENABLE ROW LEVEL SECURITY;

-- RLS policies
DROP POLICY IF EXISTS "Users can view their own memories" ON public.user_memories;
DROP POLICY IF EXISTS "Users can create their own memories" ON public.user_memories;
DROP POLICY IF EXISTS "Users can update their own memories" ON public.user_memories;
DROP POLICY IF EXISTS "Users can delete their own memories" ON public.user_memories;

CREATE POLICY "Users can view their own memories"
  ON public.user_memories FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Users can create their own memories"
  ON public.user_memories FOR INSERT
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can update their own memories"
  ON public.user_memories FOR UPDATE
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can delete their own memories"
  ON public.user_memories FOR DELETE
  USING (user_id = auth.uid());

-- Indexes
CREATE INDEX IF NOT EXISTS idx_user_memories_user_id
  ON public.user_memories(user_id);

CREATE INDEX IF NOT EXISTS idx_user_memories_user_category
  ON public.user_memories(user_id, category);

-- HNSW index for cosine similarity search (matches document_chunks pattern)
CREATE INDEX IF NOT EXISTS idx_user_memories_embedding
  ON public.user_memories
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 128);

-- updated_at trigger (reuses existing function from initial schema)
DROP TRIGGER IF EXISTS update_user_memories_updated_at ON public.user_memories;
CREATE TRIGGER update_user_memories_updated_at
  BEFORE UPDATE ON public.user_memories
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();


-- ============================================
-- RPC: Match memories by vector similarity
-- ============================================
-- SECURITY INVOKER so RLS filters by auth.uid() automatically.
-- Returns top-k memories above threshold, ordered by cosine similarity.
-- Also bumps access_count and last_accessed_at for returned memories.

CREATE OR REPLACE FUNCTION public.match_memories(
  query_embedding vector(1536),
  match_threshold FLOAT DEFAULT 0.5,
  match_count INT DEFAULT 5
)
RETURNS TABLE (
  memory_id UUID,
  content TEXT,
  category TEXT,
  similarity FLOAT,
  created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
  -- Increase ef_search for better recall
  SET LOCAL hnsw.ef_search = 80;

  -- Update access stats for matched memories
  UPDATE public.user_memories um
  SET access_count = um.access_count + 1,
      last_accessed_at = NOW()
  WHERE um.user_id = auth.uid()
    AND um.embedding IS NOT NULL
    AND 1 - (um.embedding <=> query_embedding) > match_threshold
    AND um.id IN (
      SELECT um2.id
      FROM public.user_memories um2
      WHERE um2.user_id = auth.uid()
        AND um2.embedding IS NOT NULL
        AND 1 - (um2.embedding <=> query_embedding) > match_threshold
      ORDER BY um2.embedding <=> query_embedding
      LIMIT match_count
    );

  RETURN QUERY
  SELECT
    um.id AS memory_id,
    um.content,
    um.category,
    (1 - (um.embedding <=> query_embedding))::FLOAT AS similarity,
    um.created_at
  FROM public.user_memories um
  WHERE um.user_id = auth.uid()
    AND um.embedding IS NOT NULL
    AND 1 - (um.embedding <=> query_embedding) > match_threshold
  ORDER BY um.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION public.match_memories IS 'Cosine similarity search for user memories with access tracking';
