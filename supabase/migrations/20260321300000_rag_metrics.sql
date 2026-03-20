-- ============================================
-- RAG Query Metrics Table
-- Phase 5 of RAG v2
-- ============================================
-- Stores per-query RAG metrics for trend analysis.

CREATE TABLE IF NOT EXISTS public.rag_query_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  query_text TEXT,
  num_candidates INTEGER,
  unique_documents INTEGER,
  top_similarity FLOAT,
  avg_similarity FLOAT,
  top_rrf_score FLOAT,
  top_rerank_score FLOAT,
  score_spread FLOAT,
  hybrid_enabled BOOLEAN,
  reranker_enabled BOOLEAN,
  t_embed_ms FLOAT,
  t_search_ms FLOAT,
  t_rerank_ms FLOAT,
  t_total_ms FLOAT,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

COMMENT ON TABLE public.rag_query_metrics IS 'Per-query RAG pipeline metrics for trend analysis';

ALTER TABLE public.rag_query_metrics ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own rag metrics" ON public.rag_query_metrics;
DROP POLICY IF EXISTS "Users can create their own rag metrics" ON public.rag_query_metrics;

CREATE POLICY "Users can view their own rag metrics"
  ON public.rag_query_metrics FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Users can create their own rag metrics"
  ON public.rag_query_metrics FOR INSERT
  WITH CHECK (user_id = auth.uid());

CREATE INDEX IF NOT EXISTS idx_rag_query_metrics_user_created
  ON public.rag_query_metrics(user_id, created_at DESC);
