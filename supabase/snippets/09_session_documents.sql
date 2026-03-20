-- ============================================
-- 09_session_documents.sql
-- Create session_documents table for "Attach to Chat"
-- Run this AFTER 02_chat_sessions.sql and 01_profiles.sql
-- ============================================

CREATE TABLE IF NOT EXISTS public.session_documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  content TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

COMMENT ON TABLE public.session_documents IS 'Full-text document attachments for a chat session (injected into system prompt)';

-- Enable Row Level Security
ALTER TABLE public.session_documents ENABLE ROW LEVEL SECURITY;

-- RLS Policies (drop first so snippet is safe to re-run)
DROP POLICY IF EXISTS "Users can view their own session documents" ON public.session_documents;
DROP POLICY IF EXISTS "Users can attach documents to their own sessions" ON public.session_documents;
DROP POLICY IF EXISTS "Users can remove their own session documents" ON public.session_documents;

CREATE POLICY "Users can view their own session documents"
  ON public.session_documents
  FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Users can attach documents to their own sessions"
  ON public.session_documents
  FOR INSERT
  WITH CHECK (
    user_id = auth.uid()
    AND session_id IN (SELECT id FROM public.chat_sessions WHERE user_id = auth.uid())
  );

CREATE POLICY "Users can remove their own session documents"
  ON public.session_documents
  FOR DELETE
  USING (user_id = auth.uid());

-- Indexes
CREATE INDEX IF NOT EXISTS idx_session_documents_session_id
  ON public.session_documents(session_id);

CREATE INDEX IF NOT EXISTS idx_session_documents_user_id
  ON public.session_documents(user_id);
