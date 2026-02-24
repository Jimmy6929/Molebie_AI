-- ============================================
-- 04_documents.sql
-- Create documents table for RAG
-- Run this FIFTH
-- ============================================

-- Create documents table
CREATE TABLE IF NOT EXISTS public.documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  storage_path TEXT NOT NULL,
  file_type TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  status TEXT DEFAULT 'pending' NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  processed_at TIMESTAMPTZ
);

-- Add comments
COMMENT ON TABLE public.documents IS 'Metadata about uploaded files for RAG';
COMMENT ON COLUMN public.documents.status IS 'pending=uploaded, processing=extracting text, completed=ready for RAG, failed=error';

-- Enable Row Level Security
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- RLS Policies for documents
CREATE POLICY "Users can view their own documents"
  ON public.documents
  FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Users can upload their own documents"
  ON public.documents
  FOR INSERT
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can delete their own documents"
  ON public.documents
  FOR DELETE
  USING (user_id = auth.uid());

-- Note: No UPDATE policy for users - status updates are done via service role
-- Service role bypasses RLS for backend processing

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON public.documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_created ON public.documents(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status ON public.documents(status);
