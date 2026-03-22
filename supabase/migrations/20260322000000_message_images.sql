-- ============================================
-- MESSAGE IMAGES MIGRATION
-- Created: 2026-03-22
-- Description: Support image attachments in chat messages
-- Images stored in Supabase Storage; metadata in this table.
-- ============================================

-- ============================================
-- MESSAGE IMAGES TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS public.message_images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID NOT NULL REFERENCES public.chat_messages(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  storage_path TEXT NOT NULL,
  filename TEXT,
  mime_type TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

COMMENT ON TABLE public.message_images IS 'Image attachments for chat messages, stored in Supabase Storage';

ALTER TABLE public.message_images ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own message images" ON public.message_images;
DROP POLICY IF EXISTS "Users can insert own message images" ON public.message_images;
DROP POLICY IF EXISTS "Users can delete own message images" ON public.message_images;

CREATE POLICY "Users can view own message images"
  ON public.message_images FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Users can insert own message images"
  ON public.message_images FOR INSERT
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can delete own message images"
  ON public.message_images FOR DELETE
  USING (user_id = auth.uid());

CREATE INDEX IF NOT EXISTS idx_message_images_message_id ON public.message_images(message_id);
CREATE INDEX IF NOT EXISTS idx_message_images_user_id ON public.message_images(user_id);

-- ============================================
-- CHAT-IMAGES STORAGE BUCKET
-- ============================================

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'chat-images',
  'chat-images',
  FALSE,
  5242880,  -- 5 MB
  ARRAY['image/jpeg', 'image/png', 'image/gif', 'image/webp']
)
ON CONFLICT (id) DO UPDATE SET
  file_size_limit = EXCLUDED.file_size_limit,
  allowed_mime_types = EXCLUDED.allowed_mime_types;

-- Storage RLS for chat-images bucket
DROP POLICY IF EXISTS "Users can upload chat images to own folder" ON storage.objects;
DROP POLICY IF EXISTS "Users can view own chat images" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete own chat images" ON storage.objects;

CREATE POLICY "Users can upload chat images to own folder"
  ON storage.objects FOR INSERT
  TO authenticated
  WITH CHECK (
    bucket_id = 'chat-images'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Users can view own chat images"
  ON storage.objects FOR SELECT
  TO authenticated
  USING (
    bucket_id = 'chat-images'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Users can delete own chat images"
  ON storage.objects FOR DELETE
  TO authenticated
  USING (
    bucket_id = 'chat-images'
    AND (storage.foldername(name))[1] = auth.uid()::text
  );
