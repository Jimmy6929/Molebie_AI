-- Add reasoning_content column to persist thinking/reasoning from AI responses
ALTER TABLE public.chat_messages
  ADD COLUMN IF NOT EXISTS reasoning_content TEXT;

-- Force PostgREST to pick up the new column immediately
NOTIFY pgrst, 'reload schema';
