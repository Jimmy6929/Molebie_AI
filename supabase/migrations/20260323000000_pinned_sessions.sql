-- Add is_pinned column to chat_sessions for pin/favorite feature
ALTER TABLE public.chat_sessions ADD COLUMN IF NOT EXISTS is_pinned BOOLEAN DEFAULT FALSE NOT NULL;
