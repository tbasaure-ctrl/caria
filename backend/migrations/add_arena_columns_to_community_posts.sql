-- Migration: Add arena-related columns to community_posts table
-- This fixes the UndefinedColumn errors in /api/community/posts and /api/community/rankings
-- Created: 2025-11-23

-- Add arena-related columns to community_posts table
ALTER TABLE community_posts 
ADD COLUMN IF NOT EXISTS is_arena_post BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS arena_thread_id UUID,
ADD COLUMN IF NOT EXISTS arena_round_id UUID,
ADD COLUMN IF NOT EXISTS arena_community VARCHAR(100);

-- Add index for arena queries to improve performance
CREATE INDEX IF NOT EXISTS idx_community_posts_arena 
ON community_posts(is_arena_post, arena_community) 
WHERE is_arena_post = TRUE;

-- Verify columns were added
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_name = 'community_posts'
AND column_name IN ('is_arena_post', 'arena_thread_id', 'arena_round_id', 'arena_community')
ORDER BY ordinal_position;
