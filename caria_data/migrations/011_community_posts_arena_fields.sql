-- Migration: Add arena-related fields to community_posts table
-- Purpose: Link community posts to Thesis Arena threads and rounds

-- Add arena fields to community_posts table
ALTER TABLE community_posts
ADD COLUMN IF NOT EXISTS arena_thread_id UUID REFERENCES thesis_arena_threads(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS arena_round_id UUID REFERENCES arena_rounds(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS arena_community VARCHAR(50), -- Which community perspective this post represents
ADD COLUMN IF NOT EXISTS is_arena_post BOOLEAN DEFAULT FALSE; -- Flag to identify arena-generated posts

-- Indexes for arena-related queries
CREATE INDEX IF NOT EXISTS idx_community_posts_arena_thread_id ON community_posts(arena_thread_id);
CREATE INDEX IF NOT EXISTS idx_community_posts_arena_round_id ON community_posts(arena_round_id);
CREATE INDEX IF NOT EXISTS idx_community_posts_is_arena_post ON community_posts(is_arena_post);

-- Comments
COMMENT ON COLUMN community_posts.arena_thread_id IS 'Link to Thesis Arena thread if this post originated from Arena';
COMMENT ON COLUMN community_posts.arena_round_id IS 'Link to specific Arena round if applicable';
COMMENT ON COLUMN community_posts.arena_community IS 'Community perspective (value_investor, crypto_bro, etc.)';
COMMENT ON COLUMN community_posts.is_arena_post IS 'Flag indicating if this post was generated from Thesis Arena';

