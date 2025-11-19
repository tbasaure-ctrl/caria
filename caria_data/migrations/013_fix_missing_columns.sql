-- Migration 013: Fix missing columns and tables
-- Purpose: Add missing columns and ensure all required tables exist

-- Add revoked column to refresh_tokens if it doesn't exist
ALTER TABLE refresh_tokens
ADD COLUMN IF NOT EXISTS revoked BOOLEAN DEFAULT FALSE;

-- Create index on revoked column for performance
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_revoked ON refresh_tokens(revoked) WHERE revoked = TRUE;

-- Ensure thesis_arena_threads table exists (from migration 010)
CREATE TABLE IF NOT EXISTS thesis_arena_threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    initial_thesis TEXT NOT NULL,
    ticker VARCHAR(10),
    initial_conviction FLOAT NOT NULL CHECK (initial_conviction >= 0 AND initial_conviction <= 100),
    current_conviction FLOAT NOT NULL CHECK (current_conviction >= 0 AND current_conviction <= 100),
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'archived')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Ensure arena_rounds table exists
CREATE TABLE IF NOT EXISTS arena_rounds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES thesis_arena_threads(id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL CHECK (round_number > 0),
    user_message TEXT,
    community_responses JSONB NOT NULL,
    conviction_before FLOAT NOT NULL CHECK (conviction_before >= 0 AND conviction_before <= 100),
    conviction_after FLOAT NOT NULL CHECK (conviction_after >= 0 AND conviction_after <= 100),
    conviction_change FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(thread_id, round_number)
);

-- Ensure community_posts has arena fields (from migration 011)
ALTER TABLE community_posts
ADD COLUMN IF NOT EXISTS arena_thread_id UUID REFERENCES thesis_arena_threads(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS arena_round_id UUID REFERENCES arena_rounds(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS arena_community VARCHAR(50),
ADD COLUMN IF NOT EXISTS is_arena_post BOOLEAN DEFAULT FALSE;

-- Create indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_thesis_arena_threads_user_id ON thesis_arena_threads(user_id);
CREATE INDEX IF NOT EXISTS idx_thesis_arena_threads_status ON thesis_arena_threads(status);
CREATE INDEX IF NOT EXISTS idx_arena_rounds_thread_id ON arena_rounds(thread_id);
CREATE INDEX IF NOT EXISTS idx_community_posts_arena_thread_id ON community_posts(arena_thread_id);
CREATE INDEX IF NOT EXISTS idx_community_posts_is_arena_post ON community_posts(is_arena_post);

-- Ensure model_retraining_triggers table exists (if needed by model_portfolio)
CREATE TABLE IF NOT EXISTS model_retraining_triggers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    trigger_type VARCHAR(50) NOT NULL,
    threshold_value FLOAT,
    current_value FLOAT,
    triggered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'resolved', 'ignored')),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_retraining_triggers_model_name ON model_retraining_triggers(model_name);
CREATE INDEX IF NOT EXISTS idx_model_retraining_triggers_status ON model_retraining_triggers(status);
CREATE INDEX IF NOT EXISTS idx_model_retraining_triggers_triggered_at ON model_retraining_triggers(triggered_at DESC);

-- Comments
COMMENT ON COLUMN refresh_tokens.revoked IS 'Flag indicating if the refresh token has been revoked';
COMMENT ON TABLE thesis_arena_threads IS 'Stores Thesis Arena challenge sessions';
COMMENT ON TABLE arena_rounds IS 'Stores individual conversation rounds in Thesis Arena';
COMMENT ON TABLE model_retraining_triggers IS 'Tracks when model retraining should be triggered based on performance thresholds';

