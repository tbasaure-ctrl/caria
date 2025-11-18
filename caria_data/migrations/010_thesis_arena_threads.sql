-- Migration: Create thesis_arena_threads and arena_rounds tables
-- Purpose: Support multi-round conversations in Thesis Arena

-- Table: thesis_arena_threads
-- Stores arena challenge sessions with initial thesis and conviction
CREATE TABLE IF NOT EXISTS thesis_arena_threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    thesis TEXT NOT NULL,
    ticker VARCHAR(10),
    initial_conviction FLOAT NOT NULL CHECK (initial_conviction >= 0 AND initial_conviction <= 100),
    current_conviction FLOAT NOT NULL CHECK (current_conviction >= 0 AND current_conviction <= 100),
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'archived')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT thesis_length CHECK (char_length(thesis) >= 10 AND char_length(thesis) <= 2000)
);

-- Table: arena_rounds
-- Stores individual rounds of conversation with communities
CREATE TABLE IF NOT EXISTS arena_rounds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES thesis_arena_threads(id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL CHECK (round_number > 0),
    user_message TEXT,
    community_responses JSONB NOT NULL, -- { "value_investor": "...", "crypto_bro": "...", ... }
    conviction_before FLOAT NOT NULL CHECK (conviction_before >= 0 AND conviction_before <= 100),
    conviction_after FLOAT NOT NULL CHECK (conviction_after >= 0 AND conviction_after <= 100),
    conviction_change FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(thread_id, round_number)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_thesis_arena_threads_user_id ON thesis_arena_threads(user_id);
CREATE INDEX IF NOT EXISTS idx_thesis_arena_threads_status ON thesis_arena_threads(status);
CREATE INDEX IF NOT EXISTS idx_thesis_arena_threads_created_at ON thesis_arena_threads(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_arena_rounds_thread_id ON arena_rounds(thread_id);
CREATE INDEX IF NOT EXISTS idx_arena_rounds_thread_round ON arena_rounds(thread_id, round_number);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_thesis_arena_threads_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
CREATE TRIGGER trigger_update_thesis_arena_threads_updated_at
    BEFORE UPDATE ON thesis_arena_threads
    FOR EACH ROW
    EXECUTE FUNCTION update_thesis_arena_threads_updated_at();

-- Comments
COMMENT ON TABLE thesis_arena_threads IS 'Stores Thesis Arena challenge sessions';
COMMENT ON TABLE arena_rounds IS 'Stores individual conversation rounds in Thesis Arena';
COMMENT ON COLUMN arena_rounds.community_responses IS 'JSON object mapping community names to their responses';

