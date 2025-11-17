-- Migration 002: Add community module tables
-- Created: 2025-01-13
-- Per user requirements: community posts with Reddit-style voting (UP votes)

-- Ensure uuid-ossp extension is enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table community_posts for shared investment thesis/ideas
CREATE TABLE IF NOT EXISTS community_posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    thesis_preview TEXT NOT NULL,  -- First part of thesis (shown in list)
    full_thesis TEXT,  -- Full thesis (shown when clicked)
    ticker VARCHAR(10),  -- Stock ticker if applicable
    analysis_merit_score FLOAT DEFAULT 0.0,  -- Score based on analysis quality (0-1)
    upvotes INTEGER DEFAULT 0,  -- Count of upvotes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_community_posts_user_id ON community_posts(user_id);
CREATE INDEX IF NOT EXISTS idx_community_posts_ticker ON community_posts(ticker);
CREATE INDEX IF NOT EXISTS idx_community_posts_upvotes ON community_posts(upvotes DESC);
CREATE INDEX IF NOT EXISTS idx_community_posts_created_at ON community_posts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_community_posts_merit_score ON community_posts(analysis_merit_score DESC);

-- Table community_votes for tracking user votes (Reddit-style UP votes only)
CREATE TABLE IF NOT EXISTS community_votes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES community_posts(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    vote_type VARCHAR(10) DEFAULT 'up',  -- Only 'up' votes per requirements
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(post_id, user_id)  -- One vote per user per post
);

CREATE INDEX IF NOT EXISTS idx_community_votes_post_id ON community_votes(post_id);
CREATE INDEX IF NOT EXISTS idx_community_votes_user_id ON community_votes(user_id);

-- Function to update upvote count when a vote is added/removed
CREATE OR REPLACE FUNCTION update_post_upvotes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE community_posts
        SET upvotes = (
            SELECT COUNT(*) FROM community_votes
            WHERE post_id = NEW.post_id AND vote_type = 'up'
        )
        WHERE id = NEW.post_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE community_posts
        SET upvotes = (
            SELECT COUNT(*) FROM community_votes
            WHERE post_id = OLD.post_id AND vote_type = 'up'
        )
        WHERE id = OLD.post_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update upvote count
CREATE TRIGGER trigger_update_post_upvotes
AFTER INSERT OR DELETE ON community_votes
FOR EACH ROW EXECUTE FUNCTION update_post_upvotes();

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at
CREATE TRIGGER trigger_update_community_posts_updated_at
BEFORE UPDATE ON community_posts
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

