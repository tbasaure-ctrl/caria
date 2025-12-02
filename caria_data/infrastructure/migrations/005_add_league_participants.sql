-- Create league_participants table to track opt-in status and anonymity
CREATE TABLE IF NOT EXISTS league_participants (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    is_anonymous BOOLEAN DEFAULT FALSE,
    display_name TEXT, -- Optional custom name, otherwise use username or "Anonymous"
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster joins
CREATE INDEX IF NOT EXISTS idx_league_participants_joined ON league_participants(joined_at);
