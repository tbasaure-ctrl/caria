-- Create league_rankings table
CREATE TABLE IF NOT EXISTS league_rankings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    score DECIMAL(10, 4) NOT NULL,
    sharpe_ratio DECIMAL(10, 4),
    cagr DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    diversification_score DECIMAL(10, 4),
    account_age_days INTEGER,
    rank INTEGER,
    percentile DECIMAL(5, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, date)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_league_rankings_date ON league_rankings(date);
CREATE INDEX IF NOT EXISTS idx_league_rankings_score ON league_rankings(score DESC);
CREATE INDEX IF NOT EXISTS idx_league_rankings_user_date ON league_rankings(user_id, date);
