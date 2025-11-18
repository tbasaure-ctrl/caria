-- Migration 012: Model Portfolio Selection and Tracking
-- Creates tables for tracking model-selected portfolios and their performance

-- Table: model_portfolios
-- Stores portfolios selected by the model for validation
CREATE TABLE IF NOT EXISTS model_portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    selection_type VARCHAR(50) NOT NULL CHECK (selection_type IN ('outlier', 'balanced', 'random')),
    regime VARCHAR(50),
    holdings JSONB NOT NULL, -- Array of {ticker: string, allocation: float}
    total_holdings INTEGER NOT NULL CHECK (total_holdings >= 10 AND total_holdings <= 20),
    initial_value NUMERIC(15, 2) DEFAULT 10000.00,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'archived')),
    notes TEXT,
    CONSTRAINT valid_holdings CHECK (jsonb_array_length(holdings) = total_holdings)
);

-- Index for querying active portfolios
CREATE INDEX IF NOT EXISTS idx_model_portfolios_status ON model_portfolios(status);
CREATE INDEX IF NOT EXISTS idx_model_portfolios_selection_type ON model_portfolios(selection_type);
CREATE INDEX IF NOT EXISTS idx_model_portfolios_created_at ON model_portfolios(created_at DESC);

-- Table: portfolio_performance
-- Tracks daily/weekly performance of model portfolios vs benchmarks
CREATE TABLE IF NOT EXISTS portfolio_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES model_portfolios(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    portfolio_value NUMERIC(15, 2) NOT NULL,
    portfolio_return_pct NUMERIC(10, 4) NOT NULL, -- Cumulative return %
    benchmark_sp500_return_pct NUMERIC(10, 4), -- S&P 500 return %
    benchmark_qqq_return_pct NUMERIC(10, 4), -- QQQ return %
    benchmark_vti_return_pct NUMERIC(10, 4), -- VTI return %
    sharpe_ratio NUMERIC(10, 4),
    max_drawdown_pct NUMERIC(10, 4),
    volatility_pct NUMERIC(10, 4),
    alpha_pct NUMERIC(10, 4), -- Alpha vs S&P 500
    beta NUMERIC(10, 4), -- Beta vs S&P 500
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(portfolio_id, date)
);

-- Indexes for performance queries
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_portfolio_id ON portfolio_performance(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_date ON portfolio_performance(date DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_portfolio_date ON portfolio_performance(portfolio_id, date DESC);

-- Table: model_retraining_triggers
-- Logs when retraining is triggered based on performance thresholds
CREATE TABLE IF NOT EXISTS model_retraining_triggers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    trigger_reason TEXT NOT NULL,
    portfolios_analyzed INTEGER NOT NULL,
    average_underperformance_pct NUMERIC(10, 4),
    threshold_met BOOLEAN DEFAULT FALSE,
    retraining_status VARCHAR(50) DEFAULT 'pending' CHECK (retraining_status IN ('pending', 'in_progress', 'completed', 'failed')),
    retraining_completed_at TIMESTAMP WITH TIME ZONE,
    notes TEXT
);

-- Index for retraining triggers
CREATE INDEX IF NOT EXISTS idx_retraining_triggers_status ON model_retraining_triggers(retraining_status);
CREATE INDEX IF NOT EXISTS idx_retraining_triggers_triggered_at ON model_retraining_triggers(triggered_at DESC);

COMMENT ON TABLE model_portfolios IS 'Portfolios selected by the model for validation and tracking';
COMMENT ON TABLE portfolio_performance IS 'Daily/weekly performance tracking of model portfolios vs benchmarks';
COMMENT ON TABLE model_retraining_triggers IS 'Logs when model retraining is triggered based on performance thresholds';

