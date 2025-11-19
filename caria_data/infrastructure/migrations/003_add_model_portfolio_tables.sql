-- Create model_portfolios table
CREATE TABLE IF NOT EXISTS model_portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    selection_type VARCHAR(50) NOT NULL,
    regime VARCHAR(50),
    holdings JSONB NOT NULL,
    total_holdings INTEGER NOT NULL,
    initial_value DECIMAL(15, 2) NOT NULL DEFAULT 10000.00,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create portfolio_performance table
CREATE TABLE IF NOT EXISTS portfolio_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES model_portfolios(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    portfolio_value DECIMAL(15, 2) NOT NULL,
    portfolio_return_pct DECIMAL(10, 4) NOT NULL,
    benchmark_sp500_return_pct DECIMAL(10, 4),
    benchmark_qqq_return_pct DECIMAL(10, 4),
    benchmark_vti_return_pct DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown_pct DECIMAL(10, 4),
    volatility_pct DECIMAL(10, 4),
    alpha_pct DECIMAL(10, 4),
    beta DECIMAL(10, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(portfolio_id, date)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_model_portfolios_status ON model_portfolios(status);
CREATE INDEX IF NOT EXISTS idx_model_portfolios_selection_type ON model_portfolios(selection_type);
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_portfolio_id ON portfolio_performance(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_date ON portfolio_performance(date);
