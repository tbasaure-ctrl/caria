-- Migration: Add C-Score Metrics Table
-- Purpose: Store time-series and component-level data for new C-Score engine

CREATE TABLE IF NOT EXISTS c_score_metrics (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Quality Components (30% weight)
    roic_current DECIMAL(10,4),
    roic_vs_wacc DECIMAL(10,4),
    fcf_conversion_rate DECIMAL(10,4),
    operating_leverage DECIMAL(10,4),
    gross_margin DECIMAL(10,4),
    moat_score DECIMAL(10,4),
    c_quality_score DECIMAL(10,4),
    
    -- Delta Components (50% weight) - THE ALPHA ENGINE
    roic_yoy_change DECIMAL(10,4),
    roic_3yr_slope DECIMAL(10,4),
    fcf_growth_rate DECIMAL(10,4),
    fcf_to_ev DECIMAL(10,4),
    revenue_per_employee_growth DECIMAL(10,4),
    margin_expansion_streak INT,
    insider_ownership_change DECIMAL(10,4),
    capex_to_fcf_ratio DECIMAL(10,4),
    c_delta_score DECIMAL(10,4),
    
    -- Mispricing Components (20% weight)
    fcf_yield_current DECIMAL(10,4),
    fcf_yield_5yr_median DECIMAL(10,4),
    fcf_yield_vs_median DECIMAL(10,4),
    ev_to_sales_forward DECIMAL(10,4),
    growth_rate DECIMAL(10,4),
    short_interest_trend VARCHAR(20),
    analyst_revision_trend VARCHAR(20),
    mispricing_adjust DECIMAL(10,4),
    
    -- Final Score
    final_c_score DECIMAL(10,4),
    
    -- Metadata
    data_quality VARCHAR(20),  -- 'complete', 'partial', 'insufficient'
    error_message TEXT,
    
    UNIQUE(ticker)
);

-- Indexes for performance
CREATE INDEX idx_final_c_score ON c_score_metrics(final_c_score DESC);
CREATE INDEX idx_ticker ON c_score_metrics(ticker);
CREATE INDEX idx_c_delta ON c_score_metrics(c_delta_score DESC);
CREATE INDEX idx_updated_at ON c_score_metrics(updated_at DESC);

-- Comments for documentation
COMMENT ON TABLE c_score_metrics IS 'Stores component-level C-Score data for quality slope + mispricing engine';
COMMENT ON COLUMN c_score_metrics.c_quality_score IS 'Business durability score (0-100)';
COMMENT ON COLUMN c_score_metrics.c_delta_score IS 'Improvement momentum score (0-100) - the alpha engine';
COMMENT ON COLUMN c_score_metrics.mispricing_adjust IS 'Contrarian multiplier (0.5-1.5)';
COMMENT ON COLUMN c_score_metrics.final_c_score IS 'Final score: (Quality^0.6) * (Delta^1.2) * Mispricing, scaled to 0-1000';
