-- GDELT Market Intelligence Schema
-- Migration: 006_add_gdelt_tables.sql

-- Raw GDELT stories
CREATE TABLE IF NOT EXISTS gdelt_news_raw (
    id BIGSERIAL PRIMARY KEY,
    gdelt_id TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,          -- 'gdelt_doc', 'gdelt_gkg'
    source_domain TEXT,             -- news outlet
    title TEXT,
    body TEXT,                      -- full article text (if fetched)
    url TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    language TEXT,
    -- GDELT-specific fields
    tone REAL,                      -- GKG "Tone" (-100 to +100)
    themes TEXT[],                  -- GDELT themes
    locations TEXT[],               -- country codes / locations
    persons TEXT[],                 -- PERSONS from GKG
    organizations TEXT[],           -- ORGS from GKG
    meta JSONB
);

-- Entity to ticker mapping
CREATE TABLE IF NOT EXISTS entity_ticker_map (
    id BIGSERIAL PRIMARY KEY,
    entity TEXT NOT NULL,           -- 'APPLE INC', 'TESLA', 'FEDERAL RESERVE'
    ticker TEXT NOT NULL,           -- 'AAPL', 'TSLA', 'SPY', 'XLF', etc.
    weight REAL NOT NULL DEFAULT 1.0,
    kind TEXT,                      -- 'company', 'index', 'sector', 'country'
    UNIQUE(entity, ticker)
);

-- Many-to-many from news to ticker
CREATE TABLE IF NOT EXISTS gdelt_news_tickers (
    news_id BIGINT REFERENCES gdelt_news_raw(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    mapping_source TEXT,            -- 'entity_match', 'keyword', etc.
    PRIMARY KEY (news_id, ticker)
);

-- Sentiment analysis results
CREATE TABLE IF NOT EXISTS gdelt_news_sentiment (
    news_id BIGINT REFERENCES gdelt_news_raw(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,       -- 'finbert_v1', 'llm_macro_v1', 'gdelt_tone'
    sentiment REAL NOT NULL,        -- -1..+1
    confidence REAL,
    relevance REAL,                 -- 0..1
    topic TEXT,                     -- 'macro', 'earnings', 'regulation', etc.
    PRIMARY KEY (news_id, model_name)
);

-- Daily macro sentiment indices
CREATE TABLE IF NOT EXISTS daily_macro_sentiment (
    date DATE NOT NULL,
    region TEXT NOT NULL,           -- 'US', 'CN', 'EU', 'GLOBAL', etc.
    tone_mean REAL,
    tone_std REAL,
    finbert_mean REAL,
    article_count INT,
    PRIMARY KEY (date, region)
);

-- Daily ticker sentiment (combines news + social)
CREATE TABLE IF NOT EXISTS daily_ticker_sentiment (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    news_tone_mean REAL,
    news_tone_std REAL,
    news_count INT,
    news_sent_mean REAL,
    news_sent_std REAL,
    social_sent_mean REAL,
    social_sent_std REAL,
    social_count INT,
    PRIMARY KEY (date, ticker)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_gdelt_news_published ON gdelt_news_raw(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_gdelt_news_source ON gdelt_news_raw(source_domain);
CREATE INDEX IF NOT EXISTS idx_gdelt_news_tickers_ticker ON gdelt_news_tickers(ticker);
CREATE INDEX IF NOT EXISTS idx_gdelt_sentiment_ticker ON daily_ticker_sentiment(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_macro_sentiment_date ON daily_macro_sentiment(date DESC, region);
