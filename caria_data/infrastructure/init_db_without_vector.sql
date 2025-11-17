-- Inicialización DB Wise Adviser (SIN extensión vector)
-- Usa este archivo si pgvector no está instalado
-- Las funcionalidades de RAG requerirán pgvector más adelante

-- Habilitar extensión uuid-ossp (requerida para UUIDs)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Tabla wisdom_chunks (sin columna embedding por ahora)
-- Nota: Para usar embeddings, instala pgvector y ejecuta init_db.sql completo
CREATE TABLE IF NOT EXISTS wisdom_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text TEXT NOT NULL,
    source VARCHAR(100),
    themes TEXT[],
    context TEXT,
    -- embedding vector(1536),  -- Comentado hasta que pgvector esté instalado
    embedding_model VARCHAR(50) DEFAULT 'text-embedding-ada-002',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    index_version VARCHAR(20) DEFAULT 'v1'
);

-- Índices básicos (sin índice vectorial)
CREATE INDEX IF NOT EXISTS idx_wisdom_themes 
ON wisdom_chunks 
USING gin (themes);

CREATE INDEX IF NOT EXISTS idx_wisdom_version 
ON wisdom_chunks (index_version);

-- Tabla prices
CREATE TABLE IF NOT EXISTS prices (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    adjusted_close FLOAT,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX idx_prices_date ON prices(date);
CREATE INDEX idx_prices_ticker_date ON prices(ticker, date DESC);

-- Tabla fundamentals
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    period_end DATE,
    revenue FLOAT,
    operating_income FLOAT,
    net_income FLOAT,
    free_cash_flow FLOAT,
    total_assets FLOAT,
    total_debt FLOAT,
    shareholders_equity FLOAT,
    roic FLOAT,
    reinvestment_rate FLOAT,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX idx_fundamentals_ticker_period ON fundamentals(ticker, period_end);

-- Tabla macro_indicators
CREATE TABLE IF NOT EXISTS macro_indicators (
    indicator VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    value FLOAT,
    source VARCHAR(20) DEFAULT 'FRED',
    PRIMARY KEY (indicator, date)
);

CREATE INDEX idx_macro_date ON macro_indicators(date);

-- Tabla predictions
CREATE TABLE IF NOT EXISTS predictions (
    ticker VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    predicted_regime VARCHAR(20),
    regime_probabilities JSONB,
    predicted_return FLOAT,
    drawdown_probability FLOAT,
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, prediction_date, model_version)
);

CREATE INDEX idx_predictions_date ON predictions(prediction_date);
CREATE INDEX idx_predictions_ticker_target ON predictions(ticker, target_date);

-- Tabla processed_features
CREATE TABLE IF NOT EXISTS processed_features (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    macro_regime VARCHAR(20),
    yield_curve_slope FLOAT,
    vix FLOAT,
    market_breadth FLOAT,
    sentiment_score FLOAT,
    roic_trend FLOAT,
    reinvestment_quality FLOAT,
    feature_version VARCHAR(20),
    PRIMARY KEY (ticker, date)
);

CREATE INDEX idx_features_date ON processed_features(date);
CREATE INDEX idx_features_regime ON processed_features(macro_regime);

-- Base de datos para MLflow
CREATE DATABASE mlflow_db;

-- Table users for authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_active ON users(is_active);

-- Table schema_migrations for tracking applied migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(255) UNIQUE NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_schema_migrations_name ON schema_migrations(migration_name);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO wise_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO wise_user;

