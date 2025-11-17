-- Inicialización DB Wise Adviser

-- Habilitar extensiones
-- Nota: vector (pgvector) requiere instalación previa
-- Si no está instalado, comenta la siguiente línea o instala pgvector
-- Ver: https://github.com/pgvector/pgvector#installation
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
    RAISE NOTICE 'Extensión vector habilitada';
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'No se pudo habilitar vector (pgvector). Algunas funcionalidades pueden no estar disponibles. Error: %', SQLERRM;
END $$;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Tabla wisdom_chunks
-- Nota: La columna embedding requiere la extensión vector
-- Si vector no está disponible, esta tabla se creará sin la columna embedding
DO $$
BEGIN
    -- Verificar si la extensión vector está disponible
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        -- Crear tabla con columna embedding
        CREATE TABLE IF NOT EXISTS wisdom_chunks (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            text TEXT NOT NULL,
            source VARCHAR(100),
            themes TEXT[],
            context TEXT,
            embedding vector(1536),
            embedding_model VARCHAR(50) DEFAULT 'text-embedding-ada-002',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            index_version VARCHAR(20) DEFAULT 'v1'
        );
        RAISE NOTICE 'Tabla wisdom_chunks creada con columna embedding';
    ELSE
        -- Crear tabla sin columna embedding
        CREATE TABLE IF NOT EXISTS wisdom_chunks (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            text TEXT NOT NULL,
            source VARCHAR(100),
            themes TEXT[],
            context TEXT,
            -- embedding vector(1536),  -- Requiere extensión vector
            embedding_model VARCHAR(50) DEFAULT 'text-embedding-ada-002',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            index_version VARCHAR(20) DEFAULT 'v1'
        );
        RAISE WARNING 'Tabla wisdom_chunks creada SIN columna embedding (pgvector no disponible)';
    END IF;
EXCEPTION WHEN OTHERS THEN
    -- Si falla por cualquier razón, intentar crear sin embedding
    BEGIN
        CREATE TABLE IF NOT EXISTS wisdom_chunks (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            text TEXT NOT NULL,
            source VARCHAR(100),
            themes TEXT[],
            context TEXT,
            embedding_model VARCHAR(50) DEFAULT 'text-embedding-ada-002',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            index_version VARCHAR(20) DEFAULT 'v1'
        );
        RAISE WARNING 'Tabla wisdom_chunks creada sin embedding debido a error: %', SQLERRM;
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'No se pudo crear tabla wisdom_chunks: %', SQLERRM;
    END;
END $$;

-- Índices vectoriales (solo si la extensión vector está disponible)
-- Nota: Este índice requiere la extensión vector
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE INDEX IF NOT EXISTS idx_wisdom_embedding 
        ON wisdom_chunks 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        RAISE NOTICE 'Índice vectorial creado';
    ELSE
        RAISE WARNING 'Extensión vector no disponible, omitiendo índice vectorial';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'No se pudo crear índice vectorial. Error: %', SQLERRM;
END $$;

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

-- Tabla audit_logs para auditoría de acciones
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at DESC);

-- Tabla refresh_tokens para gestionar tokens JWT
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    revoked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_token_hash ON refresh_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires_at ON refresh_tokens(expires_at);

-- Tabla prediction_metrics para seguimiento de modelos
CREATE TABLE IF NOT EXISTS prediction_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    endpoint VARCHAR(100) NOT NULL,
    ticker VARCHAR(10),
    prediction_value JSONB,
    confidence FLOAT,
    processing_time_ms INTEGER,
    features_used JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prediction_metrics_model ON prediction_metrics(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_prediction_metrics_endpoint ON prediction_metrics(endpoint);
CREATE INDEX IF NOT EXISTS idx_prediction_metrics_created_at ON prediction_metrics(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_metrics_user_id ON prediction_metrics(user_id);

-- Tabla model_versions para versionado de modelos
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    training_date TIMESTAMP NOT NULL,
    validation_metrics JSONB,
    feature_hash VARCHAR(64),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_versions_model_name ON model_versions(model_name);
CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active) WHERE is_active = TRUE;

-- Table schema_migrations for tracking applied migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(255) UNIQUE NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_schema_migrations_name ON schema_migrations(migration_name);

-- Registrar migración inicial si aún no está guardada
INSERT INTO schema_migrations (migration_name)
SELECT '001_add_auth_tables'
WHERE NOT EXISTS (
    SELECT 1 FROM schema_migrations WHERE migration_name = '001_add_auth_tables'
);

-- Grant permissions (solo si el rol wise_user existe)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'wise_user') THEN
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO wise_user;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO wise_user;
    ELSE
        RAISE NOTICE 'Rol wise_user no existe, omitiendo GRANT';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'No se pudieron otorgar permisos a wise_user: %', SQLERRM;
END $$;
