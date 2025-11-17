-- Caria Database Initialization Script
-- Multi-tenant system with authentication and RAG support

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- USER MANAGEMENT SCHEMA
-- ============================================================================

-- Users table (main authentication)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_active ON users(is_active);

-- API Keys (for programmatic access)
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    scopes TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_api_keys_user ON api_keys(user_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- Refresh Tokens (for JWT refresh mechanism)
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX idx_refresh_tokens_hash ON refresh_tokens(token_hash);

-- ============================================================================
-- RAG SYSTEM SCHEMA
-- ============================================================================

-- Document chunks with embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,  -- NULL = public knowledge
    content TEXT NOT NULL,
    embedding VECTOR(768),  -- Dimensión del embedding (ajustar según modelo)
    metadata JSONB,
    source VARCHAR(500),
    title VARCHAR(500),
    chunk_index INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_document_chunks_user ON document_chunks(user_id);
CREATE INDEX idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_document_chunks_metadata ON document_chunks USING gin(metadata);
CREATE INDEX idx_document_chunks_source ON document_chunks(source);

-- RAG queries history (for analytics)
CREATE TABLE IF NOT EXISTS rag_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(768),
    retrieved_chunk_ids UUID[],
    response_text TEXT,
    confidence_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rag_queries_user ON rag_queries(user_id);
CREATE INDEX idx_rag_queries_created ON rag_queries(created_at DESC);

-- ============================================================================
-- USER DATA SCHEMA (Multi-tenant)
-- ============================================================================

-- User portfolios (watchlists)
CREATE TABLE IF NOT EXISTS user_portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tickers TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, name)
);

CREATE INDEX idx_user_portfolios_user ON user_portfolios(user_id);

-- User analysis history
CREATE TABLE IF NOT EXISTS user_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ticker VARCHAR(20),
    analysis_type VARCHAR(50) NOT NULL,  -- 'dcf', 'multiples', 'scorecard', 'factors', 'regime'
    input_params JSONB,
    output_result JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_analyses_user ON user_analyses(user_id);
CREATE INDEX idx_user_analyses_ticker ON user_analyses(ticker);
CREATE INDEX idx_user_analyses_type ON user_analyses(analysis_type);
CREATE INDEX idx_user_analyses_created ON user_analyses(created_at DESC);

-- User preferences/settings
CREATE TABLE IF NOT EXISTS user_settings (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    preferred_regime VARCHAR(50),  -- 'expansion', 'slowdown', etc.
    risk_tolerance VARCHAR(50),    -- 'conservative', 'moderate', 'aggressive'
    notification_preferences JSONB,
    ui_preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- AUDIT LOG SCHEMA
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created ON audit_logs(created_at DESC);

-- ============================================================================
-- USAGE TRACKING (for billing/rate limiting)
-- ============================================================================

CREATE TABLE IF NOT EXISTS usage_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    tokens_used INTEGER,  -- For LLM calls
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_usage_metrics_user ON usage_metrics(user_id);
CREATE INDEX idx_usage_metrics_endpoint ON usage_metrics(endpoint);
CREATE INDEX idx_usage_metrics_created ON usage_metrics(created_at DESC);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_document_chunks_updated_at BEFORE UPDATE ON document_chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_portfolios_updated_at BEFORE UPDATE ON user_portfolios
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_settings_updated_at BEFORE UPDATE ON user_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- INITIAL DATA (Demo/Test User)
-- ============================================================================

-- Insert a demo user (password: "demo123")
-- Password hash generated with bcrypt
INSERT INTO users (email, username, full_name, hashed_password, is_active, is_verified)
VALUES (
    'demo@caria.com',
    'demo',
    'Demo User',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY0tSvHjqxokJoK',  -- "demo123"
    TRUE,
    TRUE
) ON CONFLICT (email) DO NOTHING;

-- Grant permissions (only if caria_user role exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'caria_user') THEN
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO caria_user;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO caria_user;
        RAISE NOTICE 'Permissions granted to caria_user';
    ELSE
        RAISE NOTICE 'Role caria_user does not exist, skipping permission grants';
    END IF;
END $$;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Caria database initialized successfully!';
    RAISE NOTICE 'Demo user created: email=demo@caria.com, password=demo123';
END $$;
