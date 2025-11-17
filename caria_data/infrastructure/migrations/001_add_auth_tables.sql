-- Migration 001: Add authentication and audit tables
-- Created: 2024-01-01

-- Ensure uuid-ossp extension is enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table audit_logs for tracking user actions
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

-- Table refresh_tokens for JWT refresh token management
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

-- Table prediction_metrics for tracking model performance
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

-- Table model_versions for model versioning
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

CREATE INDEX IF NOT EXISTS idx_schema_migrations_name ON schema_migrations(migration_name);

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

