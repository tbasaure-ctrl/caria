-- Migración: Agregar tabla de holdings de usuarios
-- Fecha: 2025-01-XX
-- Idempotente: puede ejecutarse múltiples veces sin errores

CREATE TABLE IF NOT EXISTS holdings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    quantity DECIMAL(15, 4) NOT NULL DEFAULT 0,
    average_cost DECIMAL(15, 4) NOT NULL DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_holdings_user_id ON holdings(user_id);
CREATE INDEX IF NOT EXISTS idx_holdings_ticker ON holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_holdings_user_ticker ON holdings(user_id, ticker);

-- Trigger para actualizar updated_at automáticamente
CREATE OR REPLACE FUNCTION update_holdings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Eliminar trigger si existe antes de crearlo
DROP TRIGGER IF EXISTS trigger_holdings_updated_at ON holdings;

CREATE TRIGGER trigger_holdings_updated_at
    BEFORE UPDATE ON holdings
    FOR EACH ROW
    EXECUTE FUNCTION update_holdings_updated_at();

-- Comentarios
COMMENT ON TABLE holdings IS 'Posiciones de acciones de usuarios';
COMMENT ON COLUMN holdings.user_id IS 'ID del usuario propietario';
COMMENT ON COLUMN holdings.ticker IS 'Símbolo de la acción';
COMMENT ON COLUMN holdings.quantity IS 'Cantidad de acciones';
COMMENT ON COLUMN holdings.average_cost IS 'Costo promedio por acción';

