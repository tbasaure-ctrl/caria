import os
import psycopg2
from urllib.parse import urlparse, parse_qs

def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    conn = None
    
    if database_url:
        try:
            parsed = urlparse(database_url)
            query_params = parse_qs(parsed.query)
            unix_socket_host = query_params.get('host', [None])[0]
            
            if unix_socket_host:
                conn = psycopg2.connect(
                    host=unix_socket_host,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
            elif parsed.hostname:
                conn = psycopg2.connect(
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
        except Exception as e:
            print(f"Error parsing DATABASE_URL: {e}")
    
    if conn is None:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "caria_user"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB", "caria"),
        )
    return conn

def create_tables():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            print("Creating model_portfolios table...")
            cur.execute("""
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
            """)
            
            print("Creating portfolio_performance table...")
            cur.execute("""
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
            """)
            conn.commit()
            print("Tables created successfully.")
            
            # Verify
            cur.execute("SELECT to_regclass('public.model_portfolios')")
            if cur.fetchone()[0]:
                print("VERIFICATION: model_portfolios exists.")
            else:
                print("VERIFICATION: model_portfolios DOES NOT exist.")

    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    create_tables()
