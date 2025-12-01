import os
import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply():
    sql = """
    CREATE TABLE IF NOT EXISTS league_rankings (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        date DATE NOT NULL,
        score DECIMAL(10, 4) NOT NULL,
        sharpe_ratio DECIMAL(10, 4),
        cagr DECIMAL(10, 4),
        max_drawdown DECIMAL(10, 4),
        diversification_score DECIMAL(10, 4),
        account_age_days INTEGER,
        rank INTEGER,
        percentile DECIMAL(5, 2),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, date)
    );
    CREATE INDEX IF NOT EXISTS idx_league_rankings_date ON league_rankings(date);
    CREATE INDEX IF NOT EXISTS idx_league_rankings_score ON league_rankings(score DESC);
    CREATE INDEX IF NOT EXISTS idx_league_rankings_user_date ON league_rankings(user_id, date);
    """
    
    # Connection logic
    database_url = os.getenv("DATABASE_URL")
    
    # Set encoding for console output to avoid Windows Unicode errors
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    try:
        if database_url:
            print("Connecting using DATABASE_URL...")
            conn = psycopg2.connect(database_url)
        else:
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = int(os.getenv("POSTGRES_PORT", "5432"))
            user = os.getenv("POSTGRES_USER", "caria_user")
            password = os.getenv("POSTGRES_PASSWORD", "caria_password")
            dbname = os.getenv("POSTGRES_DB", "caria")
            print(f"Connecting to {host}:{port}/{dbname} as {user}...")
            conn = psycopg2.connect(host=host, port=port, user=user, password=password, database=dbname)
            
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)
        print("Migration success!")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    apply()
