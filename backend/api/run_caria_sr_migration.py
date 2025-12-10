
import os
import psycopg2
import logging
from urllib.parse import urlparse
import sys
import codecs
from dotenv import load_dotenv

# Load env vars from .env file (if exists in current or parent dirs)
load_dotenv()

# Override standard output to handle utf-8 even on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception:
        pass # If this fails, we just continue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            parsed = urlparse(database_url)
            if parsed.hostname:
                return psycopg2.connect(
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                    client_encoding='utf-8' # Force client encoding
                )
        except Exception as e:
            logger.warning(f"Error using DATABASE_URL: {e}")
    
    # Fallback
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=os.getenv("POSTGRES_PASSWORD", "caria_password"),
        database=os.getenv("POSTGRES_DB", "caria"),
        client_encoding='utf-8' # Force client encoding
    )

def run_migration():
    """Create CARIA-SR tables."""
    sql = """
    -- Daily series of CARIA-SR by ticker
    CREATE TABLE IF NOT EXISTS caria_sr_daily (
        id SERIAL PRIMARY KEY,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        e4 DOUBLE PRECISION NOT NULL,
        sync DOUBLE PRECISION NOT NULL,
        sr DOUBLE PRECISION NOT NULL,
        regime SMALLINT NOT NULL,
        future_ret_10d DOUBLE PRECISION,
        created_at TIMESTAMP DEFAULT now(),
        UNIQUE (ticker, date)
    );

    -- Optional: stats by ticker
    CREATE TABLE IF NOT EXISTS caria_sr_stats (
        ticker TEXT PRIMARY KEY,
        auc DOUBLE PRECISION,
        mean_normal DOUBLE PRECISION,
        mean_fragile DOUBLE PRECISION,
        last_updated TIMESTAMP NOT NULL
    );
    
    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_caria_sr_daily_ticker ON caria_sr_daily(ticker);
    CREATE INDEX IF NOT EXISTS idx_caria_sr_daily_date ON caria_sr_daily(date);
    """

    try:
        conn = get_db_connection()
        conn.set_client_encoding('UTF8') # Double force
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        logger.info("CARIA-SR tables created successfully.")
        conn.close()
    except Exception as e:
        # Use repr to prevent decoding errors in the error message itself
        logger.error(f"Migration failed (repr): {repr(e)}")

if __name__ == "__main__":
    run_migration()
