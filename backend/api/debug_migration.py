
import os
import psycopg2
import logging
from urllib.parse import urlparse
import sys

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    try:
        # Check env vars (do not print secrets)
        db_url = os.getenv("DATABASE_URL")
        logger.info(f"DATABASE_URL present: {bool(db_url)}")
        
        parsed = urlparse(db_url) if db_url else None
        
        logger.info("Attempting to connect...")
        
        if parsed and parsed.hostname:
             conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path.lstrip('/'),
                client_encoding='utf-8' # Force encoding
            )
        else:
            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                user=os.getenv("POSTGRES_USER", "caria_user"),
                password=os.getenv("POSTGRES_PASSWORD", "caria_password"),
                database=os.getenv("POSTGRES_DB", "caria"),
                client_encoding='utf-8'
            )
            
        logger.info("Connected successfully.")
        
        # Test query
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        logger.info(f"Query result: {cur.fetchone()}")
        conn.close()
        
    except UnicodeDecodeError as ue:
        # If we catch unicode error, print details
        logger.error(f"Unicode Decode Error caught! {repr(ue)}")
    except Exception as e:
        # Print representation to avoid decoding crash in str(e)
        logger.error(f"Connection failed (repr): {repr(e)}")

# Test migration SQL safely
    sql = """
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
    """
    # We skip stats table for brief test
    
if __name__ == "__main__":
    test_connection()
