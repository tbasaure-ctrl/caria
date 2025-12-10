
import os
import sys
import psycopg2
import pandas as pd
import logging
from datetime import date
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
import codecs

# Add parent directory to path to import caria_sr
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caria_sr import download_credit_series, compute_sr_for_ticker

# Load env vars from backend/api/.env
env_path = Path(__file__).resolve().parent.parent / 'api' / '.env'
load_dotenv(dotenv_path=env_path)

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of assets to track
ASSETS = ["SPY","QQQ","IWM","DIA","XLF","XLE","XLK","EFA","EEM","GLD", "NVDA", "AAPL", "MSFT", "TSLA", "AMZN"]

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
                    client_encoding='utf-8'
                )
        except Exception as e:
            logger.warning(f"Error using DATABASE_URL: {e}")
    
    # Fallback
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=os.getenv("POSTGRES_PASSWORD", "aria_password"), # Warning: Using fallback if env fails
        database=os.getenv("POSTGRES_DB", "caria"),
        client_encoding='utf-8'
    )

def main():
    logger.info("Starting CARIA-SR Update Job...")
    
    # 1. Download global credit series (HYG vol)
    logger.info("Downloading Credit Series (HYG)...")
    vol_credit = download_credit_series()
    if vol_credit is None:
        logger.error("Failed to download Credit Series (HYG). Aborting.")
        return

    try:
        conn = get_db_connection()
        cur = conn.cursor()
    except Exception as e:
        logger.error(f"Failed to connect to DB: {repr(e)}")
        return

    for ticker in ASSETS:
        logger.info(f"Processing {ticker}...")
        try:
            result = compute_sr_for_ticker(ticker, vol_credit)
            if result is None:
                logger.warning(f"No result for {ticker}")
                continue
            
            df, stats = result
            
            # Upsert Stats
            cur.execute("""
                INSERT INTO caria_sr_stats (ticker, auc, mean_normal, mean_fragile, last_updated)
                VALUES (%s,%s,%s,%s, now())
                ON CONFLICT (ticker) DO UPDATE
                    SET auc = EXCLUDED.auc,
                        mean_normal = EXCLUDED.mean_normal,
                        mean_fragile = EXCLUDED.mean_fragile,
                        last_updated = now();
            """, (ticker, stats["auc"], stats["mean_normal"], stats["mean_fragile"]))

            # Upsert Daily Series
            data_tuples = []
            for idx, row in df.iterrows():
                 # Handle NaN future_ret
                fut = row["future_ret_10d"]
                if pd.isna(fut):
                    fut = None
                
                data_tuples.append((
                    ticker,
                    idx.date(),
                    float(row["E4"]),
                    float(row["sync"]),
                    float(row["SR"]),
                    int(row["regime"]),
                    fut
                ))
            
            for dt in data_tuples:
                cur.execute("""
                    INSERT INTO caria_sr_daily (ticker, date, e4, sync, sr, regime, future_ret_10d)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (ticker, date) DO UPDATE
                        SET e4 = EXCLUDED.e4,
                            sync = EXCLUDED.sync,
                            sr = EXCLUDED.sr,
                            regime = EXCLUDED.regime,
                            future_ret_10d = EXCLUDED.future_ret_10d;
                """, dt)
                
            conn.commit() # Commit after each ticker
            logger.info(f"Updated {ticker} successfully.")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            conn.rollback()

    cur.close()
    conn.close()
    logger.info("CARIA-SR Update Job Completed.")

if __name__ == "__main__":
    main()
