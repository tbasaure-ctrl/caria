import os
import psycopg2
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import codecs
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def get_db_connection():
    """Get database connection."""
    from urllib.parse import urlparse
    
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
    )

def run_migration():
    """Run the migration to add league tables."""
    # Embedded SQL to avoid encoding issues
    sql_text = """
    -- Create league_rankings table
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

    -- Create league_participants table
    CREATE TABLE IF NOT EXISTS league_participants (
        user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        is_anonymous BOOLEAN DEFAULT FALSE,
        display_name TEXT,
        joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_league_rankings_date ON league_rankings(date);
    CREATE INDEX IF NOT EXISTS idx_league_rankings_score ON league_rankings(score DESC);
    CREATE INDEX IF NOT EXISTS idx_league_rankings_user_date ON league_rankings(user_id, date);
    CREATE INDEX IF NOT EXISTS idx_league_participants_joined ON league_participants(joined_at);
    """

    try:
        conn = get_db_connection()
        conn.autocommit = False
        
        logger.info("Applying migration (embedded SQL)...")
        
        with conn.cursor() as cursor:
            cursor.execute(sql_text)
        
        conn.commit()
        logger.info("Migration applied successfully!")
        
    except Exception as e:
        logger.error(f"Error applying migration: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    run_migration()
