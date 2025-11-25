"""
Database migration: Create fundamentals_cache table for dynamic universe expansion.

This table stores fundamentals data fetched from OpenBB for stocks not in our
initial parquet files. Allows the Alpha Picker universe to grow organically.
"""

import psycopg2
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
env_file = Path(__file__).parent.parent / "api" / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip()
    print(f"✅ Loaded environment from {env_file}")
else:
    print(f"⚠️  No .env file found at {env_file}, using system environment variables")

LOGGER = logging.getLogger(__name__)

def get_database_connection():
    """Get database connection using DATABASE_URL or fallback."""
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        return psycopg2.connect(database_url)
    
    # Fallback to individual vars
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB", "caria"),
        sslmode="require" if "neon.tech" in os.getenv("POSTGRES_HOST", "") else "prefer"
    )


def create_fundamentals_cache_table():
    """Create the fundamentals_cache table if it doesn't exist."""
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        # Drop existing table if we need to recreate it (for development)
        # Comment this out in production after first run
        # cursor.execute("DROP TABLE IF EXISTS fundamentals_cache;")
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals_cache (
                ticker VARCHAR(20) PRIMARY KEY,
                fetched_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                
                -- Quality metrics (profitability)
                roic FLOAT,
                roiic FLOAT,
                return_on_equity FLOAT,
                return_on_assets FLOAT,
                gross_profit_margin FLOAT,
                net_profit_margin FLOAT,
                free_cashflow_per_share FLOAT,
                free_cashflow_yield FLOAT,
                capital_expenditures FLOAT,
                r_and_d FLOAT,
                
                -- Value metrics (valuation + growth)
                price_to_book_ratio FLOAT,
                price_to_sales_ratio FLOAT,
                enterprise_value BIGINT,
                market_cap BIGINT,
                revenue_growth FLOAT,
                net_income_growth FLOAT,
                operating_income_growth FLOAT,
                total_debt BIGINT,
                cash_and_equivalents BIGINT,
                net_debt BIGINT,
                
                -- Additional metadata
                company_name VARCHAR(255),
                sector VARCHAR(100),
                industry VARCHAR(100),
                
                -- Raw data (JSONB for flexibility)
                raw_quality_data JSONB,
                raw_value_data JSONB,
                raw_financials JSONB,
                raw_growth_data JSONB,
                
                -- Data source tracking
                data_source VARCHAR(50) DEFAULT 'openbb',
                fetch_count INTEGER DEFAULT 1,
                last_error TEXT
            );
        """)
        
        # Create indexes for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker 
            ON fundamentals_cache(ticker);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fundamentals_updated 
            ON fundamentals_cache(updated_at DESC);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fundamentals_sector 
            ON fundamentals_cache(sector);
        """)
        
        conn.commit()
        LOGGER.info("✅ fundamentals_cache table created successfully")
        
    except Exception as e:
        conn.rollback()
        LOGGER.error(f"❌ Error creating fundamentals_cache table: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def verify_table_exists():
    """Verify that the table exists and show sample data."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'fundamentals_cache'
            );
        """)
        
        exists = cursor.fetchone()[0]
        
        if exists:
            # Get count
            cursor.execute("SELECT COUNT(*) FROM fundamentals_cache;")
            count = cursor.fetchone()[0]
            LOGGER.info(f"✅ fundamentals_cache table exists with {count} rows")
            
            # Show sample data if any
            if count > 0:
                cursor.execute("""
                    SELECT ticker, company_name, sector, fetched_at
                    FROM fundamentals_cache
                    ORDER BY updated_at DESC
                    LIMIT 5;
                """)
                rows = cursor.fetchall()
                LOGGER.info("Sample data:")
                for row in rows:
                    LOGGER.info(f"  {row}")
        else:
            LOGGER.warning("⚠️  fundamentals_cache table does not exist")
            
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("CREATING FUNDAMENTALS CACHE TABLE")
    print("=" * 60)
    
    create_fundamentals_cache_table()
    verify_table_exists()
    
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run FundamentalsCacheService to populate initial data")
    print("  2. Test with a new stock valuation")
    print("  3. Verify data appears in fundamentals_cache table")
