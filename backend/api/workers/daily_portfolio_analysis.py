"""
Daily Portfolio Analysis Worker per audit document (3.2).
Executes automatic daily analysis for all user portfolios.
Can be run as cron job or background task.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.services.portfolio_analytics import get_portfolio_analytics_service

LOGGER = logging.getLogger("caria.workers.daily_portfolio_analysis")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB", "caria"),
    )


def get_all_users_with_holdings(db_conn) -> list[dict]:
    """Get all users who have holdings."""
    with db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT DISTINCT u.id, u.username, u.email
            FROM users u
            INNER JOIN holdings h ON h.user_id = u.id
            WHERE u.is_active = TRUE
            GROUP BY u.id, u.username, u.email
            HAVING COUNT(h.id) > 0
            """
        )
        return [dict(row) for row in cursor.fetchall()]


def run_daily_analysis():
    """
    Run daily portfolio analysis for all users with holdings.
    Per audit document (3.2): Daily execution for all user portfolios.
    """
    LOGGER.info("Starting daily portfolio analysis job")

    db_conn = get_db_connection()
    analytics_service = get_portfolio_analytics_service()

    try:
        # Get all users with holdings
        users = get_all_users_with_holdings(db_conn)
        LOGGER.info(f"Found {len(users)} users with holdings")

        if not users:
            LOGGER.info("No users with holdings found. Exiting.")
            return

        success_count = 0
        error_count = 0

        for user in users:
            user_id = user["id"]
            username = user["username"]

            try:
                LOGGER.info(f"Analyzing portfolio for user {username} ({user_id})")

                # Analyze portfolio
                result = analytics_service.analyze_portfolio(
                    user_id=user_id,
                    db_connection=db_conn,
                    benchmark="SPY",  # Default benchmark
                )

                LOGGER.info(
                    f"✅ Successfully analyzed portfolio for {username}: "
                    f"Sharpe={result['metrics'].get('sharpe_ratio', 'N/A'):.3f}, "
                    f"CAGR={result['metrics'].get('cagr', 'N/A'):.2%}"
                )

                # Optionally: Store results in database or send notifications
                # TODO: Store metrics in portfolio_analytics table if needed

                success_count += 1

            except ValueError as e:
                LOGGER.warning(f"⚠️  Skipping user {username}: {e}")
                error_count += 1
            except Exception as e:
                LOGGER.error(f"❌ Error analyzing portfolio for {username}: {e}", exc_info=True)
                error_count += 1

        LOGGER.info(
            f"Daily analysis complete: {success_count} successful, {error_count} errors"
        )

    except Exception as e:
        LOGGER.error(f"Fatal error in daily analysis job: {e}", exc_info=True)
        raise
    finally:
        db_conn.close()


if __name__ == "__main__":
    """
    Run as standalone script.
    Usage:
        python -m api.workers.daily_portfolio_analysis
    
    Or schedule with cron:
        0 2 * * * cd /path/to/app && python -m api.workers.daily_portfolio_analysis
    """
    run_daily_analysis()

