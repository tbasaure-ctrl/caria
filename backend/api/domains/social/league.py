"""
League Service - Handles Global Portfolio League logic.
Calculates daily scores based on Sharpe, CAGR, Drawdown, Diversification, and Account Age.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from caria.ingestion.clients.fmp_client import FMPClient
# Assuming we have a way to get historical prices, if not we might need to use FMPClient directly or a service
# from api.services.market_data import MarketDataService 

logger = logging.getLogger(__name__)

class LeagueService:
    def __init__(self):
        self.fmp_client = FMPClient()

    def calculate_daily_scores(self, db_conn, date: datetime.date = None):
        """
        Calculate scores for all users for a specific date.
        If date is None, defaults to yesterday (since we need full day data).
        """
        if date is None:
            date = datetime.utcnow().date() - timedelta(days=1)
            
        logger.info(f"Starting league score calculation for {date}")
        
        try:
            # 1. Get all active users with holdings
            users = self._get_users_with_holdings(db_conn)
            logger.info(f"Found {len(users)} users with holdings")
            
            scores = []
            
            for user in users:
                try:
                    score_data = self._calculate_user_score(db_conn, user['id'], date)
                    if score_data:
                        scores.append(score_data)
                except Exception as e:
                    logger.error(f"Error calculating score for user {user['id']}: {e}")
                    continue
            
            # 2. Calculate ranks and percentiles
            if not scores:
                logger.warning("No scores calculated")
                return

            df_scores = pd.DataFrame(scores)
            df_scores['rank'] = df_scores['score'].rank(ascending=False, method='min')
            df_scores['percentile'] = df_scores['score'].rank(pct=True) * 100
            
            # 3. Save to database
            self._save_rankings(db_conn, df_scores, date)
            logger.info(f"Saved {len(df_scores)} rankings for {date}")
            
        except Exception as e:
            logger.error(f"League calculation failed: {e}")
            raise

    def get_leaderboard(self, db_conn, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get the latest leaderboard."""
        with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get latest date
            cur.execute("SELECT MAX(date) as max_date FROM league_rankings")
            row = cur.fetchone()
            if not row or not row['max_date']:
                return []
            
            latest_date = row['max_date']
            
            cur.execute("""
                SELECT 
                    lr.rank,
                    u.username,
                    u.id as user_id,
                    lr.score,
                    lr.sharpe_ratio,
                    lr.cagr,
                    lr.max_drawdown,
                    lr.diversification_score,
                    lr.account_age_days
                FROM league_rankings lr
                JOIN users u ON lr.user_id = u.id
                WHERE lr.date = %s
                ORDER BY lr.rank ASC
                LIMIT %s OFFSET %s
            """, (latest_date, limit, offset))
            
            return cur.fetchall()

    def get_user_profile(self, db_conn, user_id: UUID) -> Dict[str, Any]:
        """Get a user's league profile including history."""
        with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Current rank
            cur.execute("""
                SELECT * FROM league_rankings 
                WHERE user_id = %s 
                ORDER BY date DESC 
                LIMIT 1
            """, (str(user_id),))
            current = cur.fetchone()
            
            # History (last 30 days)
            cur.execute("""
                SELECT date, score, rank 
                FROM league_rankings 
                WHERE user_id = %s 
                ORDER BY date ASC
                LIMIT 30
            """, (str(user_id),))
            history = cur.fetchall()
            
            return {
                "current": current,
                "history": history
            }

    def _get_users_with_holdings(self, db_conn) -> List[Dict]:
        with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT user_id as id 
                FROM holdings
            """)
            return cur.fetchall()

    def _calculate_user_score(self, db_conn, user_id: UUID, date: datetime.date) -> Optional[Dict]:
        """
        Calculate the composite score for a single user.
        Formula: 0.35 * Sharpe + 0.30 * CAGR + 0.15 * (1 - MaxDrawdown) + 0.10 * Div + 0.10 * Age
        """
        # 1. Get Holdings
        with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT ticker, quantity FROM holdings WHERE user_id = %s", (str(user_id),))
            holdings = cur.fetchall()
            
        if not holdings:
            return None

        # 2. Get Historical Data (Mocking logic for now, ideally use PortfolioAnalyticsService)
        # In a real implementation, we'd fetch 1 year of history for these tickers
        # and reconstruct the portfolio value series.
        
        # Placeholder metrics for MVP/Verification
        # We should try to use the existing PortfolioAnalyticsService if possible, 
        # but it might be slow for batch processing.
        
        # Let's assume we have a helper or we do a simplified calculation.
        # For now, I will generate semi-random metrics based on "mock" performance 
        # to ensure the pipeline works, then we refine the math.
        # TODO: Connect to real PortfolioAnalyticsService
        
        import random
        sharpe = random.uniform(0.5, 3.0)
        cagr = random.uniform(0.05, 0.40)
        max_dd = random.uniform(0.05, 0.30)
        diversification = min(len(holdings) * 10, 100) / 100.0 # Simple count based
        account_age_days = random.randint(30, 365)
        account_age_factor = min(account_age_days / 365.0, 1.0)

        # Normalize metrics for scoring (0-1 scale roughly)
        # Sharpe: >3 is amazing (1.0), <0 is bad (0.0)
        norm_sharpe = min(max(sharpe / 3.0, 0), 1.0)
        
        # CAGR: >30% is amazing (1.0)
        norm_cagr = min(max(cagr / 0.30, 0), 1.0)
        
        # Drawdown: 0 is best (1.0), >50% is bad (0.0)
        norm_dd = max(1.0 - (max_dd / 0.50), 0)
        
        score = (
            0.35 * norm_sharpe +
            0.30 * norm_cagr +
            0.15 * norm_dd +
            0.10 * diversification +
            0.10 * account_age_factor
        ) * 1000  # Scale to 0-1000

        return {
            "user_id": user_id,
            "date": date,
            "score": score,
            "sharpe_ratio": sharpe,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "diversification_score": diversification * 100, # Display as 0-100
            "account_age_days": account_age_days
        }

    def _save_rankings(self, db_conn, df_scores: pd.DataFrame, date: datetime.date):
        with db_conn.cursor() as cur:
            for _, row in df_scores.iterrows():
                cur.execute("""
                    INSERT INTO league_rankings 
                    (user_id, date, score, sharpe_ratio, cagr, max_drawdown, 
                     diversification_score, account_age_days, rank, percentile)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, date) DO UPDATE SET
                        score = EXCLUDED.score,
                        rank = EXCLUDED.rank,
                        percentile = EXCLUDED.percentile
                """, (
                    str(row['user_id']),
                    date,
                    row['score'],
                    row['sharpe_ratio'],
                    row['cagr'],
                    row['max_drawdown'],
                    row['diversification_score'],
                    row['account_age_days'],
                    int(row['rank']),
                    row['percentile']
                ))
            db_conn.commit()
