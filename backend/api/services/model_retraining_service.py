"""
Model Retraining Service.
Analyzes portfolio performance and triggers retraining when threshold is met.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from urllib.parse import urlparse, parse_qs

LOGGER = logging.getLogger("caria.services.model_retraining")


class ModelRetrainingService:
    """
    Service to analyze portfolio performance and trigger retraining.
    
    Retraining is triggered when:
    - Average underperformance exceeds threshold (e.g., -5% vs S&P 500)
    - Multiple portfolios consistently underperform
    - Performance metrics degrade significantly
    """

    def __init__(self) -> None:
        self.underperformance_threshold_pct = -5.0  # -5% vs benchmark
        self.min_portfolios_analyzed = 5  # Minimum portfolios to analyze
        self.min_days_tracking = 30  # Minimum days of tracking data

    def _get_db_connection(self):
        """Get database connection using DATABASE_URL or fallback."""
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
                LOGGER.warning(f"Error using DATABASE_URL: {e}")
        
        if conn is None:
            password = os.getenv("POSTGRES_PASSWORD")
            if not password:
                raise RuntimeError("POSTGRES_PASSWORD environment variable is required")
            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                user=os.getenv("POSTGRES_USER", "caria_user"),
                password=password,
                database=os.getenv("POSTGRES_DB", "caria"),
            )
        
        return conn

    def analyze_performance(self, days_back: int = 90) -> Dict[str, any]:
        """
        Analyze performance of all active model portfolios.
        
        Args:
            days_back: Number of days to look back for analysis
            
        Returns:
            Dictionary with analysis results and retraining recommendation
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get active portfolios
                cursor.execute(
                    """
                    SELECT id, selection_type, created_at
                    FROM model_portfolios
                    WHERE status = 'active'
                    ORDER BY created_at DESC
                    """
                )
                portfolios = cursor.fetchall()
                
                if len(portfolios) < self.min_portfolios_analyzed:
                    return {
                        "should_retrain": False,
                        "reason": f"Insufficient portfolios: {len(portfolios)} < {self.min_portfolios_analyzed}",
                        "portfolios_analyzed": len(portfolios),
                        "average_underperformance_pct": None,
                    }
                
                # Get latest performance for each portfolio
                cutoff_date = date.today() - timedelta(days=days_back)
                portfolio_performances = []
                
                for portfolio in portfolios:
                    cursor.execute(
                        """
                        SELECT 
                            portfolio_return_pct,
                            benchmark_sp500_return_pct,
                            alpha_pct,
                            date
                        FROM portfolio_performance
                        WHERE portfolio_id = %s
                            AND date >= %s
                        ORDER BY date DESC
                        LIMIT 1
                        """
                        ,
                        (str(portfolio["id"]), cutoff_date),
                    )
                    perf = cursor.fetchone()
                    
                    if perf and perf.get("benchmark_sp500_return_pct") is not None:
                        underperformance = perf["portfolio_return_pct"] - perf["benchmark_sp500_return_pct"]
                        portfolio_performances.append({
                            "portfolio_id": str(portfolio["id"]),
                            "selection_type": portfolio["selection_type"],
                            "portfolio_return": perf["portfolio_return_pct"],
                            "benchmark_return": perf["benchmark_sp500_return_pct"],
                            "underperformance": underperformance,
                            "alpha": perf.get("alpha_pct"),
                            "date": perf["date"],
                        })
                
                if len(portfolio_performances) < self.min_portfolios_analyzed:
                    return {
                        "should_retrain": False,
                        "reason": f"Insufficient performance data: {len(portfolio_performances)} portfolios",
                        "portfolios_analyzed": len(portfolio_performances),
                        "average_underperformance_pct": None,
                    }
                
                # Calculate average underperformance
                avg_underperformance = sum(p["underperformance"] for p in portfolio_performances) / len(portfolio_performances)
                
                # Check if threshold is met
                should_retrain = avg_underperformance <= self.underperformance_threshold_pct
                
                # Log trigger if threshold met
                if should_retrain:
                    self._log_retraining_trigger(
                        portfolios_analyzed=len(portfolio_performances),
                        average_underperformance_pct=avg_underperformance,
                        threshold_met=True,
                        reason=f"Average underperformance {avg_underperformance:.2f}% exceeds threshold {self.underperformance_threshold_pct}%",
                    )
                
                return {
                    "should_retrain": should_retrain,
                    "reason": f"Average underperformance: {avg_underperformance:.2f}%",
                    "portfolios_analyzed": len(portfolio_performances),
                    "average_underperformance_pct": avg_underperformance,
                    "portfolio_details": portfolio_performances,
                    "threshold": self.underperformance_threshold_pct,
                }
        finally:
            conn.close()

    def _log_retraining_trigger(
        self,
        portfolios_analyzed: int,
        average_underperformance_pct: float,
        threshold_met: bool,
        reason: str,
    ) -> None:
        """Log a retraining trigger event."""
        conn = self._get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO model_retraining_triggers (
                        trigger_reason, portfolios_analyzed, average_underperformance_pct,
                        threshold_met, retraining_status
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        reason,
                        portfolios_analyzed,
                        average_underperformance_pct,
                        threshold_met,
                        "pending",
                    ),
                )
                conn.commit()
        except Exception as e:
            LOGGER.exception(f"Error logging retraining trigger: {e}")
        finally:
            conn.close()

    def get_retraining_history(self, limit: int = 10) -> List[Dict[str, any]]:
        """Get history of retraining triggers."""
        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT 
                        id,
                        triggered_at,
                        trigger_reason,
                        portfolios_analyzed,
                        average_underperformance_pct,
                        threshold_met,
                        retraining_status,
                        retraining_completed_at,
                        notes
                    FROM model_retraining_triggers
                    ORDER BY triggered_at DESC
                    LIMIT %s
                    """
                    ,
                    (limit,),
                )
                rows = cursor.fetchall()
                
                return [
                    {
                        "id": str(row["id"]),
                        "triggered_at": row["triggered_at"].isoformat() if row["triggered_at"] else None,
                        "trigger_reason": row["trigger_reason"],
                        "portfolios_analyzed": row["portfolios_analyzed"],
                        "average_underperformance_pct": float(row["average_underperformance_pct"]) if row.get("average_underperformance_pct") else None,
                        "threshold_met": row["threshold_met"],
                        "retraining_status": row["retraining_status"],
                        "retraining_completed_at": row["retraining_completed_at"].isoformat() if row.get("retraining_completed_at") else None,
                        "notes": row.get("notes"),
                    }
                    for row in rows
                ]
        finally:
            conn.close()


def get_model_retraining_service() -> ModelRetrainingService:
    """Get ModelRetrainingService instance."""
    return ModelRetrainingService()

