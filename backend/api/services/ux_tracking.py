"""
UX Tracking Service per audit document (4.2).
Tracks user journeys: clicks and seconds per task (open account, sell stock, etc.).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional
from uuid import UUID

import psycopg2
from psycopg2.extras import RealDictCursor

LOGGER = logging.getLogger("caria.api.ux_tracking")


class UXTrackingService:
    """Service for tracking user experience metrics."""

    def __init__(self, db_connection):
        self.db = db_connection
        self._ensure_table()

    def _ensure_table(self):
        """Ensure ux_events table exists."""
        with self.db.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ux_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    event_type VARCHAR(50) NOT NULL,
                    task_name VARCHAR(100) NOT NULL,
                    clicks INTEGER DEFAULT 0,
                    seconds FLOAT DEFAULT 0.0,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_ux_events_user_id ON ux_events(user_id);
                CREATE INDEX IF NOT EXISTS idx_ux_events_task_name ON ux_events(task_name);
                CREATE INDEX IF NOT EXISTS idx_ux_events_created_at ON ux_events(created_at DESC);
                """
            )
            self.db.commit()

    def track_task(
        self,
        user_id: UUID,
        task_name: str,
        clicks: int,
        seconds: float,
        metadata: Optional[Dict] = None,
    ):
        """
        Track a user task completion.
        
        Per audit document (4.2): Track clicks and seconds per "user journey".
        Tasks: open_account, sell_stock, add_holding, etc.
        """
        with self.db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ux_events (user_id, event_type, task_name, clicks, seconds, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    str(user_id),
                    "task_completion",
                    task_name,
                    clicks,
                    seconds,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            self.db.commit()

    def get_task_metrics(self, task_name: str, days: int = 30) -> Dict:
        """
        Get aggregated metrics for a task.
        
        Returns average clicks, seconds, and completion rate.
        """
        with self.db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_completions,
                    AVG(clicks) as avg_clicks,
                    AVG(seconds) as avg_seconds,
                    MIN(seconds) as min_seconds,
                    MAX(seconds) as max_seconds,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY seconds) as median_seconds
                FROM ux_events
                WHERE task_name = %s 
                AND created_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                """,
                (task_name, days),
            )
            result = cursor.fetchone()

            if result and result["total_completions"]:
                return {
                    "task_name": task_name,
                    "total_completions": int(result["total_completions"]),
                    "avg_clicks": float(result["avg_clicks"]),
                    "avg_seconds": float(result["avg_seconds"]),
                    "min_seconds": float(result["min_seconds"]),
                    "max_seconds": float(result["max_seconds"]),
                    "median_seconds": float(result["median_seconds"]),
                }
            return {
                "task_name": task_name,
                "total_completions": 0,
                "avg_clicks": 0,
                "avg_seconds": 0,
            }

    def get_onboarding_metrics(self, user_id: UUID) -> Dict:
        """
        Get onboarding completion metrics.
        
        Per audit document (4.2): Target 4.5 minutes (Chime benchmark).
        """
        with self.db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT 
                    task_name,
                    clicks,
                    seconds,
                    created_at
                FROM ux_events
                WHERE user_id = %s 
                AND task_name LIKE 'onboarding_%'
                ORDER BY created_at ASC
                """,
                (str(user_id),),
            )
            events = [dict(row) for row in cursor.fetchall()]

            total_seconds = sum(e["seconds"] for e in events)
            total_clicks = sum(e["clicks"] for e in events)

            return {
                "user_id": str(user_id),
                "total_seconds": total_seconds,
                "total_minutes": total_seconds / 60.0,
                "total_clicks": total_clicks,
                "target_seconds": 270,  # 4.5 minutes
                "target_met": total_seconds <= 270,
                "events": events,
            }


def get_ux_tracking_service(db_connection) -> UXTrackingService:
    """Get UX tracking service instance."""
    return UXTrackingService(db_connection)

