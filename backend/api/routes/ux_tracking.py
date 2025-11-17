"""
UX Tracking endpoints per audit document (4.2).
Track user journeys and measure UX benchmarks.
"""

from __future__ import annotations

import os
from typing import Optional

import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.services.ux_tracking import get_ux_tracking_service
from caria.models.auth import UserInDB

router = APIRouter(prefix="/api/ux", tags=["UX Tracking"])


class TaskTrackingRequest(BaseModel):
    """Request model for tracking a task."""
    task_name: str = Field(..., description="Task name (e.g., 'open_account', 'add_holding')")
    clicks: int = Field(..., ge=0, description="Number of clicks")
    seconds: float = Field(..., ge=0, description="Time in seconds")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB", "caria"),
    )


@router.post("/track")
async def track_task(
    request: TaskTrackingRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Track a user task completion.
    
    Per audit document (4.2): Track clicks and seconds per user journey.
    Tasks: open_account, sell_stock, add_holding, etc.
    """
    db_conn = get_db_connection()
    try:
        service = get_ux_tracking_service(db_conn)
        service.track_task(
            user_id=current_user.id,
            task_name=request.task_name,
            clicks=request.clicks,
            seconds=request.seconds,
            metadata=request.metadata,
        )
        return {"status": "tracked", "task": request.task_name}
    finally:
        db_conn.close()


@router.get("/metrics/task/{task_name}")
async def get_task_metrics(
    task_name: str,
    days: int = Query(30, ge=1, le=365),
    current_user: Optional[UserInDB] = Depends(get_current_user),
) -> dict:
    """
    Get aggregated metrics for a task.
    
    Returns average clicks, seconds, and completion statistics.
    """
    db_conn = get_db_connection()
    try:
        service = get_ux_tracking_service(db_conn)
        metrics = service.get_task_metrics(task_name, days)
        return metrics
    finally:
        db_conn.close()


@router.get("/metrics/onboarding")
async def get_onboarding_metrics(
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Get onboarding completion metrics.
    
    Per audit document (4.2): Target 4.5 minutes (Chime benchmark).
    """
    db_conn = get_db_connection()
    try:
        service = get_ux_tracking_service(db_conn)
        metrics = service.get_onboarding_metrics(current_user.id)
        return metrics
    finally:
        db_conn.close()

