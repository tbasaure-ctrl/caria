"""
Model Portfolio endpoints.
Select and track model-selected portfolios for validation.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from urllib.parse import urlparse, parse_qs
import json

from api.dependencies import get_current_user
from api.services.portfolio_selection_service import get_portfolio_selection_service
from api.services.model_retraining_service import get_model_retraining_service
from caria.models.auth import UserInDB

LOGGER = logging.getLogger("caria.api.model_portfolio")

router = APIRouter(prefix="/api/portfolio/model", tags=["Model Portfolio"])


def _get_db_connection():
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
            raise HTTPException(status_code=500, detail="Database password not configured")
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "caria_user"),
            password=password,
            database=os.getenv("POSTGRES_DB", "caria"),
        )
    
    return conn


class PortfolioSelectRequest(BaseModel):
    """Request model for portfolio selection."""
    selection_type: str = Field(..., pattern="^(outlier|balanced|random)$")
    num_holdings: int = Field(15, ge=10, le=20)
    regime: Optional[str] = Field(None, description="Optional regime for selection")


class PortfolioSelectResponse(BaseModel):
    """Response model for portfolio selection."""
    portfolio_id: str
    selection_type: str
    holdings: List[dict] = Field(..., description="List of {ticker, allocation} objects")
    total_holdings: int
    regime: Optional[str] = None
    created_at: datetime


class PortfolioPerformanceResponse(BaseModel):
    """Response model for portfolio performance."""
    portfolio_id: str
    date: date
    portfolio_value: float
    portfolio_return_pct: float
    benchmark_sp500_return_pct: Optional[float] = None
    benchmark_qqq_return_pct: Optional[float] = None
    benchmark_vti_return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    volatility_pct: Optional[float] = None
    alpha_pct: Optional[float] = None
    beta: Optional[float] = None


@router.post("/select", response_model=PortfolioSelectResponse, status_code=201)
async def select_model_portfolio(
    request: PortfolioSelectRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> PortfolioSelectResponse:
    """
    Select a portfolio using the model for validation.
    
    Selection types:
    - outlier: Portfolios with unusual allocations (for testing edge cases)
    - balanced: Well-diversified portfolios (for testing normal cases)
    - random: Random selection (for baseline comparison)
    """
    try:
        selection_service = get_portfolio_selection_service()
        portfolio_data = selection_service.select_portfolio(
            selection_type=request.selection_type,
            num_holdings=request.num_holdings,
            regime=request.regime,
        )
        
        # Save to database
        conn = _get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    INSERT INTO model_portfolios (
                        selection_type, regime, holdings, total_holdings, initial_value, status
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (
                        portfolio_data["selection_type"],
                        portfolio_data.get("regime"),
                        json.dumps(portfolio_data["holdings"]),
                        portfolio_data["total_holdings"],
                        10000.00,  # Default initial value
                        "active",
                    ),
                )
                row = cursor.fetchone()
                conn.commit()
                
                return PortfolioSelectResponse(
                    portfolio_id=str(row["id"]),
                    selection_type=portfolio_data["selection_type"],
                    holdings=portfolio_data["holdings"],
                    total_holdings=portfolio_data["total_holdings"],
                    regime=portfolio_data.get("regime"),
                    created_at=row["created_at"],
                )
        finally:
            conn.close()
            
    except Exception as e:
        LOGGER.exception(f"Error selecting portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Error selecting portfolio: {str(e)}") from e


@router.get("/track", response_model=List[PortfolioPerformanceResponse])
async def track_portfolio_performance(
    portfolio_id: Optional[str] = Query(None, description="Portfolio ID to track"),
    start_date: Optional[date] = Query(None, description="Start date for performance data"),
    end_date: Optional[date] = Query(None, description="End date for performance data"),
    current_user: UserInDB = Depends(get_current_user),
) -> List[PortfolioPerformanceResponse]:
    """
    Get performance tracking data for model portfolios.
    
    If portfolio_id is provided, returns performance for that portfolio.
    Otherwise, returns performance for all active portfolios.
    """
    conn = _get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT 
                    pp.portfolio_id,
                    pp.date,
                    pp.portfolio_value,
                    pp.portfolio_return_pct,
                    pp.benchmark_sp500_return_pct,
                    pp.benchmark_qqq_return_pct,
                    pp.benchmark_vti_return_pct,
                    pp.sharpe_ratio,
                    pp.max_drawdown_pct,
                    pp.volatility_pct,
                    pp.alpha_pct,
                    pp.beta
                FROM portfolio_performance pp
                WHERE 1=1
            """
            params = []
            
            if portfolio_id:
                query += " AND pp.portfolio_id = %s"
                params.append(portfolio_id)
            else:
                # Only active portfolios if no specific ID
                query += """
                    AND pp.portfolio_id IN (
                        SELECT id FROM model_portfolios WHERE status = 'active'
                    )
                """
            
            if start_date:
                query += " AND pp.date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND pp.date <= %s"
                params.append(end_date)
            
            query += " ORDER BY pp.portfolio_id, pp.date DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                PortfolioPerformanceResponse(
                    portfolio_id=str(row["portfolio_id"]),
                    date=row["date"],
                    portfolio_value=float(row["portfolio_value"]),
                    portfolio_return_pct=float(row["portfolio_return_pct"]),
                    benchmark_sp500_return_pct=float(row["benchmark_sp500_return_pct"]) if row.get("benchmark_sp500_return_pct") else None,
                    benchmark_qqq_return_pct=float(row["benchmark_qqq_return_pct"]) if row.get("benchmark_qqq_return_pct") else None,
                    benchmark_vti_return_pct=float(row["benchmark_vti_return_pct"]) if row.get("benchmark_vti_return_pct") else None,
                    sharpe_ratio=float(row["sharpe_ratio"]) if row.get("sharpe_ratio") else None,
                    max_drawdown_pct=float(row["max_drawdown_pct"]) if row.get("max_drawdown_pct") else None,
                    volatility_pct=float(row["volatility_pct"]) if row.get("volatility_pct") else None,
                    alpha_pct=float(row["alpha_pct"]) if row.get("alpha_pct") else None,
                    beta=float(row["beta"]) if row.get("beta") else None,
                )
                for row in rows
            ]
    finally:
        conn.close()


@router.get("/list")
async def list_model_portfolios(
    status: Optional[str] = Query(None, pattern="^(active|completed|archived)$"),
    selection_type: Optional[str] = Query(None, pattern="^(outlier|balanced|random)$"),
    current_user: UserInDB = Depends(get_current_user),
) -> List[dict]:
    """List model portfolios with optional filtering."""
    conn = _get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT 
                    id,
                    created_at,
                    selection_type,
                    regime,
                    holdings,
                    total_holdings,
                    initial_value,
                    status,
                    notes
                FROM model_portfolios
                WHERE 1=1
            """
            params = []
            
            if status:
                query += " AND status = %s"
                params.append(status)
            
            if selection_type:
                query += " AND selection_type = %s"
                params.append(selection_type)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                {
                    "id": str(row["id"]),
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "selection_type": row["selection_type"],
                    "regime": row.get("regime"),
                    "holdings": row["holdings"],
                    "total_holdings": row["total_holdings"],
                    "initial_value": float(row["initial_value"]),
                    "status": row["status"],
                    "notes": row.get("notes"),
                }
                for row in rows
            ]
    finally:
        conn.close()


@router.get("/analyze")
async def analyze_portfolio_performance(
    days_back: int = Query(90, ge=7, le=365, description="Number of days to analyze"),
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Analyze portfolio performance and check if retraining should be triggered.
    Admin-only endpoint (can be restricted later).
    """
    try:
        retraining_service = get_model_retraining_service()
        analysis = retraining_service.analyze_performance(days_back=days_back)
        
        return {
            "analysis": analysis,
            "retraining_history": retraining_service.get_retraining_history(limit=10),
        }
    except Exception as e:
        LOGGER.exception(f"Error analyzing performance: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing performance: {str(e)}") from e

