"""
Model Validation endpoints per audit document (2.1).
Backtesting, statistical metrics (P-value, R²), and benchmarking.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.services.model_validation import get_model_validation_service
from caria.models.auth import UserInDB

LOGGER = logging.getLogger("caria.api.model_validation_routes")

router = APIRouter(prefix="/api/model/validation", tags=["Model Validation"])


class ValidationRequest(BaseModel):
    """Request model for model validation."""
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class StatisticalMetricsRequest(BaseModel):
    """Request model for statistical metrics."""
    predicted_values: list[float] = Field(..., description="Model predictions")
    actual_values: list[float] = Field(..., description="Actual observed values")


def get_db_connection():
    """Get database connection."""
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise HTTPException(
            status_code=500,
            detail="POSTGRES_PASSWORD not configured"
        )
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),  # Use 'postgres' as default for Docker
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=password,
        database=os.getenv("POSTGRES_DB", "caria"),
    )


@router.post("/backtest")
async def run_backtest(
    request: ValidationRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Run backtesting to compare model predictions with actual data.
    
    Per audit document (2.1): Requires minimum 30-100 data points.
    Compares regime predictions with actual economic indicators (NBER, VIX, SPY).
    """
    try:
        # Validate date range
        try:
            start = datetime.strptime(request.start_date, "%Y-%m-%d")
            end = datetime.strptime(request.end_date, "%Y-%m-%d")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {str(ve)}") from ve
        
        # Ensure days_diff is an integer, not a Series
        days_diff = int((end - start).days)
        if days_diff < 30:
            raise HTTPException(
                status_code=400,
                detail="Date range must be at least 30 days (minimum 30 data points required)"
            )
        
        if days_diff > 365 * 5:
            raise HTTPException(
                status_code=400,
                detail="Date range cannot exceed 5 years"
            )

        db_conn = get_db_connection()
        service = get_model_validation_service()
        
        try:
            result = service.run_full_validation(
                request.start_date,
                request.end_date,
                db_conn,
            )
            return result
        finally:
            db_conn.close()
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{type(e).__name__}: {str(e)}"
        LOGGER.exception("Backtest error: %s", error_detail)
        # Don't wrap ValueError as "Invalid date format" if it's not about dates
        if isinstance(e, ValueError) and "date" not in str(e).lower():
            raise HTTPException(status_code=500, detail=f"Validation error: {error_detail}") from e
        raise HTTPException(status_code=500, detail=f"Backtest error: {error_detail}") from e


@router.post("/statistics")
async def calculate_statistical_metrics(
    request: StatisticalMetricsRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Calculate statistical metrics: P-value and R².
    
    Per audit document (2.1): Properly communicates interpretation:
    - "Significant relationship but noisy predictions" if R² low but P-value < 0.05
    """
    try:
        import numpy as np
        
        pred_array = np.array(request.predicted_values)
        actual_array = np.array(request.actual_values)

        if len(pred_array) != len(actual_array):
            raise HTTPException(
                status_code=400,
                detail="predicted_values and actual_values must have the same length"
            )

        service = get_model_validation_service()
        result = service.calculate_statistical_metrics(pred_array, actual_array)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}") from e


@router.get("/benchmark")
async def benchmark_model(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Benchmark model performance vs simple strategies.
    
    Per audit document (2.1): Compares vs buy-and-hold and moving average crossover.
    """
    try:
        # Default to last year if dates not provided
        if not end_date:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

        # This endpoint would need model returns data
        # For now, return structure
        return {
            "message": "Benchmarking requires model returns data. Implementation in progress.",
            "start_date": start_date,
            "end_date": end_date,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}") from e


@router.post("/generate-sample-data")
async def generate_sample_data(
    n_days: int = Query(100, description="Number of days to generate"),
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Generate sample historical data for model validation.
    Creates realistic predictions vs actuals for testing.
    """
    try:
        from api.scripts.generate_validation_data import (
            create_predictions_table,
            generate_realistic_data,
            load_data_to_db,
        )
        
        db_conn = get_db_connection()
        
        try:
            # Generate data for last 6 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            df = generate_realistic_data(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                n_days=n_days
            )
            
            create_predictions_table(db_conn)
            load_data_to_db(db_conn, df)
            
            accuracy = (df['predicted_regime'] == df['actual_regime']).mean()
            
            return {
                "message": "Sample data generated successfully",
                "n_records": len(df),
                "date_range": {
                    "start": df['date'].min().isoformat(),
                    "end": df['date'].max().isoformat(),
                },
                "regime_distribution": df["actual_regime"].value_counts().to_dict(),
                "prediction_accuracy": float(accuracy),
            }
        finally:
            db_conn.close()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating sample data: {str(e)}") from e

