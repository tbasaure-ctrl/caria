"""
Portfolio Analytics endpoints using quantstats per audit document (3.2).
"""

from __future__ import annotations

import os
from datetime import datetime

import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.dependencies import get_current_user
from api.services.portfolio_analytics import get_portfolio_analytics_service
from caria.models.auth import UserInDB

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio Analytics"])


def get_db_connection() -> psycopg2.extensions.connection:
    """Get database connection."""
    from urllib.parse import urlparse, parse_qs
    
    # Try DATABASE_URL first (Cloud SQL format)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            parsed = urlparse(database_url)
            query_params = parse_qs(parsed.query)
            
            # Check for Unix socket (Cloud SQL)
            unix_socket_host = query_params.get('host', [None])[0]
            
            if unix_socket_host:
                # Use Cloud SQL Unix socket
                return psycopg2.connect(
                    host=unix_socket_host,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
            elif parsed.hostname:
                # Use normal TCP connection
                return psycopg2.connect(
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
        except Exception as e:
            logger.warning(f"Error using DATABASE_URL: {e}. Falling back to individual vars...")
    
    # Fallback to individual environment variables
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD environment variable is required")
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=password,
        database=os.getenv("POSTGRES_DB", "caria"),
    )


@router.get("/analysis")
async def get_portfolio_analysis(
    benchmark: str = Query("SPY", description="Benchmark ticker (e.g., SPY, QQQ)"),
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Get professional portfolio analysis using quantstats.

    Per audit document (3.2): Returns metrics (Sharpe, Sortino, Alpha, Beta,
    Max Drawdown, CAGR) and generates HTML tearsheet.
    """
    db_conn = get_db_connection()
    try:
        analytics_service = get_portfolio_analytics_service()
        result = analytics_service.analyze_portfolio(
            user_id=current_user.id,
            db_connection=db_conn,
            benchmark=benchmark.upper(),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing portfolio: {str(e)}"
        ) from e
    finally:
        db_conn.close()


@router.get("/analysis/report")
async def get_portfolio_report(
    benchmark: str = Query("SPY", description="Benchmark ticker"),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get HTML tearsheet report file.
    """
    from pathlib import Path

    db_conn = get_db_connection()
    try:
        analytics_service = get_portfolio_analytics_service()
        result = analytics_service.analyze_portfolio(
            user_id=current_user.id,
            db_connection=db_conn,
            benchmark=benchmark.upper(),
        )

        html_path = Path(result["html_report_path"])
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")

        return FileResponse(
            html_path,
            media_type="text/html",
            filename=f"portfolio_report_{current_user.id}.html",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating report: {str(e)}"
        ) from e
    finally:
        db_conn.close()


@router.get("/analysis/metrics")
async def get_portfolio_metrics(
    benchmark: str = Query("SPY", description="Benchmark ticker"),
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Get portfolio metrics only (without HTML generation).
    Faster endpoint for quick metrics retrieval.
    """
    import logging
    import sys

    LOGGER = logging.getLogger("caria.api.portfolio_analytics")

    # Force logging to stdout to ensure we see it
    LOGGER.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in LOGGER.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        LOGGER.addHandler(handler)

    LOGGER.info(
        f"=== get_portfolio_metrics CALLED === user={current_user.id}, "
        f"benchmark={benchmark}"
    )

    try:
        LOGGER.info("Getting database connection...")
        db_conn = get_db_connection()
        LOGGER.info("Database connection established")

        LOGGER.info("Getting analytics service...")
        analytics_service = get_portfolio_analytics_service()
        LOGGER.info("Analytics service obtained")

        LOGGER.info(f"Calling analyze_portfolio for user {current_user.id}...")
        result = analytics_service.analyze_portfolio(
            user_id=current_user.id,
            db_connection=db_conn,
            benchmark=benchmark.upper(),
        )
        LOGGER.info("analyze_portfolio completed successfully")

        return {
            "metrics": result["metrics"],
            "holdings_count": result["holdings_count"],
            "analysis_date": result["analysis_date"],
        }
    except ValueError as e:
        LOGGER.warning(f"ValueError in get_portfolio_metrics: {e}")
        # Para casos sin holdings o sin datos, devolvemos métricas vacías en vez de 500
        error_msg = str(e)
        if (
            "No holdings" in error_msg
            or "price data" in error_msg
            or "Could not calculate" in error_msg
        ):
            return {
                "metrics": {},
                "holdings_count": 0,
                "analysis_date": datetime.utcnow().isoformat(),
                "error": error_msg,
            }
        raise HTTPException(status_code=400, detail=error_msg) from e
    except Exception as e:
        LOGGER.exception(f"Exception in get_portfolio_metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error analyzing portfolio: {str(e)}"
        ) from e
    finally:
        if "db_conn" in locals():
            db_conn.close()
            LOGGER.info("Database connection closed")
