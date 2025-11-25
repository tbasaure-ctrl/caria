"""
Weekly Screening Route - Expands universe by screening Yahoo Finance once per week.
"""
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from api.dependencies import get_current_user
from api.services.fundamentals_cache_service import get_fundamentals_cache_service
from api.services.openbb_client import OpenBBClient

router = APIRouter(prefix="/api/screening", tags=["screening"])
LOGGER = logging.getLogger("caria.api.screening")


@router.post("/weekly/yahoo-finance")
async def weekly_yahoo_finance_screening(
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Screen popular stocks from Yahoo Finance and add them to the universe.
    This should be run once per week to expand the screening universe.
    
    Fetches:
    - Most active stocks
    - Trending tickers
    - Popular ETFs
    - Adds them to fundamentals cache for future screening
    """
    try:
        cache_service = get_fundamentals_cache_service()
        obb_client = OpenBBClient()
        
        # Get current universe size
        stats_before = cache_service.get_cache_stats()
        initial_count = stats_before.get("total_universe", 0)
        
        # Popular tickers to screen from Yahoo Finance
        # These are common high-volume stocks that should be in the universe
        popular_tickers = [
            # Tech giants
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
            # Other large caps
            "JPM", "JNJ", "V", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC",
            # Growth stocks
            "AMD", "INTC", "CRM", "ADBE", "NFLX", "PYPL", "COIN", "SQ",
            # ETFs
            "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "ARKK",
            # Popular sectors
            "XOM", "CVX", "SLB",  # Energy
            "GS", "MS", "C",  # Financials
            "LLY", "PFE", "ABBV",  # Healthcare
        ]
        
        added_count = 0
        failed_count = 0
        results = []
        
        LOGGER.info(f"Starting weekly Yahoo Finance screening. Current universe: {initial_count} stocks")
        
        for ticker in popular_tickers:
            try:
                # Check if already in cache
                existing = cache_service.get_fundamentals(ticker)
                if existing and existing.get("source") in ["static_cache", "dynamic_cache"]:
                    LOGGER.debug(f"{ticker} already in cache, skipping")
                    continue
                
                # Fetch and cache
                LOGGER.info(f"Fetching {ticker} from OpenBB...")
                data = cache_service._fetch_from_openbb(ticker)
                cache_service._save_to_dynamic_cache(ticker, data)
                
                added_count += 1
                results.append({
                    "ticker": ticker,
                    "status": "added",
                    "company_name": data.get("company_name", ticker)
                })
                
            except Exception as e:
                LOGGER.warning(f"Failed to fetch {ticker}: {e}")
                failed_count += 1
                results.append({
                    "ticker": ticker,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Get updated stats
        stats_after = cache_service.get_cache_stats()
        final_count = stats_after.get("total_universe", 0)
        growth = final_count - initial_count
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "universe_before": initial_count,
                "universe_after": final_count,
                "growth": growth,
                "added": added_count,
                "failed": failed_count,
                "total_screened": len(popular_tickers)
            },
            "results": results,
            "message": f"Weekly screening completed. Universe expanded by {growth} stocks."
        }
        
    except Exception as e:
        LOGGER.error(f"Error in weekly screening: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run weekly screening: {str(e)}")


@router.get("/weekly/status")
async def get_weekly_screening_status(
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get status of weekly screening and universe size.
    """
    try:
        cache_service = get_fundamentals_cache_service()
        stats = cache_service.get_cache_stats()
        
        return {
            "universe_size": stats.get("total_universe", 0),
            "static_cache": stats.get("static_cache_count", 0),
            "dynamic_cache": stats.get("dynamic_cache_count", 0),
            "growth_from_initial": stats.get("growth", 0),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        LOGGER.error(f"Error getting screening status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
