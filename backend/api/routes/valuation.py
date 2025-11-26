from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging

from ..services.simple_valuation import SimpleValuationService
from ..services.comprehensive_valuation_service import get_comprehensive_valuation_service
from ..services.fundamentals_cache_service import get_fundamentals_cache_service
from ..dependencies import get_current_user

router = APIRouter(prefix="/api/valuation", tags=["valuation"])
LOGGER = logging.getLogger("caria.api.valuation")

class ValuationRequest(BaseModel):
    current_price: float | None = None  # Optional, will fetch if not provided

@router.post("/{ticker}")
async def get_valuation(ticker: str, request: ValuationRequest):
    """
    Get simple valuation using the robust SimpleValuationService.
    Legacy endpoint - use /comprehensive for full analysis.
    """
    try:
        ticker = ticker.upper()
        current_price = request.current_price

        # If no price provided, fetch it
        if not current_price:
            LOGGER.info(f"Fetching current price for {ticker}")
            try:
                from ..services.scoring_service import ScoringService
                scoring = ScoringService()
                price_data = scoring.fmp.get_realtime_price(ticker)
                if price_data and len(price_data) > 0:
                    current_price = price_data[0].get('price', 0)
                    LOGGER.info(f"Got price for {ticker}: ${current_price}")
            except Exception as price_error:
                LOGGER.error(f"Failed to fetch price for {ticker}: {price_error}")
                raise HTTPException(status_code=400, detail=f"Could not fetch current price for {ticker}")

        if not current_price or current_price <= 0:
            raise HTTPException(status_code=400, detail=f"Invalid price for {ticker}: {current_price}")

        service = SimpleValuationService()
        result = service.get_valuation(ticker, current_price)

        if not result:
            raise HTTPException(status_code=500, detail=f"Valuation service returned no data for {ticker}")

        LOGGER.info(f"Valuation completed for {ticker} at ${current_price}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Valuation endpoint error for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/comprehensive/{ticker}")
async def get_comprehensive_valuation(
    ticker: str,
    current_user=Depends(get_current_user)
):
    """
    Get comprehensive valuation combining three methodologies:
    - Reverse DCF (implied growth rate)
    - Multiples Valuation (historical PE, PB, PS averages)
    - Monte Carlo Simulation (2-year probabilistic forecast)
    
    If ticker not in cache, fetches from OpenBB and caches for future use.
    This grows the Alpha Picker universe organically.
    
    Returns:
        Comprehensive valuation with all three methods and executive summary
    """
    try:
        service = get_comprehensive_valuation_service()
        result = await service.get_full_valuation(ticker.upper())
        
        LOGGER.info(
            f"Comprehensive valuation completed for {ticker} "
            f"(source: {result.get('data_source')})"
        )
        
        return result
    except ValueError as e:
        LOGGER.warning(f"Valuation request error for {ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        LOGGER.error(f"Comprehensive valuation error for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate valuation: {str(e)}"
        )


@router.get("/cache/stats")
async def get_cache_stats(current_user=Depends(get_current_user)):
    """
    Get statistics about the fundamentals cache (universe growth).
    
    Shows:
    - Static cache size (initial 128 stocks from parquet)
    - Dynamic cache size (stocks fetched on-demand)
    - Total universe available for screening
    """
    try:
        cache = get_fundamentals_cache_service()
        stats = cache.get_cache_stats()
        
        return {
            "static_cache_stocks": stats.get("static_cache_count", 0),
            "dynamic_cache_stocks": stats.get("dynamic_cache_count", 0),
            "total_universe": stats.get("total_universe", 0),
            "growth_from_initial": stats.get("growth", 0),
            "message": (
                f"Started with {stats.get('static_cache_count', 0)} stocks, "
                f"now have {stats.get('total_universe', 0)} in screening universe"
            )
        }
    except Exception as e:
        LOGGER.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/tickers")
async def get_all_cached_tickers(current_user=Depends(get_current_user)):
    """
    Get list of all tickers in the cache (available for screening).
    Useful for populating dropdowns or showing universe coverage.
    """
    try:
        cache = get_fundamentals_cache_service()
        tickers = cache.get_all_cached_tickers()
        
        return {
            "count": len(tickers),
            "tickers": tickers
        }
    except Exception as e:
        LOGGER.error(f"Error getting cached tickers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

