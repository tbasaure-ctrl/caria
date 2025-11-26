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
    Get simple valuation using direct FMP calls for P/E, EV/EBITDA, EV/Sales.
    Simplified approach that's more reliable.
    """
    try:
        ticker = ticker.upper()
        current_price = request.current_price

        # If no price provided, fetch it
        if not current_price:
            LOGGER.info(f"Fetching current price for {ticker}")
            try:
                from ..services.openbb_client import OpenBBClient
                obb_client = OpenBBClient()
                current_price = obb_client.get_current_price(ticker)
                if not current_price or current_price <= 0:
                    raise ValueError(f"Invalid price: {current_price}")
                LOGGER.info(f"Got price for {ticker}: ${current_price}")
            except Exception as price_error:
                LOGGER.error(f"Failed to fetch price for {ticker}: {price_error}")
                raise HTTPException(status_code=400, detail=f"Could not fetch current price for {ticker}")

        if not current_price or current_price <= 0:
            raise HTTPException(status_code=400, detail=f"Invalid price for {ticker}: {current_price}")

        # Use simplified direct FMP approach
        from ..services.openbb_client import OpenBBClient
        import os
        from openbb import obb
        
        obb_client = OpenBBClient()
        fmp_key = os.getenv("FMP_API_KEY", "").strip()
        if fmp_key:
            obb.user.credentials.fmp_api_key = fmp_key

        # Fetch ratios directly from FMP
        try:
            ratios = obb.equity.fundamental.ratios(symbol=ticker, provider="fmp", limit=1)
            metrics = obb.equity.fundamental.metrics(symbol=ticker, provider="fmp", limit=1)
            
            pe_ratio = None
            ev_ebitda = None
            ev_sales = None
            
            if ratios and hasattr(ratios, 'to_df'):
                df_ratios = ratios.to_df()
                if not df_ratios.empty:
                    row = df_ratios.iloc[0]
                    pe_ratio = row.get('peRatio') or row.get('priceEarningsRatio') or row.get('pe_ratio')
                    ev_ebitda = row.get('enterpriseValueMultiple') or row.get('evEbitda') or row.get('ev_ebitda')
                    ev_sales = row.get('priceToSalesRatio') or row.get('evSales') or row.get('ev_sales')
            
            if metrics and hasattr(metrics, 'to_df'):
                df_metrics = metrics.to_df()
                if not df_metrics.empty:
                    row = df_metrics.iloc[0]
                    if not pe_ratio:
                        pe_ratio = row.get('peRatio') or row.get('priceEarningsRatio')
                    if not ev_ebitda:
                        ev_ebitda = row.get('enterpriseValueMultiple') or row.get('evEbitda')
                    if not ev_sales:
                        ev_sales = row.get('evSales') or row.get('enterpriseValueOverRevenue')
            
            # Calculate fair value using industry averages
            # For simplicity, use reasonable defaults if multiples are missing
            fair_value = current_price
            method = "Current Price"
            
            if pe_ratio and pe_ratio > 0:
                # Use industry average P/E of 20
                eps = current_price / pe_ratio if pe_ratio > 0 else None
                if eps and eps > 0:
                    fair_value = eps * 20
                    method = "P/E Multiple (20x)"
            elif ev_ebitda and ev_ebitda > 0:
                # Use industry average EV/EBITDA of 12
                # Approximate: fair_value ≈ current_price * (12 / ev_ebitda)
                fair_value = current_price * (12 / ev_ebitda) if ev_ebitda > 0 else current_price
                method = "EV/EBITDA Multiple (12x)"
            elif ev_sales and ev_sales > 0:
                # Use industry average EV/Sales of 3
                fair_value = current_price * (3 / ev_sales) if ev_sales > 0 else current_price
                method = "EV/Sales Multiple (3x)"
            
            upside = ((fair_value - current_price) / current_price) * 100 if current_price > 0 else 0
            
            result = {
                "ticker": ticker,
                "currency": "USD",
                "current_price": current_price,
                "multiples": {
                    "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
                    "ev_ebitda": round(ev_ebitda, 2) if ev_ebitda else None,
                    "ev_sales": round(ev_sales, 2) if ev_sales else None,
                },
                "dcf": {
                    "method": method,
                    "fair_value_per_share": round(fair_value, 2),
                    "upside_percent": round(upside, 2),
                    "explanation": f"Valuation based on {method}. P/E: {pe_ratio:.2f if pe_ratio and pe_ratio > 0 else 'N/A'}, EV/EBITDA: {ev_ebitda:.2f if ev_ebitda and ev_ebitda > 0 else 'N/A'}, EV/Sales: {ev_sales:.2f if ev_sales and ev_sales > 0 else 'N/A'}"
                },
                "reverse_dcf": {
                    "method": "N/A",
                    "fair_value_per_share": round(fair_value, 2),
                    "upside_percent": round(upside, 2),
                    "explanation": "Simplified valuation using multiples"
                },
                "multiples_valuation": {
                    "method": method,
                    "fair_value": round(fair_value, 2),
                    "upside_percent": round(upside, 2),
                    "explanation": f"Fair value calculated using {method}"
                }
            }
            
            LOGGER.info(f"✅ Simple valuation completed for {ticker}: ${current_price} → ${fair_value:.2f} ({upside:+.1f}%)")
            return result
            
        except Exception as e:
            LOGGER.warning(f"Direct FMP valuation failed for {ticker}: {e}, falling back to SimpleValuationService")
            # Fallback to original service
            service = SimpleValuationService()
            result = service.get_valuation(ticker, current_price)
            if not result:
                raise HTTPException(status_code=500, detail=f"Valuation service returned no data for {ticker}")
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

