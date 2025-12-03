"""
API Routes for Global Economic Monitor.

Provides endpoints for:
- Economic indicators by country
- Business cycle clock data
- Macroeconomic heatmap
- Currency exchange rates
- Country details
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime, timedelta
import logging

from api.dependencies import get_current_user, get_db_connection
from caria.models.auth import UserInDB
from api.models.economic_monitor import (
    IndicatorsResponse, BusinessCycleResponse, HeatmapResponse,
    CurrencyResponse, CurrencyHistoryResponse, CountryDetailsResponse,
    CountryEconomicData, BusinessCyclePoint, HeatmapCell, CurrencyRate, CurrencyHistory
)
from api.services.fred_client import FREDClient
from api.services.world_bank_client import WorldBankClient
from api.services.imf_client import IMFClient
from api.services.oecd_client import OECDClient
from api.services.economic_indicators_service import EconomicIndicatorsService

router = APIRouter(prefix="/api/economic-monitor", tags=["economic-monitor"])

logger = logging.getLogger(__name__)

# Initialize clients
fred_client = FREDClient()
wb_client = WorldBankClient()
imf_client = IMFClient()
oecd_client = OECDClient()
indicators_service = EconomicIndicatorsService()

# Country configuration
MAJOR_COUNTRIES = {
    "US": {"name": "United States", "region": "North America"},
    "GB": {"name": "United Kingdom", "region": "Europe"},
    "DE": {"name": "Germany", "region": "Europe"},
    "FR": {"name": "France", "region": "Europe"},
    "JP": {"name": "Japan", "region": "Asia"},
    "CN": {"name": "China", "region": "Asia"},
    "IN": {"name": "India", "region": "Asia"},
    "BR": {"name": "Brazil", "region": "Latin America"},
    "MX": {"name": "Mexico", "region": "Latin America"},
    "AR": {"name": "Argentina", "region": "Latin America"},
    "ZA": {"name": "South Africa", "region": "Africa"},
    "AU": {"name": "Australia", "region": "Asia"},
    "CA": {"name": "Canada", "region": "North America"},
    "IT": {"name": "Italy", "region": "Europe"},
    "ES": {"name": "Spain", "region": "Europe"},
}

MAJOR_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CNY", "AUD", "CAD", "CHF"]


@router.get("/countries", response_model=List[dict])
def get_countries():
    """Get list of supported countries."""
    return [
        {"code": code, "name": info["name"], "region": info["region"]}
        for code, info in MAJOR_COUNTRIES.items()
    ]


@router.get("/indicators", response_model=IndicatorsResponse)
def get_indicators(
    countries: Optional[str] = Query(None, description="Comma-separated country codes"),
    current_user: UserInDB = Depends(get_current_user),
):
    """Get economic indicators for specified countries."""
    try:
        country_codes = countries.split(",") if countries else list(MAJOR_COUNTRIES.keys())
        country_codes = [c.strip().upper() for c in country_codes]
        
        countries_data = []
        
        for code in country_codes:
            if code not in MAJOR_COUNTRIES:
                continue
            
            country_info = MAJOR_COUNTRIES[code]
            indicators = []
            
            # Fetch indicators based on country type (Advanced vs Emerging)
            # For now, return mock structure - will be populated with real data
            countries_data.append(CountryEconomicData(
                country_code=code,
                country_name=country_info["name"],
                region=country_info["region"],
                indicators=indicators,
                last_updated=datetime.now()
            ))
        
        return IndicatorsResponse(
            countries=countries_data,
            last_updated=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error fetching indicators: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/business-cycle", response_model=BusinessCycleResponse)
def get_business_cycle(
    current_user: UserInDB = Depends(get_current_user),
):
    """Get business cycle clock data for all countries."""
    try:
        points = []
        
        for code, info in MAJOR_COUNTRIES.items():
            # Get Industrial Production data
            ip_series = fred_client.get_industrial_production(code)
            
            if ip_series is not None and len(ip_series) > 12:
                point = indicators_service.calculate_business_cycle_point(
                    ip_series, code, info["name"]
                )
                if point:
                    points.append(point)
        
        return BusinessCycleResponse(
            points=points,
            last_updated=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error calculating business cycle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/heatmap", response_model=HeatmapResponse)
def get_heatmap(
    current_user: UserInDB = Depends(get_current_user),
):
    """Get macroeconomic heatmap data with Z-scores."""
    try:
        cells = []
        
        # This will be populated with real indicator data
        # For now, return structure
        
        return HeatmapResponse(
            cells=cells,
            countries=list(MAJOR_COUNTRIES.keys()),
            indicators=["GDP Growth", "Inflation", "Reserves/Debt", "Fiscal Balance"],
            last_updated=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/currencies", response_model=CurrencyResponse)
def get_currencies(
    current_user: UserInDB = Depends(get_current_user),
):
    """Get currency exchange rates for major economies."""
    try:
        rates = []
        
        # Major currency pairs
        currency_pairs = [
            ("EUR", "USD", "EURUSD"),
            ("GBP", "USD", "GBPUSD"),
            ("USD", "JPY", "USDJPY"),
            ("USD", "CNY", "USDCNY"),
            ("AUD", "USD", "AUDUSD"),
            ("USD", "CAD", "USDCAD"),
            ("USD", "CHF", "USDCHF"),
        ]
        
        for base, quote, pair in currency_pairs:
            rate = fred_client.get_currency_rate(pair)
            
            if rate is not None:
                # Get historical data for change calculations
                history = fred_client.get_currency_history(pair, days=365)
                
                change_1d = None
                change_1w = None
                change_1m = None
                change_1y = None
                
                change_pct_1d = None
                change_pct_1w = None
                change_pct_1m = None
                change_pct_1y = None
                
                if history is not None and len(history) > 0:
                    current_rate = float(history.iloc[-1])
                    
                    # Calculate changes
                    if len(history) > 1:
                        prev_rate = float(history.iloc[-2])
                        change_1d = current_rate - prev_rate
                        change_pct_1d = (change_1d / prev_rate * 100) if prev_rate != 0 else 0
                    
                    if len(history) > 7:
                        week_ago_rate = float(history.iloc[-7])
                        change_1w = current_rate - week_ago_rate
                        change_pct_1w = (change_1w / week_ago_rate * 100) if week_ago_rate != 0 else 0
                    
                    if len(history) > 30:
                        month_ago_rate = float(history.iloc[-30])
                        change_1m = current_rate - month_ago_rate
                        change_pct_1m = (change_1m / month_ago_rate * 100) if month_ago_rate != 0 else 0
                    
                    if len(history) > 365:
                        year_ago_rate = float(history.iloc[-365])
                        change_1y = current_rate - year_ago_rate
                        change_pct_1y = (change_1y / year_ago_rate * 100) if year_ago_rate != 0 else 0
                
                rates.append(CurrencyRate(
                    currency_pair=pair,
                    base_currency=base,
                    quote_currency=quote,
                    rate=float(rate),
                    date=datetime.now(),
                    change_1d=change_1d,
                    change_1w=change_1w,
                    change_1m=change_1m,
                    change_1y=change_1y,
                    change_pct_1d=change_pct_1d if change_1d is not None else None,
                    change_pct_1w=change_pct_1w if change_1w is not None else None,
                    change_pct_1m=change_pct_1m if change_1m is not None else None,
                    change_pct_1y=change_pct_1y if change_1y is not None else None,
                ))
        
        return CurrencyResponse(
            rates=rates,
            last_updated=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error fetching currencies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/currency/{currency_pair}", response_model=CurrencyHistoryResponse)
def get_currency_history(
    currency_pair: str,
    days: int = Query(365, ge=30, le=1825),
    current_user: UserInDB = Depends(get_current_user),
):
    """Get historical currency data for a currency pair."""
    try:
        history_series = fred_client.get_currency_history(currency_pair.upper(), days=days)
        
        if history_series is None or len(history_series) == 0:
            raise HTTPException(status_code=404, detail=f"Currency pair {currency_pair} not found")
        
        # Convert to lists
        dates = history_series.index.tolist()
        rates = history_series.values.tolist()
        
        # Map currency pair to country if possible
        country_code = None
        country_name = None
        
        return CurrencyHistoryResponse(
            history=CurrencyHistory(
                currency_pair=currency_pair.upper(),
                dates=dates,
                rates=rates,
                country_code=country_code,
                country_name=country_name
            ),
            last_updated=datetime.now()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching currency history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/country/{country_code}", response_model=CountryDetailsResponse)
def get_country_details(
    country_code: str,
    current_user: UserInDB = Depends(get_current_user),
):
    """Get detailed economic data for a specific country."""
    try:
        code = country_code.upper()
        
        if code not in MAJOR_COUNTRIES:
            raise HTTPException(status_code=404, detail=f"Country {country_code} not supported")
        
        country_info = MAJOR_COUNTRIES[code]
        
        # Fetch indicators
        indicators = []
        
        # Get business cycle point
        ip_series = fred_client.get_industrial_production(code)
        business_cycle_point = None
        if ip_series is not None and len(ip_series) > 12:
            business_cycle_point = indicators_service.calculate_business_cycle_point(
                ip_series, code, country_info["name"]
            )
        
        # Get currency data (if applicable)
        currency = None
        currency_history = None
        
        return CountryDetailsResponse(
            country_code=code,
            country_name=country_info["name"],
            region=country_info["region"],
            indicators=indicators,
            currency=currency,
            currency_history=currency_history,
            business_cycle_point=business_cycle_point,
            last_updated=datetime.now()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching country details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

