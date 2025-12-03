"""
FRED (Federal Reserve Economic Data) API Client.

Provides access to FRED economic data including:
- Yield Curve (T10Y3M)
- Sahm Rule (SAHMREALTIME)
- Industrial Production
- Currency Exchange Rates
"""

import os
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import pandas as pd

try:
    from fredapi import Fred
except ImportError:
    Fred = None

LOGGER = logging.getLogger("caria.services.fred_client")


class FREDClient:
    """Client for FRED API."""

    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY", "").strip()
        self.client = None
        
        if not self.api_key:
            LOGGER.warning("FRED_API_KEY not found. FRED features unavailable.")
        elif Fred is None:
            LOGGER.warning("fredapi package not installed. Install with: pip install fredapi")
        else:
            try:
                self.client = Fred(api_key=self.api_key)
                LOGGER.info("FRED API client initialized")
            except Exception as e:
                LOGGER.error(f"Error initializing FRED client: {e}")
                self.client = None

    def is_available(self) -> bool:
        """Check if FRED client is available."""
        return self.client is not None

    def get_series(self, series_id: str, start_date: Optional[datetime] = None, 
                   end_date: Optional[datetime] = None) -> Optional[pd.Series]:
        """Get a time series from FRED."""
        if not self.is_available():
            return None

        try:
            if start_date and end_date:
                data = self.client.get_series(series_id, start=start_date, end=end_date)
            else:
                data = self.client.get_series(series_id)
            
            if data is not None and len(data) > 0:
                return data
            return None
        except Exception as e:
            LOGGER.error(f"Error fetching FRED series {series_id}: {e}")
            return None

    def get_yield_curve_spread(self) -> Optional[float]:
        """Get 10Y-3M yield curve spread (T10Y3M)."""
        data = self.get_series("T10Y3M")
        if data is not None and len(data) > 0:
            return float(data.iloc[-1])
        return None

    def get_sahm_rule(self) -> Optional[float]:
        """Get Sahm Rule indicator (SAHMREALTIME)."""
        data = self.get_series("SAHMREALTIME")
        if data is not None and len(data) > 0:
            return float(data.iloc[-1])
        return None

    def get_industrial_production(self, country_code: str = "US") -> Optional[pd.Series]:
        """Get Industrial Production index."""
        # US: INDPRO, Eurozone: PRINTO01EZQ661N, etc.
        series_map = {
            "US": "INDPRO",
            "EU": "PRINTO01EZQ661N",
            "GB": "PRINTO01GBQ661N",
            "JP": "PRINTO01JPQ661N",
        }
        
        series_id = series_map.get(country_code, "INDPRO")
        return self.get_series(series_id)

    def get_currency_rate(self, currency_pair: str) -> Optional[float]:
        """Get currency exchange rate from FRED."""
        # FRED currency series IDs
        series_map = {
            "EURUSD": "DEXUSEU",
            "GBPUSD": "DEXUSUK",
            "USDJPY": "DEXJPUS",
            "USDCNY": "DEXCHUS",
            "AUDUSD": "DEXUSAL",
            "USDCAD": "DEXCAUS",
            "USDCHF": "DEXSZUS",
        }
        
        series_id = series_map.get(currency_pair.upper())
        if not series_id:
            return None
        
        data = self.get_series(series_id)
        if data is not None and len(data) > 0:
            return float(data.iloc[-1])
        return None

    def get_currency_history(self, currency_pair: str, days: int = 365) -> Optional[pd.Series]:
        """Get historical currency data."""
        series_map = {
            "EURUSD": "DEXUSEU",
            "GBPUSD": "DEXUSUK",
            "USDJPY": "DEXJPUS",
            "USDCNY": "DEXCHUS",
            "AUDUSD": "DEXUSAL",
            "USDCAD": "DEXCAUS",
            "USDCHF": "DEXSZUS",
        }
        
        series_id = series_map.get(currency_pair.upper())
        if not series_id:
            return None
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.get_series(series_id, start_date=start_date, end_date=end_date)

