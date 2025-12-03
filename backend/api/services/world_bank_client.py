"""
World Bank Open Data API Client.

Provides access to World Bank economic indicators:
- External Debt
- Foreign Exchange Reserves
- GDP Growth
- Credit-to-GDP ratios
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
import requests

LOGGER = logging.getLogger("caria.services.world_bank_client")

WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2"


class WorldBankClient:
    """Client for World Bank Open Data API (public, no key required)."""

    def __init__(self):
        self.base_url = WORLD_BANK_BASE_URL
        LOGGER.info("World Bank API client initialized (public API)")

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make a request to World Bank API."""
        try:
            url = f"{self.base_url}/{endpoint}"
            params["format"] = "json"
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            LOGGER.error(f"Error fetching World Bank data: {e}")
            return None

    def get_indicator(self, indicator_code: str, country_code: str = "all", 
                     start_year: Optional[int] = None, end_year: Optional[int] = None) -> Optional[List[Dict]]:
        """
        Get indicator data for country/countries.
        
        Args:
            indicator_code: WB indicator code (e.g., "DT.DOD.DECT.CD" for external debt)
            country_code: ISO country code or "all"
            start_year: Start year for data
            end_year: End year for data
        """
        params = {
            "country": country_code,
            "indicator": indicator_code,
        }
        
        if start_year:
            params["date"] = f"{start_year}:{end_year or datetime.now().year}"
        
        data = self._make_request("country", params)
        
        if data and len(data) > 1 and isinstance(data[1], list):
            return data[1]  # Second element contains the data
        return None

    def get_external_debt(self, country_code: str) -> Optional[float]:
        """Get total external debt stock (DT.DOD.DECT.CD) in current USD."""
        data = self.get_indicator("DT.DOD.DECT.CD", country_code)
        if data and len(data) > 0:
            # Get most recent non-null value
            for entry in reversed(data):
                if entry.get("value") is not None:
                    try:
                        return float(entry["value"])
                    except (ValueError, TypeError):
                        continue
        return None

    def get_reserves(self, country_code: str) -> Optional[float]:
        """Get total reserves (FI.RES.TOTL.CD) in current USD."""
        data = self.get_indicator("FI.RES.TOTL.CD", country_code)
        if data and len(data) > 0:
            for entry in reversed(data):
                if entry.get("value") is not None:
                    try:
                        return float(entry["value"])
                    except (ValueError, TypeError):
                        continue
        return None

    def get_gdp_growth(self, country_code: str) -> Optional[float]:
        """Get GDP growth rate (NY.GDP.MKTP.KD.ZG) as percentage."""
        data = self.get_indicator("NY.GDP.MKTP.KD.ZG", country_code)
        if data and len(data) > 0:
            for entry in reversed(data):
                if entry.get("value") is not None:
                    try:
                        return float(entry["value"])
                    except (ValueError, TypeError):
                        continue
        return None

    def get_short_term_debt(self, country_code: str) -> Optional[float]:
        """Get short-term external debt (DT.DOD.DSTC.CD) in current USD."""
        data = self.get_indicator("DT.DOD.DSTC.CD", country_code)
        if data and len(data) > 0:
            for entry in reversed(data):
                if entry.get("value") is not None:
                    try:
                        return float(entry["value"])
                    except (ValueError, TypeError):
                        continue
        return None

    def calculate_reserve_adequacy(self, country_code: str) -> Optional[float]:
        """Calculate reserves to short-term debt ratio (Greenspan-Guidotti Rule)."""
        reserves = self.get_reserves(country_code)
        short_term_debt = self.get_short_term_debt(country_code)
        
        if reserves is not None and short_term_debt is not None and short_term_debt > 0:
            return reserves / short_term_debt
        return None

