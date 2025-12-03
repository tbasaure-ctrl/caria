"""
OECD SDMX-JSON API Client.

Provides access to OECD economic indicators:
- Composite Leading Indicators (CLI)
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
import requests

LOGGER = logging.getLogger("caria.services.oecd_client")

OECD_BASE_URL = "https://stats.oecd.org/SDMX-JSON/data"


class OECDClient:
    """Client for OECD SDMX-JSON API (public, no key required, but has rate limits)."""

    def __init__(self):
        self.base_url = OECD_BASE_URL
        LOGGER.info("OECD API client initialized (public API, rate limit: ~20 req/hour)")

    def _make_request(self, query: str) -> Optional[Dict]:
        """Make a request to OECD API."""
        try:
            url = f"{self.base_url}/{query}"
            # OECD API expects specific format: /data/{dataflow}/{key}/all
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            LOGGER.error(f"Error fetching OECD data: {e}")
            return None

    def get_cli(self, country_code: str) -> Optional[float]:
        """
        Get Composite Leading Indicator (CLI) amplitude adjusted.
        
        OECD CLI dataflow structure:
        CLI/{country_code}.M/all?detail=full
        
        Note: OECD API uses complex SDMX structure. This is simplified.
        """
        # OECD country codes: USA, GBR, JPN, DEU, FRA, etc.
        oecd_country_map = {
            "US": "USA",
            "GB": "GBR",
            "JP": "JPN",
            "DE": "DEU",
            "FR": "FRA",
            "IT": "ITA",
            "CA": "CAN",
            "AU": "AUS",
        }
        
        oecd_code = oecd_country_map.get(country_code.upper())
        if not oecd_code:
            return None
        
        try:
            # CLI amplitude adjusted monthly data
            # Format: CLI/{country}.M/all?detail=full
            query = f"CLI/{oecd_code}.M/all?detail=full"
            data = self._make_request(query)
            
            if data:
                # Parse SDMX-JSON structure
                # Structure: data -> dataSets -> [0] -> series -> {key} -> observations -> {index} -> [value]
                # This is simplified - actual parsing requires navigating SDMX structure
                # For now, return None and implement full parsing in service layer
                LOGGER.warning("OECD CLI parsing not fully implemented - requires SDMX structure navigation")
                return None
        except Exception as e:
            LOGGER.warning(f"Could not fetch OECD CLI for {country_code}: {e}")
        
        return None

    def get_cli_history(self, country_code: str, months: int = 24) -> Optional[List[Dict[str, Any]]]:
        """Get historical CLI data."""
        # Similar to get_cli but returns time series
        # Implementation would parse SDMX structure for time series
        return None

