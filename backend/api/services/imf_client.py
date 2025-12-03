"""
IMF (International Monetary Fund) JSON REST API Client.

Provides access to IMF economic data:
- Commodity Terms of Trade
- Direction of Trade Statistics
- World Economic Outlook data
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
import requests

LOGGER = logging.getLogger("caria.services.imf_client")

IMF_BASE_URL = "https://www.imf.org/external/datamapper/api/v1"


class IMFClient:
    """Client for IMF JSON REST API (public, no key required)."""

    def __init__(self):
        self.base_url = IMF_BASE_URL
        LOGGER.info("IMF API client initialized (public API)")

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """Make a request to IMF API."""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, params=params or {}, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            LOGGER.error(f"Error fetching IMF data: {e}")
            return None

    def get_terms_of_trade(self, country_code: str) -> Optional[float]:
        """
        Get Commodity Terms of Trade index.
        
        Note: IMF API structure is hierarchical. This is a simplified implementation.
        For production, you may need to query the DataStructure endpoint first.
        """
        # IMF uses complex dataflow/dimension structure
        # This is a placeholder - actual implementation may require:
        # 1. Query DataStructure to get codes
        # 2. Query specific dataflow (e.g., "CTOT")
        # 3. Parse nested JSON response
        
        # Example endpoint structure (simplified):
        # /CTOT/{country_code}
        try:
            endpoint = f"CTOT/{country_code}"
            data = self._make_request(endpoint)
            
            if data and "values" in data:
                # Extract most recent value
                values = data.get("values", {})
                if values:
                    # Get latest date
                    latest_date = max(values.keys())
                    return float(values[latest_date])
        except Exception as e:
            LOGGER.warning(f"Could not fetch Terms of Trade for {country_code}: {e}")
        
        return None

    def get_current_account_balance(self, country_code: str) -> Optional[float]:
        """Get current account balance as % of GDP from WEO database."""
        # WEO data structure: /WEO/{country_code}/BCA_NGDPD
        try:
            endpoint = f"WEO/{country_code}/BCA_NGDPD"
            data = self._make_request(endpoint)
            
            if data and "values" in data:
                values = data.get("values", {})
                if values:
                    latest_date = max(values.keys())
                    return float(values[latest_date])
        except Exception as e:
            LOGGER.warning(f"Could not fetch current account balance for {country_code}: {e}")
        
        return None

    def get_fiscal_balance(self, country_code: str) -> Optional[float]:
        """Get fiscal balance as % of GDP from WEO database."""
        # WEO data structure: /WEO/{country_code}/GGB_NGDPD
        try:
            endpoint = f"WEO/{country_code}/GGB_NGDPD"
            data = self._make_request(endpoint)
            
            if data and "values" in data:
                values = data.get("values", {})
                if values:
                    latest_date = max(values.keys())
                    return float(values[latest_date])
        except Exception as e:
            LOGGER.warning(f"Could not fetch fiscal balance for {country_code}: {e}")
        
        return None

    def calculate_twin_deficits(self, country_code: str) -> Optional[float]:
        """Calculate twin deficits (fiscal + current account) as % of GDP."""
        fiscal = self.get_fiscal_balance(country_code)
        current_account = self.get_current_account_balance(country_code)
        
        if fiscal is not None and current_account is not None:
            return fiscal + current_account
        return None

