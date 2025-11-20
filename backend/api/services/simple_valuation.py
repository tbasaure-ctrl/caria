import logging
import os
from typing import Any, Dict, Optional

from caria.ingestion.clients.fmp_client import FMPClient

LOGGER = logging.getLogger("caria.services.simple_valuation")

class SimpleValuationService:
    """
    A robust, simplified valuation service that:
    1. Fetches key metrics from FMP.
    2. Performs basic DCF and Multiples calculations.
    3. Handles errors gracefully and returns partial data if possible.
    """

    def __init__(self, fmp_client: Optional[FMPClient] = None):
        self.fmp_client = fmp_client or FMPClient(api_key=os.getenv("FMP_API_KEY", "").strip())

    def get_valuation(self, ticker: str, current_price: float) -> Dict[str, Any]:
        """
        Get a comprehensive valuation for a ticker.
        """
        try:
            # 1. Fetch Data
            metrics = self._fetch_metrics(ticker)
            if not metrics:
                LOGGER.warning(f"Could not fetch metrics for {ticker}")
                return self._fallback_response(ticker, current_price, "Data unavailable")

            # 2. Calculate DCF
            dcf = self._calculate_dcf(metrics, current_price)

            # 3. Calculate Multiples
            multiples = self._calculate_multiples(metrics)

            return {
                "ticker": ticker,
                "currency": metrics.get("currency", "USD"),
                "current_price": current_price,
                "dcf": dcf,
                "multiples": multiples
            }

        except Exception as e:
            LOGGER.exception(f"Error in simple valuation for {ticker}: {e}")
            return self._fallback_response(ticker, current_price, str(e))

    def _fetch_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch necessary metrics from FMP."""
        try:
            # Get Key Metrics (TTM)
            key_metrics = self.fmp_client.get_key_metrics_ttm(ticker)
            if not key_metrics:
                return None
            
            # Get Financial Ratios (TTM)
            ratios = self.fmp_client.get_ratios_ttm(ticker)
            
            # Get Profile (for beta, industry, etc.)
            profile = self.fmp_client.get_company_profile(ticker)

            # Get Growth (for growth rates)
            growth = self.fmp_client.get_financial_growth(ticker)

            return {
                **(key_metrics[0] if key_metrics else {}),
                **(ratios[0] if ratios else {}),
                **(profile[0] if profile else {}),
                "growth": growth[0] if growth else {}
            }
        except Exception as e:
            LOGGER.error(f"Error fetching FMP data: {e}")
            return None

    def _calculate_dcf(self, metrics: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Perform a simplified DCF calculation."""
        try:
            fcf_per_share = metrics.get("freeCashFlowPerShareTTM", 0)
            if not fcf_per_share or fcf_per_share <= 0:
                 # Fallback to EPS if FCF is negative/missing
                fcf_per_share = metrics.get("netIncomePerShareTTM", 0)
            
            if fcf_per_share <= 0:
                 return {
                    "method": "N/A (Negative FCF/Earnings)",
                    "fair_value_per_share": 0,
                    "upside_percent": 0,
                    "implied_return_cagr": 0,
                    "assumptions": self._default_assumptions(),
                    "explanation": "Cannot calculate DCF with negative Free Cash Flow or Earnings."
                }

            # Assumptions
            growth_rate = min(metrics.get("growth", {}).get("freeCashFlowGrowth", 0.10), 0.15) # Cap at 15%
            if growth_rate < 0.02: growth_rate = 0.05 # Min 5%
            
            discount_rate = 0.10 # Simplified WACC
            terminal_growth = 0.03
            years = 5

            # Projection
            future_cash_flows = []
            for i in range(1, years + 1):
                fcf = fcf_per_share * ((1 + growth_rate) ** i)
                future_cash_flows.append(fcf / ((1 + discount_rate) ** i))

            # Terminal Value
            terminal_val = (fcf_per_share * ((1 + growth_rate) ** years) * (1 + terminal_growth)) / (discount_rate - terminal_growth)
            terminal_val_discounted = terminal_val / ((1 + discount_rate) ** years)

            fair_value = sum(future_cash_flows) + terminal_val_discounted
            upside = ((fair_value - current_price) / current_price) * 100

            return {
                "method": "Simplified DCF (5y Growth)",
                "fair_value_per_share": round(fair_value, 2),
                "upside_percent": round(upside, 2),
                "implied_return_cagr": round(discount_rate + (upside/100)/years, 4), # Rough approx
                "assumptions": {
                    "fcf_yield_start": round(fcf_per_share/current_price, 4),
                    "high_growth_rate": round(growth_rate, 4),
                    "high_growth_years": years,
                    "fade_years": 5,
                    "terminal_growth_rate": terminal_growth,
                    "discount_rate": discount_rate,
                    "horizon_years": 10
                },
                "explanation": f"Based on FCF/share of ${fcf_per_share:.2f} growing at {growth_rate*100:.1f}% for 5 years."
            }

        except Exception as e:
            LOGGER.error(f"DCF Calculation error: {e}")
            return {
                "method": "Error",
                "fair_value_per_share": 0,
                "upside_percent": 0,
                "implied_return_cagr": 0,
                "assumptions": self._default_assumptions(),
                "explanation": "Calculation failed."
            }

    def _calculate_multiples(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "method": "Market Multiples",
            "multiples": {
                "PE": metrics.get("peRatioTTM", 0),
                "PB": metrics.get("pbRatioTTM", 0),
                "EV/EBITDA": metrics.get("enterpriseValueOverEBITDATTM", 0),
                "Div Yield": metrics.get("dividendYieldPercentageTTM", 0)
            },
            "explanation": "Current market multiples based on TTM data."
        }

    def _default_assumptions(self):
        return {
            "fcf_yield_start": 0,
            "high_growth_rate": 0,
            "high_growth_years": 0,
            "fade_years": 0,
            "terminal_growth_rate": 0,
            "discount_rate": 0,
            "horizon_years": 0
        }

    def _fallback_response(self, ticker, price, error_msg):
        return {
            "ticker": ticker,
            "currency": "USD",
            "current_price": price,
            "dcf": {
                "method": "Error",
                "fair_value_per_share": 0,
                "upside_percent": 0,
                "implied_return_cagr": 0,
                "assumptions": self._default_assumptions(),
                "explanation": f"Valuation failed: {error_msg}"
            },
            "multiples": {
                "method": "Error",
                "multiples": {},
                "explanation": "Data unavailable"
            }
        }
