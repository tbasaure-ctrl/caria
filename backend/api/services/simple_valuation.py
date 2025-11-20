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

            # 3. Calculate Reverse DCF (Implied Growth)
            reverse_dcf = self._calculate_reverse_dcf(metrics, current_price, dcf["assumptions"])

            # 4. Calculate Multiples Valuation (Fair Value based on Hist Avg)
            multiples_val = self._calculate_historical_multiples_valuation(ticker, metrics)

            # 5. Current Multiples
            current_multiples = self._calculate_multiples(metrics)

            return {
                "ticker": ticker,
                "currency": metrics.get("currency", "USD"),
                "current_price": current_price,
                "dcf": dcf,
                "reverse_dcf": reverse_dcf,
                "multiples_valuation": multiples_val,
                "multiples": current_multiples
            }

        except Exception as e:
            LOGGER.exception(f"Error in simple valuation for {ticker}: {e}")
            return self._fallback_response(ticker, current_price, str(e))

    def _fetch_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch necessary metrics from FMP."""
        try:
            # Get Key Metrics (Annual/Quarterly)
            key_metrics = self.fmp_client.get_key_metrics(ticker, period="quarter")
            
            # Get Financial Ratios
            ratios = self.fmp_client.get_financial_ratios(ticker, period="quarter")
            
            # Get Growth
            growth = self.fmp_client.get_financial_growth(ticker, period="quarter")

            return {
                **(key_metrics[0] if key_metrics else {}),
                **(ratios[0] if ratios else {}),
                "growth": growth[0] if growth else {}
            }
        except Exception as e:
            LOGGER.error(f"Error fetching FMP data: {e}")
            return None

    def _calculate_dcf(self, metrics: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Perform a simplified DCF calculation."""
        try:
            fcf_per_share = metrics.get("freeCashFlowPerShareTTM") or metrics.get("freeCashFlowPerShare")
            if not fcf_per_share:
                 # Fallback to EPS
                fcf_per_share = metrics.get("netIncomePerShareTTM") or metrics.get("netIncomePerShare") or 0
            
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

    def _calculate_reverse_dcf(self, metrics: Dict[str, Any], current_price: float, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Implied Growth Rate (Reverse DCF).
        Solves for 'g' such that DCF(g) == current_price.
        """
        try:
            fcf_per_share = metrics.get("freeCashFlowPerShareTTM") or metrics.get("freeCashFlowPerShare")
            if not fcf_per_share:
                fcf_per_share = metrics.get("netIncomePerShareTTM") or metrics.get("netIncomePerShare") or 0
            
            if fcf_per_share <= 0:
                return {"implied_growth_rate": 0, "explanation": "N/A (Negative FCF)"}

            discount_rate = assumptions.get("discount_rate", 0.10)
            terminal_growth = assumptions.get("terminal_growth_rate", 0.03)
            years = 5

            # Binary search for implied growth
            low = -0.50
            high = 1.00
            implied_growth = 0.0
            
            for _ in range(20): # 20 iterations is enough precision
                mid = (low + high) / 2
                
                # Calculate DCF with 'mid' growth
                future_cash_flows = []
                for i in range(1, years + 1):
                    fcf = fcf_per_share * ((1 + mid) ** i)
                    future_cash_flows.append(fcf / ((1 + discount_rate) ** i))
                
                terminal_val = (fcf_per_share * ((1 + mid) ** years) * (1 + terminal_growth)) / (discount_rate - terminal_growth)
                terminal_val_discounted = terminal_val / ((1 + discount_rate) ** years)
                fair_val = sum(future_cash_flows) + terminal_val_discounted
                
                if fair_val > current_price:
                    high = mid
                else:
                    low = mid
                implied_growth = mid

            return {
                "implied_growth_rate": round(implied_growth, 4),
                "explanation": f"The market is pricing in a {implied_growth*100:.1f}% annual growth rate for the next 5 years."
            }
        except Exception as e:
            LOGGER.error(f"Reverse DCF error: {e}")
            return {"implied_growth_rate": 0, "explanation": "Calculation failed"}

    def _calculate_historical_multiples_valuation(self, ticker: str, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valuation based on 5-year historical average multiples.
        """
        try:
            # We need historical ratios. FMPClient has get_financial_ratios(period='annual')
            # We'll fetch last 5 years annual
            hist_ratios = self.fmp_client.get_financial_ratios(ticker, period="annual")
            if not hist_ratios:
                return {"method": "N/A", "fair_value": 0, "explanation": "No historical data"}

            # Take last 5 entries (assuming sorted desc date)
            last_5 = hist_ratios[:5]
            
            avg_pe = sum(x.get("peRatio", 0) for x in last_5) / len(last_5)
            avg_pb = sum(x.get("priceToBookRatio", 0) for x in last_5) / len(last_5)
            
            # Current metrics
            eps = current_metrics.get("netIncomePerShareTTM") or current_metrics.get("netIncomePerShare") or 0
            bps = current_metrics.get("bookValuePerShareTTM") or current_metrics.get("bookValuePerShare") or 0
            
            vals = []
            if avg_pe > 0 and eps > 0:
                vals.append(avg_pe * eps)
            if avg_pb > 0 and bps > 0:
                vals.append(avg_pb * bps)
            
            if not vals:
                return {"method": "N/A", "fair_value": 0, "explanation": "Negative earnings/book value"}
            
            fair_value = sum(vals) / len(vals)
            
            return {
                "method": "5y Avg Multiples (PE & PB)",
                "fair_value": round(fair_value, 2),
                "avg_pe": round(avg_pe, 2),
                "avg_pb": round(avg_pb, 2),
                "explanation": f"Fair value derived from 5y Avg PE ({avg_pe:.1f}x) and PB ({avg_pb:.1f}x)."
            }

        except Exception as e:
            LOGGER.error(f"Multiples Valuation error: {e}")
            return {"method": "Error", "fair_value": 0, "explanation": "Calculation failed"}

    def _calculate_multiples(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "method": "Market Multiples",
            "multiples": {
                "PE": metrics.get("peRatioTTM") or metrics.get("peRatio") or 0,
                "PB": metrics.get("pbRatioTTM") or metrics.get("pbRatio") or 0,
                "EV/EBITDA": metrics.get("enterpriseValueOverEBITDATTM") or metrics.get("enterpriseValueOverEBITDA") or 0,
                "Div Yield": metrics.get("dividendYieldPercentageTTM") or metrics.get("dividendYieldPercentage") or 0
            },
            "explanation": "Current market multiples."
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
            "reverse_dcf": {"implied_growth_rate": 0, "explanation": "N/A"},
            "multiples_valuation": {"method": "Error", "fair_value": 0, "explanation": "N/A"},
            "multiples": {
                "method": "Error",
                "multiples": {},
                "explanation": "Data unavailable"
            }
        }
