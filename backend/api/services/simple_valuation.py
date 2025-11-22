import logging
import os
import statistics
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
            multiples_val = self._calculate_historical_multiples_valuation(ticker, metrics, current_price)

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

    def _calculate_historical_multiples_valuation(
        self,
        ticker: str,
        current_metrics: Dict[str, Any],
        current_price: float,
    ) -> Dict[str, Any]:
        """
        Valuation based on 5-year historical average multiples.
        """
        try:
            # We need historical ratios. FMPClient has get_financial_ratios(period='annual')
            # We'll fetch last 5 years annual
            hist_ratios = self.fmp_client.get_financial_ratios(ticker, period="annual")
            if not hist_ratios:
                return {"method": "N/A", "fair_value": 0, "explanation": "No historical EV multiples"}

            last_entries = hist_ratios[:5]
            ev_sales_samples = [
                x.get("enterpriseValueOverRevenue")
                for x in last_entries
                if x.get("enterpriseValueOverRevenue")
            ]
            ev_ebitda_samples = [
                x.get("enterpriseValueOverEBITDA")
                for x in last_entries
                if x.get("enterpriseValueOverEBITDA")
            ]

            ev_sales_median = statistics.median(ev_sales_samples) if ev_sales_samples else None
            ev_ebitda_median = statistics.median(ev_ebitda_samples) if ev_ebitda_samples else None

            shares = current_metrics.get("sharesOutstanding") or current_metrics.get("weightedAverageShsOutDil")
            if not shares:
                market_cap = current_metrics.get("marketCap")
                if market_cap and current_price:
                    shares = market_cap / current_price

            revenue = current_metrics.get("revenueTTM")
            if not revenue and shares and current_metrics.get("revenuePerShareTTM"):
                revenue = current_metrics["revenuePerShareTTM"] * shares

            ebitda = current_metrics.get("ebitdaTTM")
            if not ebitda and shares and current_metrics.get("ebitdaPerShareTTM"):
                ebitda = current_metrics["ebitdaPerShareTTM"] * shares

            net_debt = current_metrics.get("netDebt")
            if net_debt is None:
                debt = current_metrics.get("totalDebt")
                cash = current_metrics.get("cashAndShortTermInvestments")
                if debt is not None and cash is not None:
                    net_debt = debt - cash
                else:
                    net_debt = 0

            fair_values = []
            breakdown: Dict[str, float] = {}

            if ev_sales_median and revenue and shares:
                enterprise_value_sales = ev_sales_median * revenue
                equity_value_sales = enterprise_value_sales - (net_debt or 0)
                if equity_value_sales > 0:
                    fair_value_sales = equity_value_sales / shares
                    fair_values.append(fair_value_sales)
                    breakdown["ev_sales"] = round(fair_value_sales, 2)

            if ev_ebitda_median and ebitda and shares:
                enterprise_value_ebitda = ev_ebitda_median * ebitda
                equity_value_ebitda = enterprise_value_ebitda - (net_debt or 0)
                if equity_value_ebitda > 0:
                    fair_value_ebitda = equity_value_ebitda / shares
                    fair_values.append(fair_value_ebitda)
                    breakdown["ev_ebitda"] = round(fair_value_ebitda, 2)

            if not fair_values:
                return {
                    "method": "EV Multiples (3-5y median)",
                    "fair_value": 0,
                    "ev_sales_median": ev_sales_median,
                    "ev_ebitda_median": ev_ebitda_median,
                    "breakdown": breakdown,
                    "explanation": "Insufficient revenue/EBITDA data to compute EV multiples.",
                }

            fair_value = sum(fair_values) / len(fair_values)
            return {
                "method": "EV Multiples (3-5y median)",
                "fair_value": round(fair_value, 2),
                "ev_sales_median": round(ev_sales_median, 2) if ev_sales_median else None,
                "ev_ebitda_median": round(ev_ebitda_median, 2) if ev_ebitda_median else None,
                "breakdown": breakdown,
                "explanation": "Fair value calculated using median EV/Sales and EV/EBITDA over the last 3-5 years.",
            }

        except Exception as e:
            LOGGER.error(f"Multiples Valuation error: {e}")
            return {"method": "Error", "fair_value": 0, "explanation": "EV multiple calculation failed"}

    def _calculate_multiples(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "method": "Market Multiples",
            "multiples": {
                "EV/Sales": metrics.get("enterpriseValueOverRevenueTTM") or metrics.get("enterpriseValueOverRevenue") or 0,
                "EV/EBITDA": metrics.get("enterpriseValueOverEBITDATTM") or metrics.get("enterpriseValueOverEBITDA") or 0,
            },
            "explanation": "Enterprise value multiples vs. sales and EBITDA."
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
