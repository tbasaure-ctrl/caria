import logging
import statistics
from typing import Any, Dict, List, Optional

from api.services.openbb_client import openbb_client

LOGGER = logging.getLogger("caria.services.simple_valuation")


def _sorted_by_date(data: List[Dict[str, Any]], reverse: bool = True) -> List[Dict[str, Any]]:
    def _key(item: Dict[str, Any]) -> str:
        return item.get("date") or item.get("period_ending") or item.get("period") or ""

    return sorted(data, key=_key, reverse=reverse)


def _first_value(record: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if record is None:
            break
        value = record.get(key)
        if value not in (None, "", "NA"):
            return value
    return None


class SimpleValuationService:
    """
    Valuation toolkit backed by OpenBB data.
    """

    def __init__(self) -> None:
        self.client = openbb_client

    def get_valuation(self, ticker: str, current_price: float) -> Dict[str, Any]:
        try:
            dataset = self._fetch_dataset(ticker)
            if not dataset:
                LOGGER.warning("Dataset missing for %s", ticker)
                return self._fallback_response(ticker, current_price, "OpenBB data unavailable")

            latest_price = dataset.get("latest_price") or current_price
            if not latest_price:
                return self._fallback_response(ticker, current_price, "No price data")

            metrics = self._build_metrics(dataset)
            if not metrics:
                return self._fallback_response(ticker, latest_price, "Metrics unavailable")

            dcf = self._calculate_dcf(metrics, latest_price)
            reverse_dcf = self._calculate_reverse_dcf(metrics, latest_price, dcf["assumptions"])
            multiples_val = self._calculate_historical_multiples_valuation(metrics, latest_price)
            current_multiples = self._calculate_multiples(metrics)

            return {
                "ticker": ticker.upper(),
                "currency": metrics.get("currency", "USD"),
                "current_price": latest_price,
                "dcf": dcf,
                "reverse_dcf": reverse_dcf,
                "multiples_valuation": multiples_val,
                "multiples": current_multiples,
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Error in simple valuation for %s: %s", ticker, exc)
            return self._fallback_response(ticker, current_price, str(exc))

    def _fetch_dataset(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            return self.client.get_ticker_data(ticker)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("OpenBB fetch failed for %s: %s", ticker, exc)
            return None

    def _build_metrics(self, dataset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        income = _sorted_by_date(dataset["financials"].get("income_statement", []))
        balance = _sorted_by_date(dataset["financials"].get("balance_sheet", []))
        cash = _sorted_by_date(dataset["financials"].get("cash_flow", []))
        multiples_history = dataset.get("multiples", [])

        if not income or not cash:
            return None

        latest_income = income[0]
        latest_cash = cash[0]
        latest_balance = balance[0] if balance else {}
        latest_multiples = multiples_history[0] if multiples_history else {}

        shares = (
            _first_value(latest_income, ["weightedAverageShsOutDil", "weightedAverageShsOut"])
            or _first_value(latest_multiples, ["shares_outstanding", "sharesOutstanding"])
            or _first_value(latest_balance, ["commonStockSharesOutstanding"])
        )

        revenue = _first_value(latest_income, ["totalRevenue", "revenue"])
        gross_profit = _first_value(latest_income, ["grossProfit"])
        operating_income = _first_value(latest_income, ["operatingIncome", "ebit"])
        ebitda = _first_value(latest_income, ["ebitda"])
        fcf = _first_value(latest_cash, ["freeCashFlow", "fcf"])
        currency = latest_income.get("currency") or latest_multiples.get("currency") or "USD"

        fcf_per_share = None
        if fcf and shares not in (None, 0):
            fcf_per_share = fcf / shares

        revenue_growth = None
        if len(income) >= 2:
            prev_revenue = _first_value(income[1], ["totalRevenue", "revenue"])
            if revenue and prev_revenue and prev_revenue != 0:
                revenue_growth = (revenue - prev_revenue) / abs(prev_revenue)

        fcf_growth = None
        if len(cash) >= 2 and len(income) >= 2:
            prev_fcf = _first_value(cash[1], ["freeCashFlow", "fcf"])
            prev_shares = (
                _first_value(income[1], ["weightedAverageShsOutDil", "weightedAverageShsOut"])
                or shares
            )
            if prev_fcf and prev_shares and fcf_per_share and prev_shares != 0:
                prev_fcf_per_share = prev_fcf / prev_shares
                if prev_fcf_per_share != 0:
                    fcf_growth = (fcf_per_share - prev_fcf_per_share) / abs(prev_fcf_per_share)

        net_debt = (
            _first_value(latest_balance, ["totalDebt", "shortLongTermDebtTotal", "longTermDebtTotal"])
            or 0
        ) - (_first_value(latest_balance, ["cashAndCashEquivalents", "cash"]) or 0)

        revenue_series = [x for x in (revenue,) if x is not None]

        return {
            "currency": currency,
            "freeCashFlowPerShare": fcf_per_share,
            "growth": {"freeCashFlowGrowth": fcf_growth or revenue_growth or 0.05},
            "enterpriseValueOverRevenue": _first_value(
                latest_multiples,
                ["ev_to_revenue", "enterpriseValueRevenueMultiple", "evSales", "enterpriseValueOverRevenue"],
            ),
            "enterpriseValueOverEBITDA": _first_value(
                latest_multiples,
                ["ev_to_ebitda", "enterpriseValueToEbitda", "evEbitda", "enterpriseValueOverEBITDA"],
            ),
            "enterpriseValueOverRevenueTTM": _first_value(
                latest_multiples,
                ["enterpriseValueOverRevenueTTM", "ev_to_revenue"],
            ),
            "enterpriseValueOverEBITDATTM": _first_value(
                latest_multiples,
                ["enterpriseValueOverEBITDATTM", "ev_to_ebitda"],
            ),
            "netDebt": net_debt,
            "sharesOutstanding": shares,
            "revenueTTM": revenue,
            "ebitdaTTM": ebitda,
            "gross_margin": (gross_profit / revenue) if revenue and gross_profit else None,
            "operating_margin": (operating_income / revenue) if revenue and operating_income else None,
            "multiples_history": multiples_history,
            "income_statements": income,
            "cash_flow_statements": cash,
            "revenue_series": revenue_series,
        }

    def _calculate_dcf(self, metrics: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Perform a simplified DCF calculation."""
        try:
            fcf_per_share = metrics.get("freeCashFlowPerShare")
            if not fcf_per_share:
                 # Fallback to EPS
                return {
                    "method": "N/A",
                    "fair_value_per_share": 0,
                    "upside_percent": 0,
                    "implied_return_cagr": 0,
                    "assumptions": self._default_assumptions(),
                    "explanation": "Cannot calculate DCF without Free Cash Flow.",
                }
            
            if fcf_per_share <= 0:
                return {
                    "method": "N/A (Negative FCF)",
                    "fair_value_per_share": 0,
                    "upside_percent": 0,
                    "implied_return_cagr": 0,
                    "assumptions": self._default_assumptions(),
                    "explanation": "Cannot calculate DCF with negative Free Cash Flow.",
                }

            # Assumptions
            growth_rate = metrics.get("growth", {}).get("freeCashFlowGrowth") or 0.08
            growth_rate = max(min(growth_rate, 0.18), 0.03)
            
            discount_rate = 0.10  # Simplified WACC
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

    def _calculate_historical_multiples_valuation(self, metrics: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Valuation based on 3-5 year median EV/Sales and EV/EBITDA using OpenBB history.
        """
        try:
            hist_ratios = metrics.get("multiples_history") or []
            if not hist_ratios:
                return {"method": "N/A", "fair_value": 0, "explanation": "No historical EV multiples"}

            last_entries = hist_ratios[:8]
            ev_sales_samples = [
                _first_value(
                    x,
                    ["ev_to_revenue", "enterpriseValueRevenueMultiple", "evSales", "enterpriseValueOverRevenue"],
                )
                for x in last_entries
                if _first_value(
                    x,
                    ["ev_to_revenue", "enterpriseValueRevenueMultiple", "evSales", "enterpriseValueOverRevenue"],
                )
            ]
            ev_ebitda_samples = [
                _first_value(
                    x,
                    ["ev_to_ebitda", "enterpriseValueToEbitda", "evEbitda", "enterpriseValueOverEBITDA"],
                )
                for x in last_entries
                if _first_value(
                    x,
                    ["ev_to_ebitda", "enterpriseValueToEbitda", "evEbitda", "enterpriseValueOverEBITDA"],
                )
            ]

            ev_sales_median = statistics.median(ev_sales_samples) if ev_sales_samples else None
            ev_ebitda_median = statistics.median(ev_ebitda_samples) if ev_ebitda_samples else None

            shares = metrics.get("sharesOutstanding")
            if not shares:
                return {
                    "method": "EV Multiples (3-5y median)",
                    "fair_value": 0,
                    "ev_sales_median": ev_sales_median,
                    "ev_ebitda_median": ev_ebitda_median,
                    "breakdown": {},
                    "explanation": "Shares outstanding unavailable",
                }

            revenue = metrics.get("revenueTTM")
            ebitda = metrics.get("ebitdaTTM")
            net_debt = metrics.get("netDebt") or 0

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
