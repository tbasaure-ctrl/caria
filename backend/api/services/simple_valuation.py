import logging
import os
import statistics
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

from api.services.openbb_client import OpenBBClient

LOGGER = logging.getLogger("caria.services.simple_valuation")

class SimpleValuationService:
    """
    A robust, simplified valuation service that:
    1. Fetches key metrics from OpenBB (FMP provider).
    2. Performs basic DCF and Multiples calculations.
    3. Handles errors gracefully and returns partial data if possible.
    """

    def __init__(self, obb_client: Optional[OpenBBClient] = None):
        self.obb_client = obb_client or OpenBBClient()
        self._fetch_cache = {}
        self._cache_ttl = 300  # 5 minutes

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
        """Fetch necessary metrics from OpenBB/FMP with caching. Ensures FMP is used first."""
        # Check cache first
        cache_key = f"{ticker}_metrics"
        if cache_key in self._fetch_cache:
            cached_data, timestamp = self._fetch_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self._cache_ttl):
                LOGGER.debug(f"Using cached metrics for {ticker}")
                return cached_data

        try:
            # Force FMP provider for critical data
            LOGGER.info(f"Fetching metrics for {ticker} from FMP via OpenBB...")
            
            # Try to get comprehensive data
            data = self.obb_client.get_ticker_data(ticker)
            
            # Extract Data
            key_metrics = self._extract_first(data.get("key_metrics"))
            ratios = self._extract_first(data.get("multiples"))
            
            # Handle financials - it's a dict with income_statement, cash_flow
            financials_dict = data.get("financials", {})
            income_statement = self._extract_first(financials_dict.get("income_statement"))
            cash_flow = self._extract_first(financials_dict.get("cash_flow"))
            # Merge income and cash flow
            financials = {**(income_statement or {}), **(cash_flow or {})}
            
            growth = self._extract_first(data.get("growth"))

            # Map fields to expected format if necessary
            # FMP Key Metrics usually match well, but we might need to alias some
            mapped_metrics = {
                **(key_metrics or {}),
                **(ratios or {}),
                **(financials or {}),
                "growth": growth or {}
            }
            
            # Ensure critical fields exist (aliases)
            if "freeCashFlowPerShare" in mapped_metrics and "freeCashFlowPerShareTTM" not in mapped_metrics:
                mapped_metrics["freeCashFlowPerShareTTM"] = mapped_metrics["freeCashFlowPerShare"]
            
            if "netIncomePerShare" in mapped_metrics and "netIncomePerShareTTM" not in mapped_metrics:
                mapped_metrics["netIncomePerShareTTM"] = mapped_metrics["netIncomePerShare"]

            # If we don't have critical data, try direct FMP calls
            if not mapped_metrics.get("freeCashFlowPerShareTTM") and not mapped_metrics.get("revenueTTM"):
                LOGGER.warning(f"Missing critical metrics for {ticker}, trying direct FMP calls...")
                try:
                    # Try direct FMP calls for missing data
                    from openbb import obb
                    import os
                    fmp_key = os.getenv("FMP_API_KEY", "").strip()
                    if fmp_key:
                        obb.user.credentials.fmp_api_key = fmp_key

                    # Get key metrics directly
                    metrics_direct = obb.equity.fundamental.metrics(symbol=ticker, provider="fmp", limit=1)
                    if metrics_direct and hasattr(metrics_direct, 'to_df'):
                        df_metrics = metrics_direct.to_df()
                        if not df_metrics.empty:
                            direct_metrics = df_metrics.iloc[0].replace({float('nan'): None}).to_dict()
                            # Merge with existing
                            for k, v in direct_metrics.items():
                                if v is not None and k not in mapped_metrics:
                                    mapped_metrics[k] = v

                    # Get ratios directly
                    ratios_direct = obb.equity.fundamental.ratios(symbol=ticker, provider="fmp", limit=1)
                    if ratios_direct and hasattr(ratios_direct, 'to_df'):
                        df_ratios = ratios_direct.to_df()
                        if not df_ratios.empty:
                            direct_ratios = df_ratios.iloc[0].replace({float('nan'): None}).to_dict()
                            for k, v in direct_ratios.items():
                                if v is not None and k not in mapped_metrics:
                                    mapped_metrics[k] = v

                    LOGGER.info(f"✅ Successfully fetched additional metrics for {ticker} from FMP")
                except Exception as direct_e:
                    LOGGER.warning(f"Direct FMP calls failed for {ticker}: {direct_e}")

            # Validate we have at least some data
            if not mapped_metrics:
                LOGGER.error(f"No metrics retrieved for {ticker}")
                return None

            LOGGER.info(f"✅ Retrieved {len(mapped_metrics)} metrics for {ticker}")
            LOGGER.info(f"Available metric keys: {list(mapped_metrics.keys())[:20]}")

            # Cache the result
            cache_key = f"{ticker}_metrics"
            self._fetch_cache[cache_key] = (mapped_metrics, datetime.now())

            return mapped_metrics
        except Exception as e:
            LOGGER.error(f"Error fetching OpenBB/FMP data for {ticker}: {e}")
            return None

    def _extract_first(self, obb_object: Any) -> Dict[str, Any]:
        """Helper to extract the first row/result from an OBBject."""
        if not obb_object:
            return {}
        try:
            if hasattr(obb_object, 'to_df'):
                df = obb_object.to_df()
                if not df.empty:
                    # Convert to dict, handling potential NaN
                    return df.iloc[0].replace({float('nan'): None}).to_dict()
            if hasattr(obb_object, 'results'):
                if isinstance(obb_object.results, list) and obb_object.results:
                    res = obb_object.results[0]
                    if hasattr(res, 'model_dump'):
                        return res.model_dump()
                    if isinstance(res, dict):
                        return res
                    return res.__dict__
        except Exception as e:
            LOGGER.warning(f"Error extracting data from OBBject: {e}")
        return {}

    def _calculate_dcf(self, metrics: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Perform a simplified DCF calculation."""
        try:
            # Try multiple field name variations for FCF
            fcf_per_share = (
                metrics.get("freeCashFlowPerShareTTM") or
                metrics.get("freeCashFlowPerShare") or
                metrics.get("free_cash_flow_per_share") or
                metrics.get("fcfPerShare")
            )
            LOGGER.info(f"DCF calculation - FCF per share: {fcf_per_share}, Current price: {current_price}")

            if not fcf_per_share:
                 # Fallback to EPS with more field variations
                fcf_per_share = (
                    metrics.get("netIncomePerShareTTM") or
                    metrics.get("netIncomePerShare") or
                    metrics.get("eps") or
                    metrics.get("epsttm") or
                    0
                )
                LOGGER.info(f"No FCF found, using EPS fallback: {fcf_per_share}")

            if fcf_per_share <= 0:
                LOGGER.warning(f"Negative or zero FCF ({fcf_per_share}), trying P/E and P/B fallbacks")
                # Fallback 1: P/E Valuation with more field variations
                eps = (
                    metrics.get("netIncomePerShareTTM") or
                    metrics.get("netIncomePerShare") or
                    metrics.get("eps") or
                    metrics.get("epsttm") or
                    metrics.get("net_income_per_share")
                )
                LOGGER.info(f"Trying P/E fallback - EPS: {eps}, Current Price: {current_price}")

                if eps and eps > 0 and current_price > 0:
                    pe_ratio = current_price / eps
                    industry_avg_pe = 20  # More realistic for tech/growth stocks

                    # More lenient PE range
                    if 2 < pe_ratio < 150:  # Sanity check
                        fair_value = eps * industry_avg_pe
                        upside = ((fair_value - current_price) / current_price) * 100

                        LOGGER.info(f"P/E Fallback successful - EPS: {eps:.2f}, PE: {pe_ratio:.2f}, Fair Value: {fair_value:.2f}, Upside: {upside:.2f}%")
                        return {
                            "method": "P/E Fallback (No FCF)",
                            "fair_value_per_share": round(fair_value, 2),
                            "upside_percent": round(upside, 2),
                            "implied_return_cagr": round(upside/100/5, 4),
                            "assumptions": {
                                **self._default_assumptions(),
                                "pe_used": industry_avg_pe,
                                "current_pe": round(pe_ratio, 2),
                                "eps": round(eps, 2)
                            },
                            "explanation": f"Using P/E valuation as FCF is negative. EPS: ${eps:.2f}, Target P/E: {industry_avg_pe}"
                        }
                    else:
                        LOGGER.warning(f"P/E ratio out of range: {pe_ratio:.2f}")
                else:
                    LOGGER.warning(f"Cannot compute PE: eps={eps}, current_price={current_price}")

                # Fallback 2: Price/Book Valuation
                book_value_per_share = (
                    metrics.get("bookValuePerShareTTM") or
                    metrics.get("bookValuePerShare") or
                    metrics.get("book_value_per_share")
                )
                LOGGER.info(f"Trying P/B fallback - Book Value: {book_value_per_share}")

                if book_value_per_share and book_value_per_share > 0:
                    target_pb = 1.5
                    fair_value = book_value_per_share * target_pb
                    upside = ((fair_value - current_price) / current_price) * 100 if current_price > 0 else 0

                    LOGGER.info(f"P/B Fallback successful - Book Value: {book_value_per_share}, Fair Value: {fair_value}, Upside: {upside}%")
                    return {
                        "method": "P/B Fallback (No FCF/Earnings)",
                        "fair_value_per_share": round(fair_value, 2),
                        "upside_percent": round(upside, 2),
                        "implied_return_cagr": 0,
                        "assumptions": {
                            **self._default_assumptions(),
                            "pb_used": target_pb,
                            "book_value": round(book_value_per_share, 2)
                        },
                        "explanation": f"Using Price-to-Book. Book Value: ${book_value_per_share:.2f}"
                    }

                # Final fallback: return zero with detailed explanation
                LOGGER.error("All valuation fallbacks failed - returning zero")
                return {
                    "method": "N/A (Negative FCF/Earnings)",
                    "fair_value_per_share": 0,
                    "upside_percent": 0,
                    "implied_return_cagr": 0,
                    "assumptions": self._default_assumptions(),
                    "explanation": "Cannot calculate valuation: negative FCF, earnings, and book value unavailable."
                }

            # Assumptions
            growth_rate = min(metrics.get("growth", {}).get("freeCashFlowGrowth", 0.10), 0.15) # Cap at 15%
            if growth_rate < 0.02: growth_rate = 0.05 # Min 5%

            discount_rate = 0.10 # Simplified WACC
            terminal_growth = 0.03
            years = 5

            LOGGER.info(f"DCF calculation with positive FCF: {fcf_per_share}, growth: {growth_rate}, discount: {discount_rate}")

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

            LOGGER.info(f"DCF result - Fair Value: ${fair_value:.2f}, Upside: {upside:.2f}%")

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
            # Fetch historical ratios (last 5 years annual)
            hist_ratios_obj = self.obb_client.get_multiples(ticker, limit=5, period="annual")
            hist_ratios = self._extract_list(hist_ratios_obj)
            
            if not hist_ratios:
                return {"method": "N/A", "fair_value": 0, "explanation": "No historical EV multiples"}

            last_entries = hist_ratios[:5]
            ev_sales_samples = [
                x.get("enterpriseValueOverRevenue") or x.get("enterprise_value_to_revenue")
                for x in last_entries
                if x.get("enterpriseValueOverRevenue") or x.get("enterprise_value_to_revenue")
            ]
            ev_ebitda_samples = [
                x.get("enterpriseValueOverEBITDA") or x.get("enterprise_value_to_ebitda")
                for x in last_entries
                if x.get("enterpriseValueOverEBITDA") or x.get("enterprise_value_to_ebitda")
            ]

            ev_sales_median = statistics.median(ev_sales_samples) if ev_sales_samples else None
            ev_ebitda_median = statistics.median(ev_ebitda_samples) if ev_ebitda_samples else None

            shares = current_metrics.get("sharesOutstanding") or current_metrics.get("weightedAverageShsOutDil")
            if not shares:
                market_cap = current_metrics.get("marketCap")
                if market_cap and current_price:
                    shares = market_cap / current_price

            revenue = current_metrics.get("revenueTTM") or current_metrics.get("revenue")
            if not revenue and shares and current_metrics.get("revenuePerShareTTM"):
                revenue = current_metrics["revenuePerShareTTM"] * shares

            ebitda = current_metrics.get("ebitdaTTM") or current_metrics.get("ebitda")
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

    def _extract_list(self, obb_object: Any) -> list:
        """Helper to extract a list of dicts from an OBBject."""
        if not obb_object:
            return []
        try:
            if hasattr(obb_object, 'to_df'):
                df = obb_object.to_df()
                if not df.empty:
                    return df.replace({float('nan'): None}).to_dict(orient='records')
            if hasattr(obb_object, 'results'):
                if isinstance(obb_object.results, list):
                    return [
                        res.model_dump() if hasattr(res, 'model_dump') else (res if isinstance(res, dict) else res.__dict__)
                        for res in obb_object.results
                    ]
        except Exception as e:
            LOGGER.warning(f"Error extracting list from OBBject: {e}")
        return []

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
