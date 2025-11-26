"""
Enhanced Valuation Service

Provides robust intrinsic value calculation using multiple methodologies:
1. DCF (Discounted Cash Flow) - Improved with better FCF estimation
2. Multiples Valuation - Historical averages (P/E, P/B, P/S, EV/EBITDA, EV/Sales)
3. Graham Number - Conservative intrinsic value estimate
4. Asset-Based Valuation - Book value adjustments
5. Earnings Power Value (EPV) - Based on normalized earnings

Also provides enhanced Monte Carlo simulation with fundamental adjustments.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import statistics

from api.services.openbb_client import OpenBBClient

LOGGER = logging.getLogger("caria.services.enhanced_valuation")

class EnhancedValuationService:
    """
    Enhanced valuation service with multiple intrinsic value methods
    and improved Monte Carlo simulation.
    """
    
    def __init__(self):
        self.obb_client = OpenBBClient()
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    def get_intrinsic_value(self, ticker: str, current_price: float) -> Dict[str, Any]:
        """
        Calculate intrinsic value using multiple methods.
        Returns consensus value and individual method results.
        """
        ticker = ticker.upper()
        LOGGER.info(f"Calculating intrinsic value for {ticker} at ${current_price:.2f}")
        
        # Fetch comprehensive data
        data = self._fetch_comprehensive_data(ticker)
        if not data:
            return self._error_response(ticker, current_price, "Could not fetch financial data")
        
        results = {
            "ticker": ticker,
            "current_price": current_price,
            "currency": "USD",
            "timestamp": datetime.now().isoformat(),
            "methods": {}
        }
        
        # Method 1: Enhanced DCF
        try:
            dcf_result = self._calculate_enhanced_dcf(data, current_price)
            results["methods"]["dcf"] = dcf_result
        except Exception as e:
            LOGGER.warning(f"DCF calculation failed for {ticker}: {e}")
            results["methods"]["dcf"] = {"error": str(e), "fair_value": None}
        
        # Method 2: Multiples Valuation (Historical)
        try:
            multiples_result = self._calculate_multiples_valuation(data, current_price)
            results["methods"]["multiples"] = multiples_result
        except Exception as e:
            LOGGER.warning(f"Multiples valuation failed for {ticker}: {e}")
            results["methods"]["multiples"] = {"error": str(e), "fair_value": None}
        
        # Method 3: Graham Number
        try:
            graham_result = self._calculate_graham_number(data, current_price)
            results["methods"]["graham"] = graham_result
        except Exception as e:
            LOGGER.warning(f"Graham Number calculation failed for {ticker}: {e}")
            results["methods"]["graham"] = {"error": str(e), "fair_value": None}
        
        # Method 4: Earnings Power Value (EPV)
        try:
            epv_result = self._calculate_epv(data, current_price)
            results["methods"]["epv"] = epv_result
        except Exception as e:
            LOGGER.warning(f"EPV calculation failed for {ticker}: {e}")
            results["methods"]["epv"] = {"error": str(e), "fair_value": None}
        
        # Method 5: Asset-Based Valuation
        try:
            asset_result = self._calculate_asset_based(data, current_price)
            results["methods"]["asset_based"] = asset_result
        except Exception as e:
            LOGGER.warning(f"Asset-based valuation failed for {ticker}: {e}")
            results["methods"]["asset_based"] = {"error": str(e), "fair_value": None}
        
        # Calculate consensus intrinsic value
        fair_values = []
        method_weights = {
            "dcf": 0.30,
            "multiples": 0.25,
            "graham": 0.20,
            "epv": 0.15,
            "asset_based": 0.10
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method_name, method_result in results["methods"].items():
            if "error" not in method_result and method_result.get("fair_value") is not None:
                fv = method_result["fair_value"]
                if fv > 0:
                    fair_values.append(fv)
                    weight = method_weights.get(method_name, 0.1)
                    weighted_sum += fv * weight
                    total_weight += weight
        
        if fair_values:
            # Consensus value: weighted average
            consensus_value = weighted_sum / total_weight if total_weight > 0 else statistics.median(fair_values)
            
            # Also calculate simple median and mean
            median_value = statistics.median(fair_values)
            mean_value = statistics.mean(fair_values)
            
            results["intrinsic_value"] = {
                "consensus": round(consensus_value, 2),
                "median": round(median_value, 2),
                "mean": round(mean_value, 2),
                "min": round(min(fair_values), 2),
                "max": round(max(fair_values), 2),
                "methods_used": len(fair_values),
                "upside_percent": round(((consensus_value - current_price) / current_price) * 100, 2),
                "margin_of_safety": round(((consensus_value - current_price) / consensus_value) * 100, 2) if consensus_value > 0 else 0
            }
            
            # Interpretation
            upside = results["intrinsic_value"]["upside_percent"]
            if upside > 30:
                interpretation = f"Significantly undervalued ({upside:.1f}% upside). Strong buy signal."
            elif upside > 15:
                interpretation = f"Moderately undervalued ({upside:.1f}% upside). Attractive opportunity."
            elif upside > -5:
                interpretation = f"Fairly valued ({upside:+.1f}% upside). Trading near intrinsic value."
            elif upside > -20:
                interpretation = f"Slightly overvalued ({abs(upside):.1f}% downside). Proceed with caution."
            else:
                interpretation = f"Significantly overvalued ({abs(upside):.1f}% downside). Consider avoiding."
            
            results["intrinsic_value"]["interpretation"] = interpretation
        else:
            results["intrinsic_value"] = {
                "consensus": None,
                "error": "Insufficient data to calculate intrinsic value",
                "upside_percent": 0
            }
        
        LOGGER.info(f"✅ Intrinsic value calculation complete for {ticker}: ${results['intrinsic_value'].get('consensus', 'N/A')}")
        return results
    
    def _fetch_comprehensive_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive financial data with multiple fallbacks."""
        cache_key = f"{ticker}_comprehensive"
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self._cache_ttl):
                return cached_data
        
        try:
            # Get comprehensive data from OpenBB
            data = self.obb_client.get_ticker_data(ticker)
            
            if not data:
                LOGGER.warning(f"No data returned from OpenBB for {ticker}")
                return None
            
            # Extract key metrics
            key_metrics = data.get("key_metrics", {})
            if isinstance(key_metrics, list) and key_metrics:
                key_metrics = key_metrics[0]
            
            # Extract financials
            financials = data.get("financials", {})
            income_statement = financials.get("income_statement", [])
            cash_flow = financials.get("cash_flow", [])
            
            # Get latest income statement
            latest_income = income_statement[0] if income_statement else {}
            
            # Get latest cash flow
            latest_cf = cash_flow[0] if cash_flow else {}
            
            # Get multiples history
            multiples_history = data.get("multiples", [])
            
            # Get price history for volatility calculation
            price_history = data.get("price_history", [])
            
            # Extract current price
            current_price = data.get("latest_price", 0)
            if not current_price and price_history:
                current_price = price_history[-1].get("close", 0)
            
            # Compile comprehensive data
            comprehensive = {
                "ticker": ticker,
                "current_price": current_price,
                "key_metrics": key_metrics,
                "income_statement": latest_income,
                "cash_flow": latest_cf,
                "multiples_history": multiples_history,
                "price_history": price_history,
                "profile": data.get("profile", {})
            }
            
            # Cache the result
            self._cache[cache_key] = (comprehensive, datetime.now())
            
            return comprehensive
            
        except Exception as e:
            LOGGER.error(f"Error fetching comprehensive data for {ticker}: {e}")
            return None
    
    def _calculate_enhanced_dcf(self, data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Enhanced DCF calculation with better FCF estimation."""
        key_metrics = data.get("key_metrics", {})
        income = data.get("income_statement", {})
        cash_flow = data.get("cash_flow", {})
        
        # Try multiple sources for FCF per share
        fcf_per_share = (
            key_metrics.get("freeCashFlowPerShareTTM") or
            key_metrics.get("freeCashFlowPerShare") or
            key_metrics.get("fcfPerShare") or
            0
        )
        
        # If no FCF per share, calculate from total FCF and shares
        if not fcf_per_share or fcf_per_share <= 0:
            total_fcf = (
                cash_flow.get("freeCashFlow") or
                cash_flow.get("operatingCashFlow") or
                key_metrics.get("freeCashFlow") or
                0
            )
            shares = (
                key_metrics.get("sharesOutstanding") or
                key_metrics.get("weightedAverageSharesOutstanding") or
                0
            )
            if total_fcf > 0 and shares > 0:
                fcf_per_share = total_fcf / shares
        
        # If still no FCF, estimate from earnings
        if not fcf_per_share or fcf_per_share <= 0:
            eps = (
                key_metrics.get("netIncomePerShareTTM") or
                key_metrics.get("netIncomePerShare") or
                key_metrics.get("eps") or
                0
            )
            if eps > 0:
                # Estimate FCF as 80% of earnings (conservative)
                fcf_per_share = eps * 0.80
        
        if fcf_per_share <= 0:
            return {
                "method": "DCF",
                "fair_value": None,
                "error": "Insufficient FCF data",
                "explanation": "Cannot calculate DCF without free cash flow data"
            }
        
        # Estimate growth rate from historical data or use conservative default
        growth_rate = 0.05  # Default 5%
        
        # Try to get growth from metrics
        if "growth" in data:
            growth_data = data.get("growth", {})
            fcf_growth = growth_data.get("freeCashFlowGrowth") or growth_data.get("fcfGrowth")
            if fcf_growth:
                growth_rate = min(max(float(fcf_growth), 0.02), 0.15)  # Cap between 2% and 15%
        
        # Use sector/industry averages if available
        # For tech stocks, use higher growth; for mature companies, lower
        sector = data.get("profile", {}).get("sector", "").lower()
        if "technology" in sector or "tech" in sector:
            growth_rate = max(growth_rate, 0.08)  # At least 8% for tech
        elif "utilities" in sector or "consumer staples" in sector:
            growth_rate = min(growth_rate, 0.06)  # Cap at 6% for defensive
        
        # DCF assumptions
        discount_rate = 0.10  # 10% WACC (can be adjusted)
        terminal_growth = 0.03  # 3% terminal growth
        high_growth_years = 5
        fade_years = 5
        
        # Calculate DCF
        future_cash_flows = []
        for year in range(1, high_growth_years + 1):
            fcf = fcf_per_share * ((1 + growth_rate) ** year)
            pv = fcf / ((1 + discount_rate) ** year)
            future_cash_flows.append(pv)
        
        # Terminal value
        terminal_fcf = fcf_per_share * ((1 + growth_rate) ** high_growth_years) * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        terminal_value_pv = terminal_value / ((1 + discount_rate) ** high_growth_years)
        
        fair_value = sum(future_cash_flows) + terminal_value_pv
        
        return {
            "method": "Enhanced DCF",
            "fair_value": round(fair_value, 2),
            "fcf_per_share": round(fcf_per_share, 2),
            "growth_rate": round(growth_rate, 4),
            "discount_rate": discount_rate,
            "terminal_growth": terminal_growth,
            "explanation": f"DCF based on FCF/share of ${fcf_per_share:.2f} growing at {growth_rate*100:.1f}% for {high_growth_years} years"
        }
    
    def _calculate_multiples_valuation(self, data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Calculate fair value using historical multiples."""
        multiples_history = data.get("multiples_history", [])
        key_metrics = data.get("key_metrics", {})
        
        if not multiples_history:
            return {
                "method": "Multiples",
                "fair_value": None,
                "error": "No historical multiples data",
                "explanation": "Cannot calculate multiples valuation without historical data"
            }
        
        # Extract historical multiples (last 5 years)
        pe_ratios = []
        pb_ratios = []
        ps_ratios = []
        ev_ebitda = []
        ev_sales = []
        
        for entry in multiples_history[:5]:
            pe = entry.get("peRatio") or entry.get("priceEarningsRatio")
            pb = entry.get("priceToBookRatio") or entry.get("pbRatio")
            ps = entry.get("priceToSalesRatio") or entry.get("psRatio")
            ev_e = entry.get("enterpriseValueMultiple") or entry.get("evEbitda")
            ev_s = entry.get("enterpriseValueOverRevenue") or entry.get("evSales")
            
            if pe and pe > 0:
                pe_ratios.append(pe)
            if pb and pb > 0:
                pb_ratios.append(pb)
            if ps and ps > 0:
                ps_ratios.append(ps)
            if ev_e and ev_e > 0:
                ev_ebitda.append(ev_e)
            if ev_s and ev_s > 0:
                ev_sales.append(ev_s)
        
        # Calculate median multiples
        median_pe = statistics.median(pe_ratios) if pe_ratios else None
        median_pb = statistics.median(pb_ratios) if pb_ratios else None
        median_ps = statistics.median(ps_ratios) if ps_ratios else None
        
        # Get current fundamentals
        eps = (
            key_metrics.get("netIncomePerShareTTM") or
            key_metrics.get("netIncomePerShare") or
            key_metrics.get("eps") or
            0
        )
        book_value = (
            key_metrics.get("bookValuePerShareTTM") or
            key_metrics.get("bookValuePerShare") or
            0
        )
        revenue_per_share = (
            key_metrics.get("revenuePerShareTTM") or
            key_metrics.get("revenuePerShare") or
            0
        )
        
        fair_values = []
        
        # P/E valuation
        if median_pe and eps > 0:
            fv_pe = eps * median_pe
            fair_values.append(fv_pe)
        
        # P/B valuation
        if median_pb and book_value > 0:
            fv_pb = book_value * median_pb
            fair_values.append(fv_pb)
        
        # P/S valuation
        if median_ps and revenue_per_share > 0:
            fv_ps = revenue_per_share * median_ps
            fair_values.append(fv_ps)
        
        if not fair_values:
            return {
                "method": "Multiples",
                "fair_value": None,
                "error": "Insufficient data for multiples calculation",
                "explanation": "Need at least one of: EPS, Book Value, or Revenue per Share"
            }
        
        fair_value = statistics.median(fair_values)
        
        return {
            "method": "Historical Multiples",
            "fair_value": round(fair_value, 2),
            "median_pe": round(median_pe, 2) if median_pe else None,
            "median_pb": round(median_pb, 2) if median_pb else None,
            "median_ps": round(median_ps, 2) if median_ps else None,
            "explanation": f"Based on median historical multiples over last 5 years"
        }
    
    def _calculate_graham_number(self, data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Calculate Graham Number (conservative intrinsic value)."""
        key_metrics = data.get("key_metrics", {})
        
        eps = (
            key_metrics.get("netIncomePerShareTTM") or
            key_metrics.get("netIncomePerShare") or
            key_metrics.get("eps") or
            0
        )
        book_value = (
            key_metrics.get("bookValuePerShareTTM") or
            key_metrics.get("bookValuePerShare") or
            0
        )
        
        if eps <= 0 or book_value <= 0:
            return {
                "method": "Graham Number",
                "fair_value": None,
                "error": "Need positive EPS and Book Value",
                "explanation": "Graham Number requires both earnings and book value"
            }
        
        # Graham Number = sqrt(22.5 * EPS * Book Value)
        graham_number = np.sqrt(22.5 * eps * book_value)
        
        return {
            "method": "Graham Number",
            "fair_value": round(graham_number, 2),
            "eps": round(eps, 2),
            "book_value": round(book_value, 2),
            "explanation": f"Conservative intrinsic value: sqrt(22.5 × ${eps:.2f} × ${book_value:.2f})"
        }
    
    def _calculate_epv(self, data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Calculate Earnings Power Value (normalized earnings)."""
        key_metrics = data.get("key_metrics", {})
        income = data.get("income_statement", {})
        
        # Get normalized earnings (use TTM or average of last few years)
        eps = (
            key_metrics.get("netIncomePerShareTTM") or
            key_metrics.get("netIncomePerShare") or
            key_metrics.get("eps") or
            0
        )
        
        if eps <= 0:
            return {
                "method": "EPV",
                "fair_value": None,
                "error": "Need positive earnings",
                "explanation": "EPV requires positive earnings per share"
            }
        
        # Use a conservative discount rate (cost of equity)
        discount_rate = 0.12  # 12% for EPV
        
        # EPV = Normalized Earnings / Discount Rate
        epv = eps / discount_rate
        
        return {
            "method": "Earnings Power Value",
            "fair_value": round(epv, 2),
            "normalized_eps": round(eps, 2),
            "discount_rate": discount_rate,
            "explanation": f"EPV = ${eps:.2f} / {discount_rate*100:.0f}% = ${epv:.2f}"
        }
    
    def _calculate_asset_based(self, data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Calculate asset-based valuation."""
        key_metrics = data.get("key_metrics", {})
        
        book_value = (
            key_metrics.get("bookValuePerShareTTM") or
            key_metrics.get("bookValuePerShare") or
            0
        )
        
        if book_value <= 0:
            return {
                "method": "Asset-Based",
                "fair_value": None,
                "error": "Need positive book value",
                "explanation": "Asset-based valuation requires book value"
            }
        
        # For asset-based, use book value as conservative estimate
        # In practice, you might adjust for intangibles, etc.
        fair_value = book_value * 1.2  # 20% premium for going concern
        
        return {
            "method": "Asset-Based",
            "fair_value": round(fair_value, 2),
            "book_value": round(book_value, 2),
            "explanation": f"Based on book value per share of ${book_value:.2f} with going concern adjustment"
        }
    
    def _error_response(self, ticker: str, current_price: float, error_msg: str) -> Dict[str, Any]:
        """Return error response."""
        return {
            "ticker": ticker,
            "current_price": current_price,
            "error": error_msg,
            "intrinsic_value": {
                "consensus": None,
                "error": error_msg
            }
        }
