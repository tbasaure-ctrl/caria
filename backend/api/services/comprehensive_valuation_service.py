"""
Comprehensive Valuation Service

Integrates three valuation methodologies:
1. Reverse DCF - Implied growth rate market is pricing in
2. Multiples Valuation - Fair value based on historical PE, PB, PS
3. Monte Carlo Simulation - Probabilistic 2-year price forecast

Uses FundamentalsCacheService for data (expanding universe).
"""

import logging
from typing import Dict, Any
from datetime import datetime
import asyncio

from api.services.simple_valuation import SimpleValuationService
from api.services.monte_carlo_service import MonteCarloService, get_monte_carlo_service
from api.services.fundamentals_cache_service import get_fundamentals_cache_service
from api.services.openbb_client import openbb_client

LOGGER = logging.getLogger("caria.services.comprehensive_valuation")


class ComprehensiveValuationService:
    """
    Unified valuation service combining Reverse DCF, Multiples, and Monte Carlo.
    Fetches data on-demand and caches for future use (expanding universe).
    """
    
    def __init__(self):
        self.simple_val = SimpleValuationService()
        self.monte_carlo = get_monte_carlo_service()
        self.cache = get_fundamentals_cache_service()
    
    async def get_full_valuation(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive valuation for any ticker.
        
        Process:
        1. Get fundamentals from cache or fetch from OpenBB
        2. Get current price
        3. Run all three valuation methods
        4. Return combined results with executive summary
        
        Returns:
            {
                "ticker": str,
                "current_price": float,
                "data_source": "static_cache" | "dynamic_cache" | "realtime_fetched",
                "reverse_dcf": {...},
                "multiples_valuation": {...},
                "monte_carlo": {...},
                "summary": str
            }
        """
        ticker = ticker.upper()
        LOGGER.info(f"Running comprehensive valuation for {ticker}")
        
        try:
            # 1. Get fundamentals (from cache or fetch)
            fund_result = self.cache.get_fundamentals(ticker)
            fundamentals = fund_result["data"]
            data_source = fund_result["source"]
            fetched_at = fund_result.get("fetched_at")
            
            LOGGER.info(f"Fundamentals source for {ticker}: {data_source}")
            
            # 2. Get current price
            price_data = openbb_client.get_current_price(ticker)
            current_price = price_data.get("price")
            
            if not current_price:
                raise ValueError(f"Could not get current price for {ticker}")
            
            # 3. Run all valuation methods
            
            # 3a. Reverse DCF
            try:
                assumptions = self.simple_val._default_assumptions()
                reverse_dcf = self.simple_val._calculate_reverse_dcf(
                    fundamentals, current_price, assumptions
                )
                LOGGER.info(f"Reverse DCF completed for {ticker}")
            except Exception as e:
                LOGGER.warning(f"Reverse DCF failed for {ticker}: {e}")
                reverse_dcf = {"error": str(e), "implied_growth_rate": None}
            
            # 3b. Multiples Valuation
            try:
                multiples_val = self.simple_val._calculate_historical_multiples_valuation(
                    ticker, fundamentals, current_price
                )
                LOGGER.info(f"Multiples valuation completed for {ticker}")
            except Exception as e:
                LOGGER.warning(f"Multiples valuation failed for {ticker}: {e}")
                multiples_val = {"error": str(e)}
            
            # 3c. Monte Carlo Simulation (async)
            try:
                monte_carlo_result = await asyncio.to_thread(
                    self.monte_carlo.run_stock_forecast,
                    ticker,
                    horizon_years=2,
                    simulations=10000
                )
                LOGGER.info(f"Monte Carlo simulation completed for {ticker}")
            except Exception as e:
                LOGGER.warning(f"Monte Carlo simulation failed for {ticker}: {e}")
                monte_carlo_result = {"error": str(e)}
            
            # 4. Generate interpretations and summary
            reverse_dcf_interpretation = self._interpret_reverse_dcf(reverse_dcf)
            multiples_interpretation = self._interpret_multiples(multiples_val, current_price)
            monte_carlo_interpretation = self._interpret_monte_carlo(monte_carlo_result)
            
            executive_summary = self._generate_executive_summary(
                ticker,
                current_price,
                reverse_dcf,
                multiples_val,
                monte_carlo_result
            )
            
            # 5. Combine all results
            result = {
                "ticker": ticker,
                "company_name": fundamentals.get("company_name", ticker),
                "sector": fundamentals.get("sector"),
                "industry": fundamentals.get("industry"),
                "current_price": current_price,
                "data_source": data_source,
                "data_freshness": self._get_data_freshness(data_source, fetched_at),
                "timestamp": datetime.now().isoformat(),
                
                "reverse_dcf": {
                    "implied_growth_rate": reverse_dcf.get("implied_growth_rate"),
                    "interpretation": reverse_dcf_interpretation,
                    "assumptions": reverse_dcf.get("assumptions", {}),
                    "details": reverse_dcf
                },
                
                "multiples_valuation": {
                    "pe_based_fair_value": multiples_val.get("pe_fair_value"),
                    "pb_based_fair_value": multiples_val.get("pb_fair_value"),
                    "ps_based_fair_value": multiples_val.get("ps_fair_value"),
                    "average_fair_value": multiples_val.get("avg_fair_value"),
                    "upside_downside_pct": multiples_val.get("upside_percent"),
                    "interpretation": multiples_interpretation,
                    "details": multiples_val
                },
                
                "monte_carlo": {
                    "horizon_years": 2,
                    "simulations": 10000,
                    "percentiles": monte_carlo_result.get("percentiles", {}),
                    "expected_value": monte_carlo_result.get("expected_value"),
                    "probability_positive_return": monte_carlo_result.get("prob_positive"),
                    "interpretation": monte_carlo_interpretation,
                    "chart_data": monte_carlo_result.get("plotly_data"),
                    "histogram_data": monte_carlo_result.get("histogram_data")
                },
                
                "executive_summary": executive_summary,
                
                "metadata": {
                    "valuation_date": datetime.now().isoformat(),
                    "data_source": data_source,
                    "cache_universe_size": len(self.cache.get_all_cached_tickers())
                }
            }
            
            return result
            
        except Exception as e:
            LOGGER.error(f"Comprehensive valuation failed for {ticker}: {e}")
            raise
    
    def _interpret_reverse_dcf(self, reverse_dcf: Dict) -> str:
        """Generate human-readable interpretation of Reverse DCF results."""
        growth = reverse_dcf.get("implied_growth_rate")
        
        if growth is None or "error" in reverse_dcf:
            return "Unable to calculate implied growth rate with available data."
        
        growth_pct = growth * 100
        
        if growth < 0:
            return f"⚠️ Market expects negative growth ({growth_pct:.1f}%). Stock may be overvalued or facing headwinds."
        elif growth < 0.03:
            return f"📉 Very conservative growth expectations ({growth_pct:.1f}%). Market pricing in minimal growth."
        elif growth < 0.08:
            return f"📊 Moderate growth expectations ({growth_pct:.1f}%). Reasonable pricing for mature company."
        elif growth < 0.15:
            return f"📈 Healthy growth expectations ({growth_pct:.1f}%). Market optimistic about prospects."
        elif growth < 0.25:
            return f"🚀 High growth expectations ({growth_pct:.1f}%). Strong confidence in future performance."
        else:
            return f"⚠️ Very aggressive growth expectations ({growth_pct:.1f}%). May be overvalued - difficult to sustain."
    
    def _interpret_multiples(self, multiples_val: Dict, current_price: float) -> str:
        """Generate human-readable interpretation of Multiples valuation."""
        if "error" in multiples_val:
            return "Unable to calculate multiples-based valuation with available data."
        
        avg_fair = multiples_val.get("avg_fair_value")
        upside = multiples_val.get("upside_percent")
        
        if avg_fair is None or upside is None:
            return "Insufficient historical data for multiples valuation."
        
        if upside > 30:
            return f"💰 Significantly undervalued ({upside:.1f}% upside). Trading well below historical averages."
        elif upside > 15:
            return f"✅ Moderately undervalued ({upside:.1f}% upside). Attractive entry point vs historical norms."
        elif upside > -5:
            return f"📊 Fairly valued. Trading close to historical average multiples."
        elif upside > -20:
            return f"⚠️ Slightly overvalued ({abs(upside):.1f}% downside). Above historical averages."
        else:
            return f"🚨 Significantly overvalued ({abs(upside):.1f}% downside). Trading well above historical norms."
    
    def _interpret_monte_carlo(self, mc_result: Dict) -> str:
        """Generate human-readable interpretation of Monte Carlo simulation."""
        if "error" in mc_result:
            return "Unable to run Monte Carlo simulation with available data."
        
        prob_positive = mc_result.get("prob_positive", 0)
        percentiles = mc_result.get("percentiles", {})
        p50 = percentiles.get("50th", 0)
        
        confidence = ""
        if prob_positive > 0.70:
            confidence = "High probability of positive returns"
        elif prob_positive > 0.55:
            confidence = "Moderate probability of positive returns"
        else:
            confidence = "Low probability of positive returns"
        
        return f"{confidence} ({prob_positive*100:.0f}%). Median 2-year outcome: ${p50:.2f}"
    
    def _generate_executive_summary(
        self,
        ticker: str,
        current_price: float,
        reverse_dcf: Dict,
        multiples_val: Dict,
        monte_carlo: Dict
    ) -> str:
        """Generate executive summary combining all three methodologies."""
        
        summary_parts = [f"**{ticker}** is trading at **${current_price:,.2f}**."]
        
        # Reverse DCF insight
        implied_growth = reverse_dcf.get("implied_growth_rate")
        if implied_growth is not None:
            summary_parts.append(
                f"The market is pricing in **{implied_growth*100:.1f}% annual growth**, "
                f"according to reverse DCF analysis."
            )
        
        # Multiples insight
        upside = multiples_val.get("upside_percent")
        if upside is not None:
            if upside > 0:
                summary_parts.append(
                    f"Historical multiples suggest **{upside:.1f}% upside potential**."
                )
            else:
                summary_parts.append(
                    f"Historical multiples indicate the stock is **{abs(upside):.1f}% above fair value**."
                )
        
        # Monte Carlo insight
        prob_positive = monte_carlo.get("prob_positive")
        if prob_positive is not None:
            summary_parts.append(
                f"Monte Carlo simulation shows a **{prob_positive*100:.0f}% probability** "
                f"of positive returns over the next 2 years."
            )
        
        # Overall verdict
        signals = []
        if upside and upside > 15:
            signals.append("undervalued")
        if prob_positive and prob_positive > 0.65:
            signals.append("favorable outlook")
        if implied_growth and 0.05 < implied_growth < 0.20:
            signals.append("reasonable growth expectations")
        
        if len(signals) >= 2:
            verdict = "✅ **Multiple positive signals** suggest this could be an attractive opportunity."
        elif len(signals) == 1:
            verdict = "📊 **Mixed signals** - some positive indicators but proceed with caution."
        else:
            verdict = "⚠️ **Limited positive signals** - careful analysis recommended before investing."
        
        summary_parts.append(verdict)
        
        return " ".join(summary_parts)
    
    def _get_data_freshness(self, source: str, fetched_at) -> str:
        """Get human-readable data freshness indicator."""
        if source == "static_cache":
            return "Historical snapshot"
        elif source == "realtime_fetched":
            return "Real-time (just fetched)"
        elif source == "dynamic_cache":
            if fetched_at:
                age = datetime.now() - fetched_at
                if age.total_seconds() < 3600:
                    return f"Recent ({age.seconds // 60} minutes old)"
                elif age.total_seconds() < 86400:
                    return f"Today ({age.seconds // 3600} hours old)"
                else:
                    return f"{age.days} days old"
            return "Cached"
        else:
            return "Unknown"


# Singleton instance
_comprehensive_val_service = None

def get_comprehensive_valuation_service() -> ComprehensiveValuationService:
    """Get singleton instance of ComprehensiveValuationService."""
    global _comprehensive_val_service
    if _comprehensive_val_service is None:
        _comprehensive_val_service = ComprehensiveValuationService()
    return _comprehensive_val_service
