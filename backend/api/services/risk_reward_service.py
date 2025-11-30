"""
Risk-Reward Engine Service

Calculates Bear/Base/Bull scenarios, Risk-Reward Ratio (RRR), and Expected Value (EV)
for stock investments with educational explanations.
"""

import logging
import math
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from caria.ingestion.clients.fmp_client import FMPClient

LOGGER = logging.getLogger("caria.api.risk_reward")


class RiskRewardService:
    """Service for calculating risk-reward scenarios and metrics."""

    def __init__(self) -> None:
        self.fmp = FMPClient()

    def analyze(
        self,
        ticker: str,
        horizon_months: int = 24,
        probabilities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze risk-reward for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            horizon_months: Time horizon in months (12, 24, or 36)
            probabilities: Optional dict with bear, base, bull probabilities (must sum to 1.0)
                          If None, defaults to equal probabilities (0.33 each)
        
        Returns:
            Dict with scenarios, metrics, and explanations
        """
        ticker = ticker.upper()
        horizon_years = horizon_months / 12.0

        # Fetch data
        try:
            financials = self._fetch_financials(ticker)
            prices_raw = self.fmp.get_price_history(ticker, start_date=None, end_date=None)
            
            # Convert prices to list of dicts
            prices = []
            if isinstance(prices_raw, list):
                prices = prices_raw
            elif hasattr(prices_raw, '__iter__'):
                for item in prices_raw:
                    if isinstance(item, dict):
                        prices.append(item)
                    elif hasattr(item, '__dict__'):
                        prices.append(item.__dict__)
            
            quote = self.fmp.get_realtime_price(ticker)
            
            if not financials or not prices or not quote:
                raise ValueError(f"Insufficient data for {ticker}")
                
        except Exception as e:
            LOGGER.error(f"Failed to fetch data for {ticker}: {e}")
            raise RuntimeError(f"Data fetch failed for {ticker}: {str(e)}")

        current_price = quote.get("price") or quote.get("last") or 0.0
        if not current_price or current_price <= 0:
            raise ValueError(f"Invalid current price for {ticker}")

        # Calculate fundamentals metrics
        metrics = self._calculate_fundamentals_metrics(financials, quote)
        
        # Get volatility metrics
        volatility_metrics = self._get_volatility_metrics(prices, current_price)
        
        # Build scenarios
        scenarios = self._build_scenarios(
            ticker=ticker,
            current_price=current_price,
            metrics=metrics,
            volatility_metrics=volatility_metrics,
            horizon_years=horizon_years,
        )
        
        # Set probabilities (default to equal if not provided)
        if probabilities is None:
            probabilities = {"bear": 0.33, "base": 0.34, "bull": 0.33}
        else:
            # Validate probabilities sum to 1.0
            total = sum(probabilities.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Probabilities must sum to 1.0, got {total}")
        
        # Compute risk-reward metrics
        risk_reward_metrics = self._compute_risk_reward(scenarios, probabilities)
        
        # Generate explanations
        explanations = self._generate_explanations(
            scenarios, probabilities, risk_reward_metrics, ticker
        )
        
        return {
            "ticker": ticker,
            "horizon_months": horizon_months,
            "current_price": round(current_price, 2),
            "scenarios": scenarios,
            "metrics": risk_reward_metrics,
            "explanations": explanations,
            "volatility_metrics": volatility_metrics,
        }

    def _fetch_financials(self, ticker: str) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch annual financial statements."""
        return {
            "income": self.fmp.get_income_statement(ticker, period="annual"),
            "balance": self.fmp.get_balance_sheet(ticker, period="annual"),
            "cash": self.fmp.get_cash_flow(ticker, period="annual"),
        }

    def _calculate_fundamentals_metrics(
        self, financials: Dict[str, List[Dict]], quote: Dict
    ) -> Dict[str, float]:
        """Calculate fundamental metrics needed for scenario projections."""
        
        def get_val(stmts, key, idx=0):
            if stmts and len(stmts) > idx:
                val = stmts[idx].get(key, 0)
                return float(val) if val else 0.0
            return 0.0

        inc = financials["income"]
        bal = financials["balance"]
        cash = financials["cash"]
        
        if len(inc) < 1:
            return {}
        
        # Latest year metrics
        revenue = get_val(inc, "revenue", 0)
        net_income = get_val(inc, "netIncome", 0)
        op_income = get_val(inc, "operatingIncome", 0)
        op_cash_flow = get_val(cash, "operatingCashFlow", 0)
        capex = abs(get_val(cash, "capitalExpenditure", 0))  # Make positive
        fcf = op_cash_flow - capex
        
        total_debt = get_val(bal, "totalDebt", 0)
        cash_equiv = get_val(bal, "cashAndCashEquivalents", 0)
        total_equity = get_val(bal, "totalStockholdersEquity", 0)
        
        # Market cap and enterprise value
        market_cap = quote.get("marketCap") or 0.0
        shares_outstanding = quote.get("sharesOutstanding") or 0.0
        if not market_cap and shares_outstanding:
            market_cap = quote.get("price", 0) * shares_outstanding
        
        enterprise_value = market_cap + total_debt - cash_equiv
        
        # Margins
        net_margin = (net_income / revenue) if revenue > 0 else 0.0
        op_margin = (op_income / revenue) if revenue > 0 else 0.0
        fcf_margin = (fcf / revenue) if revenue > 0 else 0.0
        
        # Growth rates (3-year CAGR if available)
        revenue_cagr = 0.0
        if len(inc) >= 4:
            rev_t = get_val(inc, "revenue", 0)
            rev_t3 = get_val(inc, "revenue", 3)
            if rev_t3 > 0 and rev_t > 0:
                revenue_cagr = (rev_t / rev_t3) ** (1.0 / 3.0) - 1.0
        
        # Valuation multiples
        pe_ratio = quote.get("pe") or 0.0
        ev_sales = (enterprise_value / revenue) if revenue > 0 else 0.0
        fcf_yield = (fcf / enterprise_value) if enterprise_value > 0 else 0.0
        
        # Historical min/max for scenario bounds
        revenue_growth_min = -0.10  # Default -10%
        revenue_growth_max = 0.20   # Default +20%
        
        if len(inc) >= 4:
            # Calculate historical growth rates
            growth_rates = []
            for i in range(min(3, len(inc) - 1)):
                rev_curr = get_val(inc, "revenue", i)
                rev_prev = get_val(inc, "revenue", i + 1)
                if rev_prev > 0:
                    growth = (rev_curr / rev_prev) - 1.0
                    growth_rates.append(growth)
            
            if growth_rates:
                revenue_growth_min = min(growth_rates) - 0.05  # Add buffer
                revenue_growth_max = max(growth_rates) + 0.10  # Add buffer
        
        return {
            "revenue": revenue,
            "net_income": net_income,
            "fcf": fcf,
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "shares_outstanding": shares_outstanding,
            "net_margin": net_margin,
            "op_margin": op_margin,
            "fcf_margin": fcf_margin,
            "revenue_cagr": revenue_cagr,
            "pe_ratio": pe_ratio,
            "ev_sales": ev_sales,
            "fcf_yield": fcf_yield,
            "revenue_growth_min": revenue_growth_min,
            "revenue_growth_max": revenue_growth_max,
        }

    def _get_volatility_metrics(
        self, prices: List[Dict], current_price: float
    ) -> Dict[str, float]:
        """Calculate volatility and drawdown metrics."""
        if not prices or len(prices) < 30:
            return {
                "volatility": 0.20,  # Default 20% annualized
                "max_drawdown": 0.30,  # Default 30%
            }
        
        # Sort prices by date (oldest first)
        sorted_prices = sorted(prices, key=lambda x: x.get('date', ''))
        
        # Extract closes
        closes = []
        for p in sorted_prices:
            close = p.get('close') or p.get('Close') or 0.0
            if close > 0:
                closes.append(float(close))
        
        if len(closes) < 30:
            return {
                "volatility": 0.20,
                "max_drawdown": 0.30,
            }
        
        # Calculate daily returns
        closes_series = pd.Series(closes)
        daily_returns = closes_series.pct_change().dropna()
        
        # Annualized volatility (252 trading days)
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0.20
        
        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.30
        
        return {
            "volatility": max(0.10, min(1.0, volatility)),  # Clamp between 10% and 100%
            "max_drawdown": max(0.10, min(0.90, max_drawdown)),  # Clamp between 10% and 90%
        }

    def _build_scenarios(
        self,
        ticker: str,
        current_price: float,
        metrics: Dict[str, float],
        volatility_metrics: Dict[str, float],
        horizon_years: float,
    ) -> Dict[str, Dict[str, float]]:
        """
        Build Bear, Base, and Bull scenarios.
        
        Returns:
            Dict with bear, base, bull scenarios, each containing:
            - price: Future price target
            - return_pct: Return percentage (as decimal, e.g., 0.20 for 20%)
        """
        revenue = metrics.get("revenue", 0)
        fcf = metrics.get("fcf", 0)
        shares_outstanding = metrics.get("shares_outstanding", 0)
        net_margin = metrics.get("net_margin", 0)
        fcf_margin = metrics.get("fcf_margin", 0)
        revenue_cagr = metrics.get("revenue_cagr", 0)
        pe_ratio = metrics.get("pe_ratio", 0)
        ev_sales = metrics.get("ev_sales", 0)
        fcf_yield = metrics.get("fcf_yield", 0)
        
        revenue_growth_min = metrics.get("revenue_growth_min", -0.10)
        revenue_growth_max = metrics.get("revenue_growth_max", 0.20)
        
        volatility = volatility_metrics.get("volatility", 0.20)
        max_drawdown = volatility_metrics.get("max_drawdown", 0.30)
        
        # Default assumptions if data missing
        if revenue <= 0:
            revenue = 1e9  # $1B default
        if shares_outstanding <= 0:
            shares_outstanding = 1e8  # 100M shares default
        if revenue_cagr == 0:
            revenue_cagr = 0.05  # 5% default growth
        
        # BEAR CASE
        bear_revenue_growth = max(-0.20, revenue_growth_min - 0.05)  # Cap at -20%
        bear_revenue_future = revenue * ((1 + bear_revenue_growth) ** horizon_years)
        bear_margin = max(0.01, net_margin * 0.75)  # Compress margin by 25%
        bear_fcf_margin = max(0.01, fcf_margin * 0.75)
        bear_fcf_future = bear_revenue_future * bear_fcf_margin
        
        # Bear multiple: contract by 20-40% or use lower bound
        if pe_ratio > 0:
            bear_pe = pe_ratio * 0.65  # Contract by 35%
        else:
            bear_pe = 10.0  # Conservative default
        
        # Use FCF yield approach if available, else PE
        if fcf_yield > 0:
            bear_fcf_yield = fcf_yield * 1.5  # Higher yield = lower valuation
            bear_market_cap = bear_fcf_future / bear_fcf_yield if bear_fcf_yield > 0 else bear_fcf_future * 15
        else:
            bear_net_income_future = bear_revenue_future * bear_margin
            bear_market_cap = bear_net_income_future * bear_pe
        
        bear_price = bear_market_cap / shares_outstanding if shares_outstanding > 0 else current_price * 0.70
        bear_return = (bear_price - current_price) / current_price
        
        # Ensure bear case is negative
        if bear_return > -0.05:  # If not negative enough, use volatility-based
            bear_return = -max_drawdown * 0.8  # Use 80% of max drawdown
            bear_price = current_price * (1 + bear_return)
        
        # BASE CASE
        base_revenue_growth = revenue_cagr  # Continue current trajectory
        base_revenue_future = revenue * ((1 + base_revenue_growth) ** horizon_years)
        base_margin = net_margin  # Maintain current margin
        base_fcf_margin = fcf_margin
        base_fcf_future = base_revenue_future * base_fcf_margin
        
        if pe_ratio > 0:
            base_pe = pe_ratio  # Maintain current multiple
        else:
            base_pe = 15.0  # Market average
        
        if fcf_yield > 0:
            base_fcf_yield = fcf_yield  # Maintain current yield
            base_market_cap = base_fcf_future / base_fcf_yield if base_fcf_yield > 0 else base_fcf_future * 20
        else:
            base_net_income_future = base_revenue_future * base_margin
            base_market_cap = base_net_income_future * base_pe
        
        base_price = base_market_cap / shares_outstanding if shares_outstanding > 0 else current_price * 1.05
        base_return = (base_price - current_price) / current_price
        
        # BULL CASE
        bull_revenue_growth = min(0.50, revenue_growth_max + 0.05)  # Cap at 50%
        bull_revenue_future = revenue * ((1 + bull_revenue_growth) ** horizon_years)
        bull_margin = min(0.50, net_margin * 1.15)  # Expand margin by 15%
        bull_fcf_margin = min(0.50, fcf_margin * 1.15)
        bull_fcf_future = bull_revenue_future * bull_fcf_margin
        
        if pe_ratio > 0:
            bull_pe = pe_ratio * 1.10  # Expand multiple by 10%
        else:
            bull_pe = 25.0  # Growth stock default
        
        if fcf_yield > 0:
            bull_fcf_yield = fcf_yield * 0.85  # Lower yield = higher valuation
            bull_market_cap = bull_fcf_future / bull_fcf_yield if bull_fcf_yield > 0 else bull_fcf_future * 30
        else:
            bull_net_income_future = bull_revenue_future * bull_margin
            bull_market_cap = bull_net_income_future * bull_pe
        
        bull_price = bull_market_cap / shares_outstanding if shares_outstanding > 0 else current_price * 1.50
        bull_return = (bull_price - current_price) / current_price
        
        # Ensure bull case is positive and meaningful
        if bull_return < 0.10:
            bull_return = 0.30  # Minimum 30% upside
            bull_price = current_price * (1 + bull_return)
        
        return {
            "bear": {
                "price": round(bear_price, 2),
                "return_pct": round(bear_return, 4),
            },
            "base": {
                "price": round(base_price, 2),
                "return_pct": round(base_return, 4),
            },
            "bull": {
                "price": round(bull_price, 2),
                "return_pct": round(bull_return, 4),
            },
        }

    def _compute_risk_reward(
        self, scenarios: Dict[str, Dict[str, float]], probabilities: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compute risk-reward metrics.
        
        Returns:
            Dict with upside, downside, rrr, expected_value
        """
        bear_return = scenarios["bear"]["return_pct"]
        base_return = scenarios["base"]["return_pct"]
        bull_return = scenarios["bull"]["return_pct"]
        
        p_bear = probabilities.get("bear", 0.33)
        p_base = probabilities.get("base", 0.34)
        p_bull = probabilities.get("bull", 0.33)
        
        # Upside = Bull return (positive)
        upside = max(0.0, bull_return)
        
        # Downside = Absolute value of bear return (always positive)
        downside = abs(min(0.0, bear_return))
        
        # Risk-Reward Ratio
        rrr = upside / downside if downside > 0 else 999.0  # Avoid division by zero
        
        # Expected Value
        expected_value = (
            p_bull * bull_return +
            p_base * base_return +
            p_bear * bear_return
        )
        
        return {
            "upside": round(upside, 4),
            "downside": round(downside, 4),
            "rrr": round(rrr, 2),
            "expected_value": round(expected_value, 4),
        }

    def _generate_explanations(
        self,
        scenarios: Dict[str, Dict[str, float]],
        probabilities: Dict[str, float],
        metrics: Dict[str, Any],
        ticker: str,
    ) -> Dict[str, str]:
        """Generate educational explanations."""
        
        bear_price = scenarios["bear"]["price"]
        bear_return = scenarios["bear"]["return_pct"]
        base_price = scenarios["base"]["price"]
        base_return = scenarios["base"]["return_pct"]
        bull_price = scenarios["bull"]["price"]
        bull_return = scenarios["bull"]["return_pct"]
        
        p_bear = probabilities.get("bear", 0.33)
        p_base = probabilities.get("base", 0.34)
        p_bull = probabilities.get("bull", 0.33)
        
        upside = metrics["upside"]
        downside = metrics["downside"]
        rrr = metrics["rrr"]
        ev = metrics["expected_value"]
        
        # Summary
        summary = (
            f"In your bull case, if things go well, {ticker} could return ~{bull_return*100:.0f}% "
            f"(reaching ${bull_price:.2f}). In your bear case, if things go wrong, you might lose "
            f"~{abs(bear_return)*100:.0f}% (falling to ${bear_price:.2f}). "
            f"The risk/reward ratio is about {rrr:.1f}:1, which means every 1 unit of potential loss "
            f"is matched by {rrr:.1f} units of potential gain."
        )
        
        # EV Breakdown
        ev_breakdown = (
            f"We estimate a {p_bull*100:.0f}% chance of +{bull_return*100:.0f}%, "
            f"a {p_base*100:.0f}% chance of {base_return*100:+.0f}%, "
            f"and a {p_bear*100:.0f}% chance of {bear_return*100:.0f}%. "
            f"Expected value = {p_bull:.2f}×{bull_return:.2f} + {p_base:.2f}×{base_return:.2f} + "
            f"{p_bear:.2f}×({bear_return:.2f}) = {p_bull*bull_return:.2f} + {p_base*base_return:.2f} + "
            f"{p_bear*bear_return:.2f} = {ev:.2f}, or {ev*100:.0f}%."
        )
        
        # Analogy
        if ev > 0 and rrr > 1.5:
            if p_bull < 0.4:
                analogy = (
                    f"This setup is similar to a poker hand where you win ~{p_bull*100:.0f}% of the time "
                    f"but the pot odds give you more than {rrr:.1f}:1 on your money. Even though you lose "
                    f"more often than you win, the wins are large enough to make the expected value positive."
                )
            else:
                analogy = (
                    f"This looks like a favorable poker hand with {p_bull*100:.0f}% win probability and "
                    f"{rrr:.1f}:1 pot odds. The expected value is positive, similar to a casino game where "
                    f"the player has a slight edge rather than the house."
                )
        elif ev < 0:
            analogy = (
                f"This setup is similar to a casino game where the house has the edge. Despite potentially "
                f"high upside ({upside*100:.0f}%), the low probability of the bull case ({p_bull*100:.0f}%) "
                f"and the risk of loss ({downside*100:.0f}%) result in negative expected value. "
                f"Like buying insurance when you're unlikely to need it, this might not be a rational bet."
            )
        else:
            analogy = (
                f"This is similar to a balanced poker hand with roughly equal probabilities. The risk-reward "
                f"ratio of {rrr:.1f}:1 suggests moderate asymmetry, and the expected value is {ev*100:+.0f}%. "
                f"A rational investor might consider this a reasonable opportunity, but position sizing "
                f"should reflect the uncertainty."
            )
        
        # Position Sizing Guidance
        if ev > 0.15 and rrr > 2.0:
            position_sizing = (
                f"A rational investor might choose a larger position size here, as the expected value is "
                f"strongly positive ({ev*100:.0f}%) and the risk-reward ratio is favorable ({rrr:.1f}:1). "
                f"However, the decision and exact position size are up to you based on your risk tolerance "
                f"and portfolio allocation."
            )
        elif ev > 0 and rrr > 1.0:
            position_sizing = (
                f"A rational investor might choose a moderate position size here. The expected value is "
                f"positive ({ev*100:.0f}%) but the risk-reward ratio ({rrr:.1f}:1) suggests moderate "
                f"asymmetry. Consider your overall portfolio allocation and risk tolerance."
            )
        elif ev < 0:
            position_sizing = (
                f"A rational investor might choose a smaller position or avoid this opportunity entirely, "
                f"as the expected value is negative ({ev*100:.0f}%). Despite potentially high upside, "
                f"the probabilities suggest this is not a favorable bet. The decision is ultimately yours."
            )
        else:
            position_sizing = (
                f"A rational investor might choose a smaller position size given the uncertainty. "
                f"The expected value is {ev*100:+.0f}% and the risk-reward ratio is {rrr:.1f}:1. "
                f"Consider how this fits into your overall portfolio strategy."
            )
        
        return {
            "summary": summary,
            "ev_breakdown": ev_breakdown,
            "analogy": analogy,
            "position_sizing": position_sizing,
        }

