import logging
import math
import statistics
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from caria.ingestion.clients.fmp_client import FMPClient

LOGGER = logging.getLogger("caria.api.scoring")


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


class ScoringService:
    """Calculates Quality / Valuation / Momentum and composite Hidden Gem Score."""

    def __init__(self) -> None:
        self.fmp = FMPClient()

    def get_scores(self, ticker: str) -> Dict[str, Any]:
        """
        Get scores for a single ticker.
        Note: For true 'relative' scoring (Z-scores), we ideally need a universe.
        Here we will use robust normalization against fixed 'market standard' baselines 
        so that a single ticker score is still meaningful.
        """
        ticker = ticker.upper()
        
        # 1. Fetch Data
        try:
            financials = self._fetch_financials(ticker)
            prices = self.fmp.get_price_history(ticker) # Returns list of dicts
            quote = self.fmp.get_realtime_price(ticker)
        except Exception as e:
            LOGGER.error(f"Error fetching data for {ticker}: {e}")
            raise RuntimeError(f"Data fetch failed for {ticker}")

        if not financials or not prices or not quote:
             LOGGER.warning(f"Insufficient data for {ticker}")
             # Return empty/default score structure
             return self._empty_score(ticker)

        # 2. Calculate Raw Metrics
        metrics = self._calculate_metrics(ticker, financials, prices, quote)
        if not metrics:
             LOGGER.warning(f"Insufficient metrics calculated for {ticker}")
             return self._empty_score(ticker)
        
        # 3. Calculate Scores (Quality, Valuation, Momentum)
        # We use 'standard' Z-score-like normalization using fixed market means/stds
        # derived from typical market data (approximate) to allow single-ticker scoring.
        
        quality_score, quality_details = self._calculate_quality_score(metrics)
        valuation_score, valuation_details = self._calculate_valuation_score(metrics)
        momentum_score, momentum_details = self._calculate_momentum_score(metrics)
        
        # 4. Composite Score
        # Weights: 30% Quality, 30% Valuation, 25% Momentum, 15% Bonus/Penalty
        # We'll normalize components to 0-100 first.
        
        hidden_gem_score = (
            0.30 * quality_score +
            0.30 * valuation_score +
            0.25 * momentum_score
            # + 0.15 * bonus (simplified for now to re-normalize the above or add specific bonuses)
        )
        # Adjust denominator to normalize back to 100 if we don't have the bonus term explicitly
        hidden_gem_score = hidden_gem_score / 0.85 

        return {
            "ticker": ticker,
            "cScore": round(hidden_gem_score, 0), # Maintain API compatibility with 'cScore'
            "compositeScore": round(hidden_gem_score, 0), # Alias for frontend compatibility
            "hiddenGemScore": round(hidden_gem_score, 0),
            "qualityScore": round(quality_score, 0),
            "valuationScore": round(valuation_score, 0),
            "momentumScore": round(momentum_score, 0),
            "classification": self._classify_score(hidden_gem_score),
            "current_price": metrics.get("price"),
            "details": {
                "quality": quality_details,
                "valuation": valuation_details,
                "momentum": momentum_details,
                "metrics": metrics
            },
            "explanations": self._build_explanations(quality_details, valuation_details, momentum_details)
        }

    def _fetch_financials(self, ticker: str) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch annual financial statements."""
        return {
            "income": self.fmp.get_income_statement(ticker, period="annual"),
            "balance": self.fmp.get_balance_sheet(ticker, period="annual"),
            "cash": self.fmp.get_cash_flow(ticker, period="annual"),
        }

    def _calculate_metrics(
        self, 
        ticker: str, 
        financials: Dict[str, List[Dict]], 
        prices: List[Dict], 
        quote: Dict
    ) -> Dict[str, float]:
        """Compute raw fundamental and technical metrics."""
        
        # Helper to safely get latest or n-th value
        def get_val(stmts, key, idx=0):
            if stmts and len(stmts) > idx:
                return stmts[idx].get(key, 0)
            return 0

        inc = financials["income"]
        bal = financials["balance"]
        cash = financials["cash"]
        
        # Need at least 2 years for growth/ROIIC, 3 for CAGRs
        if len(inc) < 2 or len(bal) < 2:
            return {}

        # --- Fundamentals (Latest Year) ---
        op_income = get_val(inc, "operatingIncome")
        net_income = get_val(inc, "netIncome")
        revenue = get_val(inc, "revenue")
        op_cash_flow = get_val(cash, "operatingCashFlow")
        capex = get_val(cash, "capitalExpenditure")
        
        total_debt = get_val(bal, "totalDebt")
        cash_equiv = get_val(bal, "cashAndCashEquivalents")
        total_equity = get_val(bal, "totalStockholdersEquity")
        
        # --- Derived Fundamentals ---
        fcf = op_cash_flow - capex
        invested_capital = total_debt + total_equity - cash_equiv
        
        # Margins
        net_margin = (net_income / revenue) if revenue else 0
        op_margin = (op_income / revenue) if revenue else 0
        
        # ROIC
        roic = (op_income / invested_capital) if invested_capital and invested_capital != 0 else 0
        
        # ROIIC (Return on Incremental Invested Capital)
        # (OpIncome_t - OpIncome_t-1) / (IC_t - IC_t-1)
        prev_op_income = get_val(inc, "operatingIncome", 1)
        prev_invested_capital = (
            get_val(bal, "totalDebt", 1) + 
            get_val(bal, "totalStockholdersEquity", 1) - 
            get_val(bal, "cashAndCashEquivalents", 1)
        )
        
        delta_op = op_income - prev_op_income
        delta_ic = invested_capital - prev_invested_capital
        
        roiic = 0.0
        if delta_ic != 0:
            roiic = delta_op / delta_ic
        
        # CAGRs (3 Year)
        # (Value_t / Value_t-3)^(1/3) - 1
        rev_cagr_3y = 0.0
        fcf_cagr_3y = 0.0
        
        if len(inc) >= 4:
            rev_t = get_val(inc, "revenue", 0)
            rev_t3 = get_val(inc, "revenue", 3)
            if rev_t3 > 0 and rev_t > 0:
                rev_cagr_3y = (rev_t / rev_t3)**(1/3) - 1

        if len(cash) >= 4:
            fcf_t = get_val(cash, "operatingCashFlow", 0) - get_val(cash, "capitalExpenditure", 0)
            fcf_t3 = get_val(cash, "operatingCashFlow", 3) - get_val(cash, "capitalExpenditure", 3)
            # FCF can be negative, standard CAGR formula breaks. 
            # We'll use absolute growth if positive, else 0 for simplicity or handle signs.
            # Simplified: if both positive
            if fcf_t3 > 0 and fcf_t > 0:
                fcf_cagr_3y = (fcf_t / fcf_t3)**(1/3) - 1

        # Leverage
        debt_to_equity = (total_debt / total_equity) if total_equity else 0
        
        # --- Valuation ---
        market_cap = quote.get("marketCap") or (quote.get("price", 0) * quote.get("sharesOutstanding", 0))
        # fallback if marketCap missing in quote
        if not market_cap and len(inc) > 0:
             # Very rough fallback or fetch profile. 
             # For now, let's assume quote has it or we can't calculate EV based valuation.
             pass

        enterprise_value = market_cap + total_debt - cash_equiv
        
        fcf_yield = (fcf / enterprise_value) if enterprise_value and enterprise_value > 0 else 0
        
        # EV/EBITDA
        ebitda = get_val(inc, "ebitda")
        ev_ebitda = (enterprise_value / ebitda) if ebitda and ebitda > 0 else 0
        
        # --- Momentum ---
        # prices is list of dicts, assume sorted by date desc or asc? FMP historical is usually DESC (newest first).
        # Double check sort order. FMP 'historical-price-full' returns 'historical' list usually new to old.
        # Let's sort to be safe: Oldest first
        sorted_prices = sorted(prices, key=lambda x: x['date'])
        
        current_p = quote.get("price", 0)
        
        def get_ret(days):
            if len(sorted_prices) > days:
                old_p = sorted_prices[-days]['close']
                if old_p:
                    return (current_p - old_p) / old_p
            return 0.0

        ret_3m = get_ret(63)
        ret_12m = get_ret(252)
        
        # Volatility (Daily Returns Std Dev last 3 months)
        recent_closes = [p['close'] for p in sorted_prices[-63:]]
        daily_rets = pd.Series(recent_closes).pct_change().dropna()
        volatility = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 0 else 0
        
        return {
            "roic": roic,
            "roiic": roiic,
            "revenue_cagr_3y": rev_cagr_3y,
            "fcf_cagr_3y": fcf_cagr_3y,
            "net_margin": net_margin,
            "debt_to_equity": debt_to_equity,
            "fcf_yield": fcf_yield,
            "ev_ebitda": ev_ebitda,
            "return_3m": ret_3m,
            "return_12m": ret_12m,
            "volatility": volatility,
            "price": current_p,
            "market_cap": market_cap
        }

    def _calculate_quality_score(self, metrics: Dict[str, float]) -> tuple[float, Dict[str, Any]]:
        # We use 'Standard' scoring logic: map metric to 0-100 based on healthy thresholds
        
        # ROIC: > 15% is great (100), < 5% is bad (0)
        s_roic = self._score_metric(metrics["roic"], 0.05, 0.25)
        
        # ROIIC: > 20% is great
        s_roiic = self._score_metric(metrics["roiic"], 0.05, 0.30)
        
        # Revenue CAGR: > 15% great
        s_rev = self._score_metric(metrics["revenue_cagr_3y"], 0.0, 0.20)
        
        # FCF CAGR: > 15% great
        s_fcf = self._score_metric(metrics["fcf_cagr_3y"], 0.0, 0.20)
        
        # Margin: > 20% great
        s_margin = self._score_metric(metrics["net_margin"], 0.05, 0.25)
        
        # Leverage: < 0.5 great (100), > 2.0 bad (0). (Inverse)
        s_lev = 100 - self._score_metric(metrics["debt_to_equity"], 0.0, 2.0)
        
        # Weighted Quality
        score = (
            0.25 * s_roic +
            0.15 * s_roiic +
            0.20 * s_rev +
            0.10 * s_fcf +
            0.15 * s_margin +
            0.15 * s_lev
        )
        
        return score, {
            "drivers": {
                "ROIC": round(s_roic),
                "ROIIC": round(s_roiic),
                "Growth": round((s_rev + s_fcf)/2),
                "Margins": round(s_margin),
                "Safety": round(s_lev)
            },
            "metrics": {k: metrics[k] for k in ["roic", "roiic", "revenue_cagr_3y", "debt_to_equity"]}
        }

    def _calculate_valuation_score(self, metrics: Dict[str, float]) -> tuple[float, Dict[str, Any]]:
        # FCF Yield: > 5% good (60), > 10% great (100)
        s_yield = self._score_metric(metrics["fcf_yield"], 0.0, 0.10)
        
        # EV/EBITDA: < 8 great (100), > 25 bad (0). (Inverse)
        s_ev = 100 - self._score_metric(metrics["ev_ebitda"], 5, 25)
        
        score = 0.6 * s_yield + 0.4 * s_ev
        
        return score, {
            "drivers": {
                "FCF Yield": round(s_yield),
                "EV/EBITDA": round(s_ev)
            },
            "metrics": {k: metrics[k] for k in ["fcf_yield", "ev_ebitda"]}
        }

    def _calculate_momentum_score(self, metrics: Dict[str, float]) -> tuple[float, Dict[str, Any]]:
        # 3m Return: > 0% good, > 20% great. Too high (>50%) might be parabolic (penalty?)
        # For now linear mapping
        s_3m = self._score_metric(metrics["return_3m"], -0.10, 0.30)
        
        # 12m Return
        s_12m = self._score_metric(metrics["return_12m"], -0.10, 0.50)
        
        # Volatility: < 20% great (100), > 60% bad (0)
        s_vol = 100 - self._score_metric(metrics["volatility"], 0.15, 0.60)
        
        score = 0.4 * s_3m + 0.4 * s_12m + 0.2 * s_vol
        
        return score, {
            "drivers": {
                "Short Term Mom": round(s_3m),
                "Long Term Mom": round(s_12m),
                "Stability": round(s_vol)
            },
            "metrics": {k: metrics[k] for k in ["return_3m", "return_12m", "volatility"]}
        }
        
    def _score_metric(self, val: float, min_v: float, max_v: float) -> float:
        """Map value to 0-100 range."""
        if math.isnan(val): return 50.0
        norm = (val - min_v) / (max_v - min_v)
        return _clamp(norm * 100, 0, 100)

    def _classify_score(self, score: float) -> str:
        if score >= 80: return "Hidden Gem"
        if score >= 65: return "Investable"
        if score >= 50: return "Watchlist"
        return "Avoid"

    def _empty_score(self, ticker: str) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "cScore": 0,
            "hiddenGemScore": 0,
            "qualityScore": 0,
            "valuationScore": 0,
            "momentumScore": 0,
            "classification": "No Data",
            "current_price": 0,
            "details": {},
            "explanations": {}
        }

    def _build_explanations(self, q, v, m) -> Dict[str, str]:
        # Simple explanations based on scores
        return {
            "summary": f"Q: {q['drivers'].get('ROIC',0)} | V: {v['drivers'].get('FCF Yield',0)} | M: {m['drivers'].get('Short Term Mom',0)}"
        }

