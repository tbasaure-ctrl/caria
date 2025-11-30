"""
C-Score Engine v2: Quality Slope + Mispricing
Identifies companies becoming winners, not already established winners.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

LOGGER = logging.getLogger("caria.services.c_score_engine")


class CScoreEngine:
    """
    Redesigned C-Score Engine that finds alpha, not just quality.
    
    Formula: C_SCORE = (C_QUALITY^0.6) * (C_DELTA^1.2) * MISPRICING_ADJUST
    
    Where:
    - C_QUALITY: Business durability (30% weight)
    - C_DELTA: Improvement momentum (50% weight) ← Alpha engine
    - MISPRICING_ADJUST: Contrarian signals (20% weight)
    """
    
    def __init__(self, fmp_key: str = None, alpha_vantage_key: str = None):
        self.fmp_key = fmp_key or os.getenv('FMP_API_KEY', '').strip()
        self.av_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY', '').strip()
        self.fmp_base = 'https://financialmodelingprep.com/api/v3'
        self.av_base = 'https://www.alphavantage.co/query'
    
    def calculate_c_score(self, ticker: str) -> Dict:
        """
        Main entry point for C-Score calculation.
        
        Returns dict with final_score and component breakdown.
        """
        try:
            # Calculate components
            quality_result = self.calculate_c_quality(ticker)
            delta_result = self.calculate_c_delta(ticker)
            mispricing_result = self.calculate_mispricing_adjust(ticker)
            
            # Extract scores (0-100 scale)
            quality_score = quality_result.get('score', 50)
            delta_score = delta_result.get('score', 50)
            mispricing_adjust = mispricing_result.get('multiplier', 1.0)
            
            # Normalize to 0-1 for formula
            q_norm = quality_score / 100
            d_norm = delta_score / 100
            
            # Apply formula: (Quality^0.6) * (Delta^1.2) * Mispricing
            final_score = (q_norm ** 0.6) * (d_norm ** 1.2) * mispricing_adjust
            final_score = final_score * 1000  # Scale to 0-1000 range
            
            return {
                "ticker": ticker,
                "final_c_score": round(final_score, 2),
                "c_quality": round(quality_score, 2),
                "c_delta": round(delta_score, 2),
                "mispricing_adjust": round(mispricing_adjust, 2),
                "breakdown": {
                    "quality_details": quality_result.get('details', {}),
                    "delta_details": delta_result.get('details', {}),
                    "mispricing_details": mispricing_result.get('details', {})
                },
                "updated_at": datetime.now().isoformat()
            }
        except Exception as e:
            LOGGER.error(f"Error calculating C-Score for {ticker}: {e}")
            return {
                "ticker": ticker,
                "final_c_score": 0,
                "error": str(e)
            }
    
    def calculate_c_quality(self, ticker: str) -> Dict:
        """
        Calculate C-QUALITY: Business durability and execution.
        
        Components:
        - ROIC vs WACC spread (30%)
        - FCF conversion (25%)
        - Operating leverage (25%)
        - Market structure/moat (20%)
        """
        try:
            ratios = self._fetch_fmp(f'/ratios-ttm/{ticker}')
            key_metrics = self._fetch_fmp(f'/key-metrics-ttm/{ticker}')
            
            if not ratios or not key_metrics:
                return {"score": 0, "details": {"error": "No data available"}}
            
            ratios = ratios[0] if isinstance(ratios, list) else ratios
            key_metrics = key_metrics[0] if isinstance(key_metrics, list) else key_metrics
            
            # 1. ROIC vs WACC Spread (30%)
            roic = ratios.get('returnOnCapitalEmployedTTM', 0) or 0
            wacc = 0.08  # Assume 8% WACC if not available
            roic_spread = max(0, (roic - wacc) * 100)  # Convert to percentage points
            roic_score = min(30, roic_spread * 3)  # Cap at 30
            
            # 2. FCF Conversion (25%)
            fcf = key_metrics.get('freeCashFlowTTM', 0) or 0
            net_income = ratios.get('netIncomeTTM', 1) or 1
            fcf_conversion = (fcf / net_income) if net_income != 0 else 0
            fcf_score = min(25, fcf_conversion * 25)  # 100% conversion = 25 points
            
            # 3. Operating Leverage (25%)
            gross_margin = ratios.get('grossProfitMarginTTM', 0) or 0
            operating_margin = ratios.get('operatingProfitMarginTTM', 0) or 0
            op_leverage = (operating_margin / gross_margin) if gross_margin > 0 else 0
            op_leverage_score = min(25, op_leverage * 30)
            
            # 4. Market Structure / Moat (20%)
            # Proxy: Gross margin > 40% indicates pricing power
            moat_score = 0
            if gross_margin > 0.4:
                moat_score = 20
            elif gross_margin > 0.3:
                moat_score = 15
            elif gross_margin > 0.2:
                moat_score = 10
            
            total_quality = roic_score + fcf_score + op_leverage_score + moat_score
            
            return {
                "score": min(100, total_quality),
                "details": {
                    "roic_vs_wacc": round(roic_spread, 2),
                    "fcf_conversion": round(fcf_conversion, 2),
                    "operating_leverage": round(op_leverage, 2),
                    "gross_margin": round(gross_margin * 100, 2),
                    "roic_score": round(roic_score, 1),
                    "fcf_score": round(fcf_score, 1),
                    "op_leverage_score": round(op_leverage_score, 1),
                    "moat_score": round(moat_score, 1)
                }
            }
        except Exception as e:
            LOGGER.error(f"Error in calculate_c_quality for {ticker}: {e}")
            return {"score": 0, "details": {"error": str(e)}}
    
    def calculate_c_delta(self, ticker: str) -> Dict:
        """
        Calculate C-DELTA: Improvement momentum (THE ALPHA ENGINE).
        
        Components:
        - ROIC acceleration YoY (25%)
        - FCF growth / EV (20%)
        - Revenue per employee growth (15%)
        - Margin expansion streak ≥ 3yrs (20%)
        - Insider ownership rising (10%)
        - Capex/FCF increasing + ROIC stable (10%)
        """
        try:
            # Fetch historical data
            ratios_hist = self._fetch_fmp(f'/ratios/{ticker}', {'limit': 8})
            key_metrics_hist = self._fetch_fmp(f'/key-metrics/{ticker}', {'limit': 8})
            insider_data = self._fetch_fmp(f'/insider-trading', {'symbol': ticker, 'limit': 50})
            
            if not ratios_hist or len(ratios_hist) < 3:
                return {"score": 0, "details": {"error": "Insufficient historical data"}}
            
            # 1. ROIC Acceleration (25%)
            roic_values = [r.get('returnOnCapitalEmployed', 0) or 0 for r in ratios_hist[:5]]
            roic_slope = self._calculate_slope(roic_values)
            roic_accel_score = min(25, max(0, roic_slope * 500))  # Positive slope = higher score
            
            # 2. FCF Growth / EV (20%)
            if key_metrics_hist and len(key_metrics_hist) >= 2:
                fcf_current = key_metrics_hist[0].get('freeCashFlowTTM', 0) or 0
                fcf_1y_ago = key_metrics_hist[1].get('freeCashFlowTTM', 1) or 1
                ev = key_metrics_hist[0].get('enterpriseValueTTM', 1) or 1
                
                fcf_growth_rate = (fcf_current - fcf_1y_ago) / abs(fcf_1y_ago) if fcf_1y_ago != 0 else 0
                fcf_to_ev = fcf_growth_rate / (ev / 1e9) if ev > 0 else 0  # Normalize by EV in billions
                fcf_score = min(20, max(0, fcf_to_ev * 100))
            else:
                fcf_score = 0
            
            # 3. Revenue per Employee Growth (15%)
            rev_per_emp_growth = self._calculate_rev_per_employee_growth(ratios_hist)
            rev_emp_score = min(15, max(0, rev_per_emp_growth * 50))
            
            # 4. Margin Expansion Streak (20%)
            margin_streak = self._detect_margin_streak(ratios_hist)
            margin_score = min(20, margin_streak * 7)  # 3+ years = 20 points
            
            # 5. Insider Ownership Rising (10%)
            insider_score = self._calculate_insider_trend(insider_data)
            
            # 6. Capex/FCF Increasing + ROIC Stable (10%)
            capex_score = self._calculate_capex_score(key_metrics_hist, roic_values)
            
            total_delta = (roic_accel_score + fcf_score + rev_emp_score + 
                          margin_score + insider_score + capex_score)
            
            return {
                "score": min(100, total_delta),
                "details": {
                    "roic_slope": round(roic_slope, 4),
                    "roic_accel_score": round(roic_accel_score, 1),
                    "fcf_score": round(fcf_score, 1),
                    "rev_emp_score": round(rev_emp_score, 1),
                    "margin_streak": margin_streak,
                    "margin_score": round(margin_score, 1),
                    "insider_score": round(insider_score, 1),
                    "capex_score": round(capex_score, 1)
                }
            }
        except Exception as e:
            LOGGER.error(f"Error in calculate_c_delta for {ticker}: {e}")
            return {"score": 0, "details": {"error": str(e)}}
    
    def calculate_mispricing_adjust(self, ticker: str) -> Dict:
        """
        Calculate MISPRICING_ADJUST: Contrarian signals.
        
        Components:
        - FCF yield vs 5yr median deviation
        - EV/S forward vs growth rate
        - Short interest trend
        - Analyst revisions
        
        Returns multiplier (0.5 to 1.5)
        """
        try:
            ratios = self._fetch_fmp(f'/ratios-ttm/{ticker}')
            ratios_hist = self._fetch_fmp(f'/ratios/{ticker}', {'limit': 20})
            
            if not ratios or not ratios_hist:
                return {"multiplier": 1.0, "details": {"error": "No data"}}
            
            ratios = ratios[0] if isinstance(ratios, list) else ratios
            
            multiplier = 1.0
            reasons = []
            
            # 1. FCF Yield vs Historical
            fcf_yield_current = ratios.get('freeCashFlowYieldTTM', 0) or 0
            fcf_yields_hist = [r.get('freeCashFlowYield', 0) or 0 for r in ratios_hist if r.get('freeCashFlowYield')]
            
            if len(fcf_yields_hist) >= 5:
                median_fcf_yield = np.median(fcf_yields_hist)
                std_fcf_yield = np.std(fcf_yields_hist)
                
                if std_fcf_yield > 0:
                    z_score = (fcf_yield_current - median_fcf_yield) / std_fcf_yield
                    if z_score > 1:  # More than 1 SD above median = undervalued
                        multiplier += 0.2
                        reasons.append("FCF yield above 5yr median")
            
            # 2. EV/S vs Growth
            ev_to_sales = ratios.get('enterpriseValueOverSalesTTM', 0) or 0
            revenue_growth = ratios.get('revenueGrowthTTM', 0) or 0
            
            if revenue_growth > 0 and ev_to_sales > 0:
                if ev_to_sales < revenue_growth:  # Growth at a discount
                    multiplier += 0.15
                    reasons.append("EV/S < growth rate")
            
            # 3. Ensure multiplier stays in range [0.5, 1.5]
            multiplier = max(0.5, min(1.5, multiplier))
            
            return {
                "multiplier": multiplier,
                "details": {
                    "fcf_yield_current": round(fcf_yield_current, 4),
                    "fcf_yield_5yr_median": round(np.median(fcf_yields_hist), 4) if fcf_yields_hist else 0,
                    "ev_to_sales": round(ev_to_sales, 2),
                    "revenue_growth": round(revenue_growth * 100, 2),
                    "reasons": reasons
                }
            }
        except Exception as e:
            LOGGER.error(f"Error in calculate_mispricing_adjust for {ticker}: {e}")
            return {"multiplier": 1.0, "details": {"error": str(e)}}
    
    # Helper Methods
    
    def _fetch_fmp(self, endpoint: str, params: Dict = None) -> Optional[List]:
        """Fetch data from FMP API."""
        if not self.fmp_key:
            return None
        
        if params is None:
            params = {}
        params['apikey'] = self.fmp_key
        
        try:
            url = f"{self.fmp_base}{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            LOGGER.warning(f"FMP fetch error {endpoint}: {e}")
            return None
    
    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate linear regression slope."""
        if not values or len(values) < 2:
            return 0.0
        
        clean_values = [v for v in values if v is not None and not np.isnan(v)]
        if len(clean_values) < 2:
            return 0.0
        
        x = np.arange(len(clean_values))
        try:
            slope, _ = np.polyfit(x, clean_values, 1)
            return slope
        except:
            return 0.0
    
    def _calculate_rev_per_employee_growth(self, ratios_hist: List[Dict]) -> float:
        """Calculate revenue per employee growth rate."""
        try:
            rev_per_emp_values = []
            for r in ratios_hist[:5]:
                revenue = r.get('revenueTTM', 0) or 0
                employees = r.get('numberOfEmployees', 0) or 0
                if revenue > 0 and employees > 0:
                    rev_per_emp_values.append(revenue / employees)
            
            if len(rev_per_emp_values) >= 2:
                growth = (rev_per_emp_values[0] - rev_per_emp_values[-1]) / rev_per_emp_values[-1]
                return growth
        except:
            pass
        return 0.0
    
    def _detect_margin_streak(self, ratios_hist: List[Dict]) -> int:
        """Count consecutive years of margin expansion."""
        try:
            margins = [r.get('netProfitMarginTTM', 0) or 0 for r in ratios_hist[:6]]
            margins = [m for m in margins if m is not None]
            
            streak = 0
            for i in range(len(margins) - 1):
                if margins[i] > margins[i + 1]:
                    streak += 1
                else:
                    break
            return streak
        except:
            return 0
    
    def _calculate_insider_trend(self, insider_data: List[Dict]) -> float:
        """Calculate insider buying trend score (0-10)."""
        try:
            if not insider_data:
                return 0.0
            
            # Count purchases in last 3 months
            three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            purchases = [t for t in insider_data 
                        if t.get('transactionDate', '') >= three_months_ago 
                        and t.get('transactionType') == 'P-Purchase']
            
            # More purchases = higher score
            return min(10, len(purchases) * 2)
        except:
            return 0.0
    
    def _calculate_capex_score(self, key_metrics_hist: List[Dict], roic_values: List[float]) -> float:
        """Score for increasing capex with stable ROIC."""
        try:
            if not key_metrics_hist or len(key_metrics_hist) < 3:
                return 0.0
            
            # Check if capex/FCF is increasing
            ratios = []
            for km in key_metrics_hist[:3]:
                fcf = km.get('freeCashFlowTTM', 0) or 0
                capex = km.get('capitalExpenditureTTM', 0) or 0
                if fcf != 0:
                    ratios.append(abs(capex) / abs(fcf))
            
            if len(ratios) >= 2 and ratios[0] > ratios[-1]:
                # Capex increasing
                roic_stable = len(roic_values) >= 3 and np.std(roic_values[:3]) < 0.05
                if roic_stable:
                    return 10.0
                return 5.0
        except:
            pass
        return 0.0
