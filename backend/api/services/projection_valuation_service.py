import logging
import os
import requests
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

LOGGER = logging.getLogger("caria.services.projection_valuation")

class ProjectionValuationService:
    """
    Service for running projection-based valuation models.
    Implements the projection model logic from the Python code.
    """
    
    def __init__(self):
        self.fmp_api_key = os.getenv("FMP_API_KEY", "").strip()
        if not self.fmp_api_key:
            LOGGER.warning("FMP_API_KEY not configured, projection valuation may fail")
    
    def get_base_data(self, ticker: str) -> Dict[str, Any]:
        """Fetches TTM or latest full year data to serve as the base year."""
        base_url = "https://financialmodelingprep.com/api/v3"
        
        try:
            # Fetch Income Statement (Annual)
            inc_stmt = requests.get(
                f"{base_url}/income-statement/{ticker}?limit=1&apikey={self.fmp_api_key}",
                timeout=10
            ).json()
            
            # Fetch Key Metrics (for shares, etc)
            metrics = requests.get(
                f"{base_url}/key-metrics-ttm/{ticker}?limit=1&apikey={self.fmp_api_key}",
                timeout=10
            ).json()
            
            # Fetch Quote (for current price)
            quote = requests.get(
                f"{base_url}/quote/{ticker}?apikey={self.fmp_api_key}",
                timeout=10
            ).json()
            
            if not inc_stmt:
                raise ValueError("Could not fetch data. Check API Key or Ticker.")
            
            data = inc_stmt[0]
            curr_metrics = metrics[0] if metrics else {}
            curr_price = quote[0]['price'] if quote and len(quote) > 0 else 0
            
            return {
                "revenue": data.get("revenue", 0),
                "operating_income": data.get("operatingIncome", 0),
                "net_income": data.get("netIncome", 0),
                "shares_outstanding": data.get("weightedAverageShsOutDil", 0) or curr_metrics.get("sharesOutstanding", 0),
                "price": curr_price,
                "fcf": (data.get("operatingIncome", 0) * 0.8)  # Rough estimate
            }
        except Exception as e:
            LOGGER.error(f"Error fetching base data for {ticker}: {e}")
            raise
    
    def run_projection_model(
        self, 
        base_data: Dict[str, Any], 
        assumptions: Dict[int, Dict[str, Any]], 
        risk_profile: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Emulates the spreadsheet logic.
        risk_profile: dict with keys like 'macro_risk' (0-1), 'industry_risk' (0-1)
        """
        
        # Initialize Risk Penalties (Default 0 means no penalty)
        macro_risk = risk_profile.get('macro_risk', 0) if risk_profile else 0
        ind_risk = risk_profile.get('industry_risk', 0) if risk_profile else 0
        
        # Helper to apply risk to a growth rate
        def adjust_growth(rate):
            return rate - (macro_risk * 0.02) - (ind_risk * 0.01)
        
        # Helper to apply risk to a margin
        def adjust_margin(margin):
            return margin - (ind_risk * 0.02)
        
        # Helper to apply risk to valuation multiples
        def adjust_multiple(mult):
            return mult * (1 - macro_risk * 0.1)
        
        # Prepare years
        years = [2024, 2025, 2026, 2027, 2028, 2029]
        
        # Initialize row dictionaries
        rows = {
            "Total Revenue": [],
            "Revenue Growth": [],
            "Operating Income": [],
            "Op Margin": [],
            "Non-GAAP Net Income": [],
            "Net Margin": [],
            "Adj Free Cash Flow": [],
            "FCF Margin": [],
            "Buybacks ($)": [],
            "Shares Outstanding": [],
            "Price Per Share (Est)": [],
            "FCF Per Share": [],
            "EPS (Non-GAAP)": [],
            "Exit Multiple (FCF)": [],
            "Price Target (FCF)": []
        }
        
        # BASE YEAR (2024)
        current_shares = base_data['shares_outstanding']
        current_rev = base_data['revenue']
        
        # Fill 2024 (Base)
        rows["Total Revenue"].append(current_rev)
        rows["Revenue Growth"].append(0.0681)  # From image (6.81%)
        rows["Operating Income"].append(base_data['operating_income'])
        rows["Op Margin"].append(base_data['operating_income'] / current_rev if current_rev > 0 else 0)
        rows["Non-GAAP Net Income"].append(base_data['net_income'])
        rows["Net Margin"].append(base_data['net_income'] / current_rev if current_rev > 0 else 0)
        rows["Adj Free Cash Flow"].append(base_data['fcf'])
        rows["FCF Margin"].append(0.2086)  # From image
        rows["Buybacks ($)"].append(0)
        rows["Shares Outstanding"].append(current_shares)
        rows["Price Per Share (Est)"].append(base_data['price'])
        rows["FCF Per Share"].append(base_data['fcf'] / current_shares if current_shares > 0 else 0)
        rows["EPS (Non-GAAP)"].append(base_data['net_income'] / current_shares if current_shares > 0 else 0)
        rows["Exit Multiple (FCF)"].append(16)
        rows["Price Target (FCF)"].append(0)
        
        # PROJECTION LOOP (2025-2029)
        prev_rev = current_rev
        prev_shares = current_shares
        
        for i, year in enumerate(years[1:], start=1):  # Start from 2025
            # Get Base Assumption
            asm = assumptions.get(year, {})
            
            # 2. Apply Risk Adjustments
            g_rate = adjust_growth(asm.get('growth', 0.05))
            op_marg = adjust_margin(asm.get('op_margin', 0.18))
            net_marg = adjust_margin(asm.get('net_margin', 0.16))
            fcf_marg = adjust_margin(asm.get('fcf_margin', 0.20))
            exit_mult = adjust_multiple(asm.get('pe_multiple', 16))
            
            # 3. Calculate Financials
            rev = prev_rev * (1 + g_rate)
            op_inc = rev * op_marg
            net_inc = rev * net_marg
            fcf = rev * fcf_marg
            
            # 4. Calculate Share Count (Buyback Logic)
            est_price = asm.get('est_price_per_share', 100)
            buyback_cash = asm.get('buybacks', 0)
            shares_repurchased = buyback_cash / est_price if est_price > 0 else 0
            shares = max(prev_shares - shares_repurchased, 0)
            
            # 5. Per Share Metrics
            fcf_per_share = fcf / shares if shares > 0 else 0
            eps = net_inc / shares if shares > 0 else 0
            
            # 6. Valuation Target
            price_target = fcf_per_share * exit_mult
            
            # Store Data
            rows["Total Revenue"].append(rev)
            rows["Revenue Growth"].append(g_rate)
            rows["Operating Income"].append(op_inc)
            rows["Op Margin"].append(op_marg)
            rows["Non-GAAP Net Income"].append(net_inc)
            rows["Net Margin"].append(net_marg)
            rows["Adj Free Cash Flow"].append(fcf)
            rows["FCF Margin"].append(fcf_marg)
            rows["Buybacks ($)"].append(buyback_cash)
            rows["Shares Outstanding"].append(shares)
            rows["Price Per Share (Est)"].append(est_price)
            rows["FCF Per Share"].append(fcf_per_share)
            rows["EPS (Non-GAAP)"].append(eps)
            rows["Exit Multiple (FCF)"].append(exit_mult)
            rows["Price Target (FCF)"].append(price_target)
            
            # Update for next loop
            prev_rev = rev
            prev_shares = shares
        
        # Convert to DataFrame format and return as dict
        result = {}
        for key, values in rows.items():
            result[key] = {year: val for year, val in zip(years, values)}
        
        return result
    
    def get_valuation(
        self, 
        ticker: str, 
        macro_risk: float = 0.0,
        industry_risk: float = 0.0
    ) -> Dict[str, Any]:
        """
        Main method to get projection-based valuation.
        Returns JSON-serializable dict.
        """
        try:
            # Fetch base data
            base_data = self.get_base_data(ticker)
            
            # Default assumptions (can be customized per ticker)
            assumptions_map = {
                2025: {
                    'growth': 0.045, 
                    'op_margin': 0.175, 
                    'net_margin': 0.160, 
                    'fcf_margin': 0.20, 
                    'buybacks': 6_000_000_000, 
                    'est_price_per_share': 85, 
                    'pe_multiple': 16
                },
                2026: {
                    'growth': 0.050, 
                    'op_margin': 0.180, 
                    'net_margin': 0.165, 
                    'fcf_margin': 0.21, 
                    'buybacks': 6_250_000_000, 
                    'est_price_per_share': 100, 
                    'pe_multiple': 16
                },
                2027: {
                    'growth': 0.060, 
                    'op_margin': 0.185, 
                    'net_margin': 0.170, 
                    'fcf_margin': 0.215, 
                    'buybacks': 6_500_000_000, 
                    'est_price_per_share': 120, 
                    'pe_multiple': 18
                },
                2028: {
                    'growth': 0.070, 
                    'op_margin': 0.1875, 
                    'net_margin': 0.1725, 
                    'fcf_margin': 0.2175, 
                    'buybacks': 6_750_000_000, 
                    'est_price_per_share': 140, 
                    'pe_multiple': 18
                },
                2029: {
                    'growth': 0.070, 
                    'op_margin': 0.190, 
                    'net_margin': 0.175, 
                    'fcf_margin': 0.22, 
                    'buybacks': 7_000_000_000, 
                    'est_price_per_share': 166, 
                    'pe_multiple': 18
                },
            }
            
            # Risk profile
            risk_profile = {
                'macro_risk': macro_risk,
                'industry_risk': industry_risk
            }
            
            # Run projection model
            projection_data = self.run_projection_model(
                base_data, 
                assumptions_map, 
                risk_profile
            )
            
            # Extract key metrics
            current_price = base_data['price']
            target_2029 = projection_data.get("Price Target (FCF)", {}).get(2029, 0)
            upside = ((target_2029 / current_price) - 1) * 100 if current_price > 0 else 0
            
            return {
                "ticker": ticker.upper(),
                "current_price": round(current_price, 2),
                "target_price_2029": round(target_2029, 2),
                "upside": round(upside, 2),
                "base_revenue": base_data['revenue'],
                "projection_data": projection_data,
                "base_data": base_data
            }
        except Exception as e:
            LOGGER.error(f"Error in projection valuation for {ticker}: {e}")
            raise

