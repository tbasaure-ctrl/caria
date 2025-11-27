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
    
    def get_financial_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Obtiene los datos base reales de FMP"""
        base_url = "https://financialmodelingprep.com/api/v3"
        try:
            inc = requests.get(
                f"{base_url}/income-statement/{ticker}?limit=1&apikey={self.fmp_api_key}",
                timeout=10
            ).json()
            quote = requests.get(
                f"{base_url}/quote/{ticker}?apikey={self.fmp_api_key}",
                timeout=10
            ).json()
            
            if not inc or not quote:
                return None
            
            data = inc[0]
            revenue = data.get("revenue", 0)
            op_inc = data.get("operatingIncome", 0)
            
            return {
                "symbol": data.get("symbol"),
                "revenue": revenue,
                "op_income": op_inc,
                "net_income": data.get("netIncome", 0),
                "shares": data.get("weightedAverageShsOutDil", 0),
                "price": quote[0].get("price", 0) if quote else 0,
                "op_margin": op_inc / revenue if revenue else 0,
                "net_margin": data.get("netIncome", 0) / revenue if revenue else 0,
            }
        except Exception as e:
            LOGGER.error(f"Error fetching financial data for {ticker}: {e}")
            return None
    
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
        Now automatically detects if company is high-margin (NVDA) or low-margin (Walmart) and projects accordingly.
        Returns JSON-serializable dict.
        """
        # 1. Obtener Realidad Actual
        base = self.get_financial_data(ticker)
        if not base:
            raise ValueError(f"Ticker {ticker} no encontrado o datos insuficientes")

        # 2. Configurar Penalizaciones por Riesgo
        # El riesgo macro afecta al múltiplo de salida (PE) y al crecimiento
        # El riesgo industria afecta a los márgenes
        risk_growth_penalty = (macro_risk * 0.03) + (industry_risk * 0.01)
        risk_margin_penalty = industry_risk * 0.05  # Si es alto, baja 5% el margen
        risk_multiple_penalty = macro_risk * 5  # Si riesgo es 1, baja 5 puntos el PE

        # 3. Proyección a 5 Años
        years = list(range(2025, 2030))
        projections = []
        
        prev_revenue = base["revenue"]
        prev_shares = base["shares"]
        
        # Supuestos dinámicos basados en la empresa
        # Detecta automáticamente si es Tech (alto margen) o Retail (bajo margen)
        base_growth = 0.15 if base["op_margin"] > 0.20 else 0.05  # Si es Tech crece más, si es retail menos
        base_pe = 25 if base["op_margin"] > 0.20 else 15  # PE base según perfil

        for year in years:
            # Ajustar Crecimiento (Decae con el tiempo + Riesgo)
            growth_rate = max(0.02, base_growth - risk_growth_penalty - ((year - 2025) * 0.01))
            
            # Ajustar Margen (Se mantiene estable o baja por competencia)
            op_margin = max(0.05, base["op_margin"] - risk_margin_penalty)
            net_margin = max(0.03, base["net_margin"] - risk_margin_penalty)
            
            # Cálculos Financieros
            revenue = prev_revenue * (1 + growth_rate)
            op_income = revenue * op_margin
            net_income = revenue * net_margin
            fcf = op_income * 0.80  # Estimación FCF
            
            # Recompras (asumimos usa el 30% del FCF para recomprar acciones)
            buyback_cash = fcf * 0.30 
            est_share_price = base["price"] * (1.08 ** (year - 2024))  # Precio sube teóricamente
            shares_repurchased = buyback_cash / est_share_price if est_share_price > 0 else 0
            shares = max(prev_shares - shares_repurchased, 0)
            
            fcf_per_share = fcf / shares if shares > 0 else 0
            
            projections.append({
                "year": year,
                "revenue": revenue,
                "growth": growth_rate,
                "op_margin": op_margin,
                "net_margin": net_margin,
                "fcf": fcf,
                "shares": shares,
                "fcf_per_share": fcf_per_share
            })
            
            prev_revenue = revenue
            prev_shares = shares

        # 4. Valoración Final (Año 2029)
        last_year = projections[-1]
        
        # Múltiplo de Salida ajustado por riesgo
        exit_multiple = base_pe - risk_multiple_penalty
        target_price = last_year["fcf_per_share"] * exit_multiple
        
        upside = ((target_price / base["price"]) - 1) * 100 if base["price"] > 0 else 0

        # Formatear proyecciones para compatibilidad con frontend
        projection_data = {}
        for proj in projections:
            year = proj["year"]
            if "Revenue" not in projection_data:
                projection_data["Total Revenue"] = {}
            if "Revenue Growth" not in projection_data:
                projection_data["Revenue Growth"] = {}
            if "Op Margin" not in projection_data:
                projection_data["Op Margin"] = {}
            if "FCF Per Share" not in projection_data:
                projection_data["FCF Per Share"] = {}
            
            projection_data["Total Revenue"][year] = proj["revenue"]
            projection_data["Revenue Growth"][year] = proj["growth"]
            projection_data["Op Margin"][year] = proj["op_margin"]
            projection_data["FCF Per Share"][year] = proj["fcf_per_share"]

        return {
            "ticker": base["symbol"],
            "current_price": round(base["price"], 2),
            "fair_value": round(target_price, 2),
            "upside_percentage": round(upside, 2),
            "risk_score": round(((macro_risk + industry_risk) / 2) * 100, 2),
            "projections": projections,  # Enviamos la tabla completa por si el usuario quiere ver detalles
            "projection_data": projection_data  # Formato legacy para compatibilidad
        }

