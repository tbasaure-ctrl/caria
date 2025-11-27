"""
Servicio de valoración basado en clasificación de crecimiento.
Implementa lógica mejorada que clasifica empresas según su tasa de crecimiento actual.
"""

import logging
import os
import requests
from typing import Dict, Any, Optional
import numpy as np

LOGGER = logging.getLogger("caria.services.growth_valuation")


class GrowthBasedValuationService:
    """
    Servicio de valoración que clasifica empresas según su perfil de crecimiento:
    - HYPER GROWTH: >30% crecimiento (Nvidia, Palantir)
    - STEADY GROWTH: 10-30% crecimiento (Apple, Google)
    - MATURE/VALUE: <10% crecimiento (Coca-Cola, Ford)
    """
    
    def __init__(self):
        self.fmp_api_key = os.getenv("FMP_API_KEY", "").strip()
        if not self.fmp_api_key:
            LOGGER.warning("FMP_API_KEY not configured, valuation may fail")
    
    def get_financial_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene datos financieros de FMP incluyendo crecimiento de ingresos.
        
        Args:
            ticker: Símbolo del ticker
            
        Returns:
            Dict con datos financieros o None si hay error
        """
        base_url = "https://financialmodelingprep.com/api/v3"
        
        try:
            # 1. Datos Básicos
            inc_response = requests.get(
                f"{base_url}/income-statement/{ticker}?limit=1&apikey={self.fmp_api_key}",
                timeout=15
            )
            inc_response.raise_for_status()
            inc = inc_response.json()
            
            quote_response = requests.get(
                f"{base_url}/quote/{ticker}?apikey={self.fmp_api_key}",
                timeout=15
            )
            quote_response.raise_for_status()
            quote = quote_response.json()
            
            # 2. Datos de Crecimiento (IMPORTANTE PARA CLASIFICACIÓN)
            growth_response = requests.get(
                f"{base_url}/financial-growth/{ticker}?limit=1&apikey={self.fmp_api_key}",
                timeout=15
            )
            growth_response.raise_for_status()
            growth = growth_response.json()
            
            if not inc or not isinstance(inc, list) or len(inc) == 0:
                LOGGER.warning(f"No income statement data for {ticker}")
                return None
            
            if not quote or not isinstance(quote, list) or len(quote) == 0:
                LOGGER.warning(f"No quote data for {ticker}")
                return None
            
            data = inc[0]
            growth_data = growth[0] if growth and isinstance(growth, list) and len(growth) > 0 else {}
            
            revenue = float(data.get("revenue", 0) or 0)
            op_inc = float(data.get("operatingIncome", 0) or 0)
            net_income = float(data.get("netIncome", 0) or 0)
            shares = float(data.get("weightedAverageShsOutDil", 0) or 0)
            price = float(quote[0].get("price", 0) or 0)
            
            # Validaciones básicas
            if revenue <= 0:
                LOGGER.warning(f"Invalid revenue for {ticker}: {revenue}")
                return None
            
            if shares <= 0:
                LOGGER.warning(f"Invalid shares for {ticker}: {shares}")
                return None
            
            if price <= 0:
                LOGGER.warning(f"Invalid price for {ticker}: {price}")
                return None
            
            # Obtener crecimiento de ingresos (puede ser None o un valor muy alto)
            current_revenue_growth = growth_data.get("revenueGrowth")
            if current_revenue_growth is None:
                # Si no hay datos de crecimiento, usar un valor conservador
                current_revenue_growth = 0.05  # 5% por defecto
                LOGGER.info(f"No growth data for {ticker}, using default 5%")
            else:
                current_revenue_growth = float(current_revenue_growth)
                # Limitar valores extremos (ej: NVDA puede tener 1.25 = 125%)
                if current_revenue_growth > 2.0:  # Más del 200%
                    LOGGER.warning(
                        f"Extreme growth rate for {ticker}: {current_revenue_growth:.2%}, "
                        "capping at 200%"
                    )
                    current_revenue_growth = 2.0
            
            return {
                "symbol": data.get("symbol", ticker.upper()),
                "revenue": revenue,
                "op_income": op_inc,
                "net_income": net_income,
                "shares": shares,
                "price": price,
                "op_margin": op_inc / revenue if revenue > 0 else 0.0,
                "net_margin": net_income / revenue if revenue > 0 else 0.0,
                "current_revenue_growth": current_revenue_growth
            }
            
        except requests.exceptions.RequestException as e:
            LOGGER.error(f"Request error fetching financial data for {ticker}: {e}")
            return None
        except (ValueError, TypeError, KeyError) as e:
            LOGGER.error(f"Data parsing error for {ticker}: {e}")
            return None
        except Exception as e:
            LOGGER.error(f"Unexpected error fetching financial data for {ticker}: {e}", exc_info=True)
            return None
    
    def classify_growth_profile(self, current_growth: float) -> Dict[str, Any]:
        """
        Clasifica el perfil de crecimiento de la empresa.
        
        Args:
            current_growth: Tasa de crecimiento actual de ingresos (ej: 0.30 = 30%)
            
        Returns:
            Dict con parámetros del perfil (base_growth_start, base_pe, decay_rate, profile_name)
        """
        if current_growth > 0.30:
            # PERFIL "HYPER GROWTH" (Nvidia, Palantir)
            # Asumimos que el crecimiento bajará gradualmente, pero empieza alto
            base_growth_start = min(current_growth, 0.40)  # Topeamos en 40% para ser prudentes
            base_pe = 35  # El mercado paga premium por crecimiento
            decay_rate = 0.05  # El crecimiento baja 5% cada año
            profile_name = "HYPER_GROWTH"
        elif current_growth > 0.10:
            # PERFIL "STEADY GROWTH" (Apple, Google)
            base_growth_start = 0.12
            base_pe = 22
            decay_rate = 0.01
            profile_name = "STEADY_GROWTH"
        else:
            # PERFIL "MATURE / VALUE" (Coca-Cola, Ford)
            base_growth_start = 0.03
            base_pe = 12
            decay_rate = 0.0
            profile_name = "MATURE_VALUE"
        
        return {
            "profile_name": profile_name,
            "base_growth_start": base_growth_start,
            "base_pe": base_pe,
            "decay_rate": decay_rate
        }
    
    def calculate_valuation(
        self,
        ticker: str,
        macro_risk: float = 0.0,
        industry_risk: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calcula valoración basada en proyecciones de crecimiento.
        
        Args:
            ticker: Símbolo del ticker
            macro_risk: Factor de riesgo macro (0-1), default 0
            industry_risk: Factor de riesgo industria (0-1), default 0
            
        Returns:
            Dict con valoración completa incluyendo proyecciones
            
        Raises:
            ValueError: Si el ticker no se encuentra o datos insuficientes
        """
        # Validar parámetros de riesgo
        macro_risk = max(0.0, min(1.0, float(macro_risk)))
        industry_risk = max(0.0, min(1.0, float(industry_risk)))
        
        # 1. Obtener datos base
        base = self.get_financial_data(ticker)
        if not base:
            raise ValueError(f"Ticker {ticker} no encontrado o datos insuficientes")
        
        # 2. Clasificar perfil de crecimiento
        current_growth = base["current_revenue_growth"]
        profile = self.classify_growth_profile(current_growth)
        
        LOGGER.info(
            f"{ticker}: Growth={current_growth:.1%}, Profile={profile['profile_name']}, "
            f"BasePE={profile['base_pe']}, BaseGrowth={profile['base_growth_start']:.1%}"
        )
        
        # 3. Ajustes de Riesgo (Risk Penalties)
        risk_pe_penalty = macro_risk * 8  # Si hay crisis, el múltiplo baja fuerte
        risk_growth_penalty = industry_risk * 0.05
        
        # 4. Proyección a 5 años (2025-2029)
        years = list(range(2025, 2030))
        projections = []
        
        prev_revenue = base["revenue"]
        prev_shares = base["shares"]
        
        # Iteración Año a Año
        for i, year in enumerate(years):
            # El crecimiento decae cada año (nadie crece al 40% por siempre)
            # Ej: Año 1: 40%, Año 2: 35%, Año 3: 30%...
            raw_growth = max(0.02, profile["base_growth_start"] - (i * profile["decay_rate"]))
            
            # Aplicamos castigo de riesgo
            final_growth = max(0.0, raw_growth - risk_growth_penalty)
            
            # Márgenes (Si es NVDA, mantenemos márgenes altos)
            op_margin = base["op_margin"] * (1 - (industry_risk * 0.1))  # Leve impacto si hay riesgo
            op_margin = max(0.05, op_margin)  # Mínimo 5% de margen operativo
            
            # Proyecciones financieras
            revenue = prev_revenue * (1 + final_growth)
            op_income = revenue * op_margin
            fcf = op_income * 0.80  # Estimación flujo de caja libre (80% del op income)
            
            # Recompra de acciones (Buybacks)
            buyback_amount = fcf * 0.20  # 20% del flujo a recompras
            est_price = base["price"] * (1.10 ** (i + 1))  # Precio sube teóricamente para calcular recompra
            shares_repurchased = buyback_amount / est_price if est_price > 0 else 0
            shares = max(prev_shares - shares_repurchased, 0)
            
            fcf_per_share = fcf / shares if shares > 0 else 0
            
            projections.append({
                "year": year,
                "revenue": round(revenue, 2),
                "growth": round(final_growth, 4),
                "op_margin": round(op_margin, 4),
                "fcf": round(fcf, 2),
                "shares": round(shares, 0),
                "fcf_per_share": round(fcf_per_share, 2)
            })
            
            prev_revenue = revenue
            prev_shares = shares
        
        # 5. Valor Terminal
        last_year = projections[-1]
        final_multiple = max(10, profile["base_pe"] - risk_pe_penalty)  # Nunca menos de 10x
        
        target_price = last_year["fcf_per_share"] * final_multiple
        upside = ((target_price / base["price"]) - 1) * 100 if base["price"] > 0 else 0
        
        return {
            "ticker": base["symbol"],
            "current_price": round(base["price"], 2),
            "fair_value": round(target_price, 2),
            "upside_percentage": round(upside, 2),
            "profile": profile["profile_name"],
            "current_revenue_growth": round(current_growth, 4),
            "base_pe": profile["base_pe"],
            "final_multiple": round(final_multiple, 2),
            "risk_adjustments": {
                "macro_risk": macro_risk,
                "industry_risk": industry_risk,
                "pe_penalty": round(risk_pe_penalty, 2),
                "growth_penalty": round(risk_growth_penalty, 4)
            },
            "projections": projections
        }
