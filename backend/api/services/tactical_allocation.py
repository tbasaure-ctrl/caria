"""
Tactical Asset Allocation (TAA) Service per audit document (2.2).
Implements macro-conditional portfolio allocation based on regime signals (Tabla 4).

Reglas de régimen:
- Alto Riesgo (Stress/Recession + VIX > 25): 30% acciones / 70% bonos
- Riesgo Moderado (Slowdown): 50% acciones / 50% bonos
- Bajo Riesgo (Expansion + VIX < 20): 70% acciones / 30% bonos
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

import yfinance as yf

LOGGER = logging.getLogger("caria.api.tactical_allocation")

# Tabla 4: Reglas de asignación macro-condicional
ALLOCATION_RULES = {
    "high_risk": {
        "stocks": 0.30,
        "bonds": 0.70,
        "cash": 0.00,
        "description": "Alto Riesgo: Stress/Recession + VIX > 25",
    },
    "moderate_risk": {
        "stocks": 0.50,
        "bonds": 0.50,
        "cash": 0.00,
        "description": "Riesgo Moderado: Slowdown",
    },
    "low_risk": {
        "stocks": 0.70,
        "bonds": 0.30,
        "cash": 0.00,
        "description": "Bajo Riesgo: Expansion + VIX < 20",
    },
    "extreme_stress": {
        "stocks": 0.20,
        "bonds": 0.60,
        "cash": 0.20,
        "description": "Estrés Extremo: VIX > 35",
    },
}


class TacticalAllocationService:
    """Service for tactical asset allocation based on regime signals."""

    def __init__(self):
        self.vix_threshold_high = 25.0
        self.vix_threshold_extreme = 35.0
        self.vix_threshold_low = 20.0

    def get_current_vix(self) -> Optional[float]:
        """Get current VIX level."""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            LOGGER.warning(f"Could not fetch VIX: {e}")
        return None

    def determine_risk_level(self, regime: str, vix: Optional[float] = None) -> str:
        """
        Determine risk level based on regime and VIX.
        
        Args:
            regime: Current regime from model ("expansion", "slowdown", "recession", "stress")
            vix: Current VIX level (fetched if not provided)
        
        Returns:
            Risk level: "high_risk", "moderate_risk", "low_risk", "extreme_stress"
        """
        if vix is None:
            vix = self.get_current_vix()

        # Extreme stress: VIX > 35
        if vix and vix > self.vix_threshold_extreme:
            return "extreme_stress"

        # High risk: Stress/Recession + VIX > 25
        if regime in ["stress", "recession"]:
            if vix and vix > self.vix_threshold_high:
                return "high_risk"
            return "moderate_risk"  # Fallback if VIX not available

        # Low risk: Expansion + VIX < 20
        if regime == "expansion":
            if vix and vix < self.vix_threshold_low:
                return "low_risk"
            return "moderate_risk"  # Expansion but VIX elevated

        # Moderate risk: Slowdown or default
        if regime == "slowdown":
            return "moderate_risk"

        # Default to moderate risk
        return "moderate_risk"

    def get_allocation(
        self, regime: str, vix: Optional[float] = None, include_etfs: bool = True
    ) -> Dict:
        """
        Get tactical asset allocation based on regime signals.
        
        Per audit document (2.2): Returns allocation percentages for stocks, bonds, cash
        based on Tabla 4 rules.
        
        Args:
            regime: Current regime from model
            vix: Optional VIX level (fetched if not provided)
            include_etfs: Whether to include ETF recommendations
        
        Returns:
            Dict with allocation percentages, risk level, and optional ETF recommendations
        """
        risk_level = self.determine_risk_level(regime, vix)
        allocation = ALLOCATION_RULES[risk_level].copy()

        result = {
            "regime": regime,
            "risk_level": risk_level,
            "vix": vix if vix else self.get_current_vix(),
            "allocation": {
                "stocks": allocation["stocks"],
                "bonds": allocation["bonds"],
                "cash": allocation["cash"],
            },
            "description": allocation["description"],
            "recommended_etfs": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add ETF recommendations if requested
        if include_etfs:
            result["recommended_etfs"] = self._get_etf_recommendations(
                allocation["stocks"], allocation["bonds"]
            )

        return result

    def _get_etf_recommendations(self, stocks_pct: float, bonds_pct: float) -> list[Dict]:
        """
        Get ETF recommendations based on allocation percentages.
        
        Returns list of ETFs with their recommended allocation percentages.
        """
        etfs = []

        # Stock ETFs
        if stocks_pct > 0:
            # Diversified stock ETFs
            if stocks_pct >= 0.5:
                etfs.append(
                    {
                        "ticker": "SPY",
                        "name": "SPDR S&P 500 ETF",
                        "allocation": stocks_pct * 0.6,  # 60% of stock allocation
                        "category": "stocks",
                    }
                )
                etfs.append(
                    {
                        "ticker": "VTI",
                        "name": "Vanguard Total Stock Market ETF",
                        "allocation": stocks_pct * 0.4,  # 40% of stock allocation
                        "category": "stocks",
                    }
                )
            else:
                etfs.append(
                    {
                        "ticker": "SPY",
                        "name": "SPDR S&P 500 ETF",
                        "allocation": stocks_pct,
                        "category": "stocks",
                    }
                )

        # Bond ETFs
        if bonds_pct > 0:
            if bonds_pct >= 0.5:
                etfs.append(
                    {
                        "ticker": "AGG",
                        "name": "iShares Core U.S. Aggregate Bond ETF",
                        "allocation": bonds_pct * 0.6,  # 60% of bond allocation
                        "category": "bonds",
                    }
                )
                etfs.append(
                    {
                        "ticker": "TLT",
                        "name": "iShares 20+ Year Treasury Bond ETF",
                        "allocation": bonds_pct * 0.4,  # 40% of bond allocation
                        "category": "bonds",
                    }
                )
            else:
                etfs.append(
                    {
                        "ticker": "AGG",
                        "name": "iShares Core U.S. Aggregate Bond ETF",
                        "allocation": bonds_pct,
                        "category": "bonds",
                    }
                )

        return etfs

    def get_detailed_allocation(
        self, regime: str, vix: Optional[float] = None
    ) -> Dict:
        """
        Get detailed allocation with sub-asset class breakdown.
        
        Extends basic allocation with more granular recommendations.
        """
        base_allocation = self.get_allocation(regime, vix, include_etfs=True)

        # Add sub-asset class breakdown
        stocks_pct = base_allocation["allocation"]["stocks"]
        bonds_pct = base_allocation["allocation"]["bonds"]
        cash_pct = base_allocation["allocation"]["cash"]

        detailed = {
            **base_allocation,
            "sub_allocations": {
                "stocks": {
                    "us_large_cap": stocks_pct * 0.50,
                    "us_small_cap": stocks_pct * 0.20,
                    "international": stocks_pct * 0.20,
                    "emerging_markets": stocks_pct * 0.10,
                },
                "bonds": {
                    "us_treasury": bonds_pct * 0.50,
                    "corporate": bonds_pct * 0.30,
                    "international": bonds_pct * 0.20,
                },
                "cash": {
                    "cash_equivalents": cash_pct,
                },
            },
        }

        return detailed


def get_tactical_allocation_service() -> TacticalAllocationService:
    """Get singleton instance."""
    return TacticalAllocationService()

