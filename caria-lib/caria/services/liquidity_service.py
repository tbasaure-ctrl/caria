import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd

from caria.models.liquidity.liquidity_engine import LiquidityEngine

LOGGER = logging.getLogger(__name__)

class LiquidityService:
    _instance = None
    _cache: Dict[str, Any] = {}
    _last_update: Optional[datetime] = None
    _cache_duration = timedelta(hours=4) # Cache for 4 hours (FRED data updates daily)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LiquidityService, cls).__new__(cls)
            cls._instance.engine = LiquidityEngine()
        return cls._instance

    def get_current_status(self) -> Dict[str, Any]:
        """
        Returns the current hydraulic score and state.
        Uses caching to avoid hitting FRED API too frequently.
        """
        now = datetime.now()
        
        if self._last_update and (now - self._last_update) < self._cache_duration and self._cache:
            return self._cache

        try:
            LOGGER.info("Fetching fresh liquidity data...")
            df = self.engine.fetch_data()
            if df is None or df.empty:
                LOGGER.error("Failed to fetch liquidity data")
                return self._get_fallback_status()

            df = self.engine.calculate_signals(df)
            if df is None or df.empty:
                LOGGER.error("Failed to calculate liquidity signals")
                return self._get_fallback_status()

            latest = df.iloc[-1]
            
            status = {
                "score": int(latest['hydraulic_score']),
                "state": latest['liquidity_state'],
                "net_liquidity_bn": float(latest['net_liquidity']),
                "yield_curve_10y2y": float(latest['T10Y2Y']),
                "last_updated": now.isoformat(),
                "trend_roc_4w": float(latest['net_liq_chg_4w'])
            }
            
            self._cache = status
            self._last_update = now
            return status

        except Exception as e:
            LOGGER.error(f"Error in LiquidityService: {e}")
            return self._get_fallback_status()

    def _get_fallback_status(self):
        """Return cached or neutral status if fetch fails."""
        if self._cache:
            return self._cache
        return {
            "score": 50,
            "state": "NEUTRAL",
            "net_liquidity_bn": 0.0,
            "yield_curve_10y2y": 0.0,
            "trend_roc_4w": 0.0,
            "last_updated": datetime.now().isoformat(),
            "error": "Data unavailable"
        }

    def get_strategy_mode(self) -> Dict[str, str]:
        """Returns the active strategy mode based on liquidity."""
        status = self.get_current_status()
        state = status.get("state", "NEUTRAL")
        
        if state == "EXPANSION":
            return {
                "mode": "AGGRESSIVE",
                "description": "Liquidity is expanding. Focus on High Quality + Value (Growth).",
                "target_quadrant": "Quality+Value"
            }
        elif state == "CONTRACTION":
            return {
                "mode": "DEFENSIVE",
                "description": "Liquidity is contracting. Focus on Pure Quality (Cash Cows). Avoid Value Traps.",
                "target_quadrant": "Quality Only"
            }
        else:
            return {
                "mode": "NEUTRAL",
                "description": "Liquidity is neutral. Balanced approach.",
                "target_quadrant": "Quality+Value"
            }
