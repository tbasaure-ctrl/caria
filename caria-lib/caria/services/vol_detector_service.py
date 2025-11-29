import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

LOGGER = logging.getLogger(__name__)

class VolDetectorService:
    """
    The Guard: CNN-inspired Vol Detector (Tactical Circuit Breaker).
    
    Detects sudden volatility spikes that the slow Liquidity Engine hasn't seen yet.
    Acts as a fast override to PAUSE buying during market-wide volatility events.
    """
    
    def __init__(self):
        self.spike_threshold = 2.0  # Current vol > 2x avg = SPIKE
        
    def check_volatility_spike(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Detects if current volatility is spiking relative to recent average.
        
        Args:
            prices: Series of recent prices (at least 20 data points)
            
        Returns:
            Dict with 'signal', 'current_vol', 'avg_vol', 'ratio'
        """
        try:
            if len(prices) < 20:
                return {
                    "signal": "INSUFFICIENT_DATA",
                    "current_vol": 0,
                    "avg_vol": 0,
                    "ratio": 0,
                    "message": "Need at least 20 price points"
                }
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Current volatility (last 5 days)
            current_vol = returns.tail(5).std()
            
            # Average volatility (prior 15 days)
            avg_vol = returns.iloc[-20:-5].std()
            
            if avg_vol == 0 or pd.isna(avg_vol):
                return {
                    "signal": "NORMAL",
                    "current_vol": float(current_vol),
                    "avg_vol": 0,
                    "ratio": 0,
                    "message": "Insufficient volatility data"
                }
            
            ratio = current_vol / avg_vol
            
            if ratio >= self.spike_threshold:
                signal = "PAUSE"
                message = f"⚠️ Volatility spike detected ({ratio:.2f}x normal). PAUSE buying."
            else:
                signal = "NORMAL"
                message = f"✅ Volatility normal ({ratio:.2f}x avg)."
            
            return {
                "signal": signal,
                "current_vol": float(current_vol),
                "avg_vol": float(avg_vol),
                "ratio": float(ratio),
                "message": message
            }
            
        except Exception as e:
            LOGGER.error(f"Error in VolDetector: {e}")
            return {
                "signal": "ERROR",
                "current_vol": 0,
                "avg_vol": 0,
                "ratio": 0,
                "message": str(e)
            }
    
    def should_pause_trading(self, prices: pd.Series) -> bool:
        """
        Simple boolean check: should we pause trading due to volatility?
        """
        result = self.check_volatility_spike(prices)
        return result["signal"] == "PAUSE"
