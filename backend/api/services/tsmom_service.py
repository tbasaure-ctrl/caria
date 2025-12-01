"""
Time Series Momentum (TSMOM) Service.
Implements logic based on Moskowitz, Ooi, and Pedersen (2012).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from api.services.alpha_vantage_client import alpha_vantage_client

LOGGER = logging.getLogger("caria.api.services.tsmom")

class TSMOMService:
    """
    Calculates Time Series Momentum signals.
    """

    def calculate_market_regime(self, symbol: str) -> Dict[str, Any]:
        """
        Calculates TSMOM signal and volatility context.
        
        Logic:
        1. Fetch monthly adjusted prices (12+ months).
        2. Calculate excess return r(t-12, t).
        3. Calculate ex-ante volatility sigma(t) (using recent monthly returns).
        4. Determine trend direction and risk context.
        """
        try:
            # 1. Data Ingestion
            monthly_data = alpha_vantage_client.get_monthly_adjusted_prices(symbol)
            
            if not monthly_data or len(monthly_data) < 13:
                return {
                    "status": "insufficient_data",
                    "trend_direction": "Neutral",
                    "trend_strength_12m": 0.0,
                    "volatility_context": "Unknown",
                    "raw_data": {}
                }

            # Convert to DataFrame for easier calculation
            df = pd.DataFrame(monthly_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Ensure we have enough data
            if len(df) < 13:
                 return {
                    "status": "insufficient_data",
                    "trend_direction": "Neutral",
                    "trend_strength_12m": 0.0,
                    "volatility_context": "Unknown",
                    "raw_data": {}
                }

            # Get current price (t) and price 12 months ago (t-12)
            # Using adjusted close to account for dividends/splits
            current_price = df.iloc[-1]['adjusted_close']
            past_price = df.iloc[-13]['adjusted_close']
            
            # 2. The Signal (Trend): Excess Return (r_t-12, t)
            # Assuming risk-free rate is negligible for this signal calculation or using simple return
            trend_return = (current_price / past_price) - 1
            
            # 3. The Context (Volatility): Ex-ante volatility (sigma_t)
            # Calculate monthly returns for the last 12 months
            df['monthly_return'] = df['adjusted_close'].pct_change()
            last_12m_returns = df['monthly_return'].tail(12)
            
            # Standard deviation of monthly returns * sqrt(12) for annualized vol
            annualized_vol = last_12m_returns.std() * np.sqrt(12)
            
            # 4. Output Logic
            trend_direction = "Bullish" if trend_return > 0 else "Bearish"
            
            # Volatility Context Thresholds (Subjective/Based on market norms)
            # Low < 15%, Medium 15-30%, High > 30%
            if annualized_vol < 0.15:
                vol_context = "Low"
            elif annualized_vol < 0.30:
                vol_context = "Medium"
            else:
                vol_context = "High"
            
            # High Risk Flag: Positive Trend but High Volatility
            risk_flag = False
            if trend_direction == "Bullish" and vol_context == "High":
                risk_flag = True
                trend_direction = "High Risk Bullish" # Modifying label for UI clarity

            return {
                "status": "success",
                "symbol": symbol,
                "trend_direction": trend_direction,
                "trend_strength_12m": float(trend_return),
                "volatility_context": vol_context,
                "annualized_volatility": float(annualized_vol),
                "risk_flag": risk_flag,
                "raw_data": {
                    "current_price": float(current_price),
                    "price_t_minus_12": float(past_price),
                    "last_12_returns": last_12m_returns.tolist() if not last_12m_returns.empty else []
                }
            }

        except Exception as e:
            LOGGER.error(f"Error calculating TSMOM for {symbol}: {e}")
            return {
                "status": "error",
                "error_detail": str(e),
                "trend_direction": "Neutral",
                "trend_strength_12m": 0.0,
                "volatility_context": "Unknown",
                "raw_data": {}
            }

tsmom_service = TSMOMService()

