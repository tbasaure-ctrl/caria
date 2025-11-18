"""
Asset Regime Classification Service.
Classifies assets by their historical performance in different economic regimes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from caria.ingestion.clients.fmp_client import FMPClient

LOGGER = logging.getLogger("caria.api.asset_regime")

# Historical regime periods (approximate)
REGIME_PERIODS = {
    "expansion": [
        ("2010-01-01", "2011-06-30"),  # Post-2008 recovery
        ("2012-01-01", "2015-12-31"),  # Mid-2010s expansion
        ("2016-07-01", "2019-12-31"),  # Late 2010s expansion
        ("2020-05-01", "2021-12-31"),  # Post-COVID recovery
    ],
    "recession": [
        ("2008-01-01", "2009-06-30"),  # Financial crisis
        ("2020-03-01", "2020-04-30"),  # COVID crash
    ],
    "slowdown": [
        ("2011-07-01", "2011-12-31"),  # 2011 slowdown
        ("2015-08-01", "2016-06-30"),  # 2015-2016 slowdown
        ("2018-10-01", "2018-12-31"),  # Late 2018 volatility
        ("2022-01-01", "2022-12-31"),  # 2022 slowdown
    ],
    "stress": [
        ("2008-09-01", "2009-03-31"),  # Financial crisis peak
        ("2020-02-20", "2020-03-31"),  # COVID crash peak
    ],
}


class AssetRegimeService:
    """Service to classify assets by regime suitability."""

    def __init__(self):
        """Initialize the service."""
        self.fmp_client = None
        try:
            self.fmp_client = FMPClient()
        except Exception as e:
            LOGGER.warning(f"FMPClient not available: {e}. Will use yfinance fallback.")

    def classify_asset(
        self, ticker: str, use_cache: bool = True
    ) -> Dict[str, float]:
        """
        Classify an asset by regime suitability.
        
        Returns scores (0-1) for each regime indicating how well the asset
        performs in that regime. Higher score = better performance.
        
        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached classifications (future enhancement)
            
        Returns:
            Dict with regime scores: {"expansion": 0.8, "recession": 0.3, ...}
        """
        try:
            # Get historical price data
            stock = yf.Ticker(ticker.upper())
            hist = stock.history(period="10y")  # 10 years of data
            
            if hist.empty:
                LOGGER.warning(f"No historical data for {ticker}")
                return self._get_default_scores()
            
            # Calculate returns for each regime period
            regime_scores = {}
            
            for regime_name, periods in REGIME_PERIODS.items():
                returns = []
                for start_date, end_date in periods:
                    period_data = hist.loc[start_date:end_date]
                    if not period_data.empty:
                        period_return = (
                            period_data["Close"].iloc[-1] / period_data["Close"].iloc[0] - 1
                        )
                        returns.append(period_return)
                
                if returns:
                    # Average return during this regime
                    avg_return = sum(returns) / len(returns)
                    # Normalize to 0-1 score (assuming -50% to +50% range)
                    score = max(0.0, min(1.0, (avg_return + 0.5) / 1.0))
                    regime_scores[regime_name] = score
                else:
                    # No data for this regime, use default
                    regime_scores[regime_name] = 0.5
            
            # Ensure all regimes have scores
            for regime in ["expansion", "slowdown", "recession", "stress"]:
                if regime not in regime_scores:
                    regime_scores[regime] = 0.5
            
            return regime_scores
            
        except Exception as e:
            LOGGER.exception(f"Error classifying asset {ticker}: {e}")
            return self._get_default_scores()
    
    def classify_assets_batch(
        self, tickers: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Classify multiple assets at once.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dict mapping ticker to regime scores
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.classify_asset(ticker)
        return results
    
    def _get_default_scores(self) -> Dict[str, float]:
        """Return default regime scores when classification fails."""
        return {
            "expansion": 0.5,
            "slowdown": 0.5,
            "recession": 0.5,
            "stress": 0.5,
        }
    
    def get_regime_suitability(
        self, ticker: str, target_regime: str
    ) -> float:
        """
        Get suitability score for a specific regime.
        
        Args:
            ticker: Stock ticker
            target_regime: Target regime ("expansion", "recession", etc.)
            
        Returns:
            Suitability score (0-1)
        """
        scores = self.classify_asset(ticker)
        return scores.get(target_regime, 0.5)


def get_asset_regime_service() -> AssetRegimeService:
    """Get singleton instance of AssetRegimeService."""
    if not hasattr(get_asset_regime_service, "_instance"):
        get_asset_regime_service._instance = AssetRegimeService()
    return get_asset_regime_service._instance

