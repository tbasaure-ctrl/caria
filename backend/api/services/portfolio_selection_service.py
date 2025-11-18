"""
Portfolio Selection Service.
Selects portfolios using the model (outliers/balanced, 10-20 holdings) for validation.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, List, Literal, Optional

import pandas as pd

from caria.clients.fmp_client import FMPClient
from caria.models.regime.hmm_regime_detector import RegimeState

LOGGER = logging.getLogger("caria.services.portfolio_selection")


class PortfolioSelectionService:
    """
    Service to select portfolios using the model for validation.
    
    Selection types:
    - outlier: Portfolios with unusual allocations (for testing edge cases)
    - balanced: Well-diversified portfolios (for testing normal cases)
    - random: Random selection (for baseline comparison)
    """

    def __init__(self, fmp_client: FMPClient) -> None:
        self.fmp_client = fmp_client
        # Common ETFs and stocks for portfolio construction
        self.common_holdings = {
            "etfs": [
                "SPY", "QQQ", "VTI", "VOO", "VEA", "VWO",  # Broad market
                "XLF", "XLK", "XLV", "XLE", "XLI", "XLP",  # Sector ETFs
                "GLD", "SLV", "TLT", "SHY", "HYG",  # Commodities/Bonds
            ],
            "stocks": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",  # Tech
                "JPM", "BAC", "WFC", "GS",  # Financials
                "JNJ", "PFE", "UNH", "ABBV",  # Healthcare
                "XOM", "CVX", "COP",  # Energy
                "WMT", "HD", "MCD", "NKE",  # Consumer
            ],
        }

    def select_portfolio(
        self,
        selection_type: Literal["outlier", "balanced", "random"],
        num_holdings: int = 15,
        regime: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Select a portfolio based on the selection type.
        
        Args:
            selection_type: Type of portfolio selection (outlier, balanced, random)
            num_holdings: Number of holdings (10-20)
            regime: Optional regime to consider for selection
            
        Returns:
            Dictionary with portfolio data including holdings and allocations
        """
        if num_holdings < 10 or num_holdings > 20:
            raise ValueError("num_holdings must be between 10 and 20")

        if selection_type == "random":
            return self._select_random_portfolio(num_holdings)
        elif selection_type == "balanced":
            return self._select_balanced_portfolio(num_holdings, regime)
        elif selection_type == "outlier":
            return self._select_outlier_portfolio(num_holdings, regime)
        else:
            raise ValueError(f"Unknown selection_type: {selection_type}")

    def _select_random_portfolio(self, num_holdings: int) -> Dict[str, any]:
        """Select a random portfolio."""
        all_holdings = self.common_holdings["etfs"] + self.common_holdings["stocks"]
        selected = random.sample(all_holdings, min(num_holdings, len(all_holdings)))
        
        # Random allocations that sum to 100%
        allocations = [random.random() for _ in selected]
        total = sum(allocations)
        allocations = [(a / total) * 100 for a in allocations]
        
        holdings = [
            {"ticker": ticker, "allocation": round(alloc, 2)}
            for ticker, alloc in zip(selected, allocations)
        ]
        
        return {
            "selection_type": "random",
            "holdings": holdings,
            "total_holdings": len(holdings),
            "regime": None,
        }

    def _select_balanced_portfolio(self, num_holdings: int, regime: Optional[str]) -> Dict[str, any]:
        """
        Select a balanced, well-diversified portfolio.
        Considers regime if provided.
        """
        # For balanced, we want diversification across asset classes
        etf_count = max(3, num_holdings // 2)  # At least half ETFs
        stock_count = num_holdings - etf_count
        
        selected_etfs = random.sample(self.common_holdings["etfs"], min(etf_count, len(self.common_holdings["etfs"])))
        selected_stocks = random.sample(self.common_holdings["stocks"], min(stock_count, len(self.common_holdings["stocks"])))
        
        selected = selected_etfs + selected_stocks
        
        # Balanced allocations (more equal distribution)
        base_allocation = 100.0 / num_holdings
        allocations = [base_allocation] * num_holdings
        
        # Add slight variation
        for i in range(num_holdings):
            allocations[i] += random.uniform(-2, 2)
        
        # Normalize to sum to 100%
        total = sum(allocations)
        allocations = [(a / total) * 100 for a in allocations]
        
        holdings = [
            {"ticker": ticker, "allocation": round(alloc, 2)}
            for ticker, alloc in zip(selected, allocations)
        ]
        
        return {
            "selection_type": "balanced",
            "holdings": holdings,
            "total_holdings": len(holdings),
            "regime": regime,
        }

    def _select_outlier_portfolio(self, num_holdings: int, regime: Optional[str]) -> Dict[str, any]:
        """
        Select an outlier portfolio with unusual allocations.
        Useful for testing edge cases and model robustness.
        """
        # For outliers, we might have:
        # - Concentrated positions (few holdings with large allocations)
        # - Unusual sector/asset class mix
        # - Extreme allocations
        
        selected = random.sample(
            self.common_holdings["etfs"] + self.common_holdings["stocks"],
            min(num_holdings, len(self.common_holdings["etfs"] + self.common_holdings["stocks"])),
        )
        
        # Create outlier allocations: some very large, some very small
        allocations = []
        # First 2-3 holdings get 60-70% total
        top_count = min(3, num_holdings)
        top_allocation = random.uniform(60, 70)
        for i in range(top_count):
            allocations.append(random.uniform(top_allocation / top_count * 0.8, top_allocation / top_count * 1.2))
        
        # Remaining holdings split the rest
        remaining = 100.0 - sum(allocations)
        remaining_count = num_holdings - top_count
        if remaining_count > 0:
            base_remaining = remaining / remaining_count
            for i in range(remaining_count):
                allocations.append(base_remaining * random.uniform(0.5, 1.5))
        
        # Normalize
        total = sum(allocations)
        allocations = [(a / total) * 100 for a in allocations]
        
        holdings = [
            {"ticker": ticker, "allocation": round(alloc, 2)}
            for ticker, alloc in zip(selected, allocations)
        ]
        
        return {
            "selection_type": "outlier",
            "holdings": holdings,
            "total_holdings": len(holdings),
            "regime": regime,
        }


def get_portfolio_selection_service() -> PortfolioSelectionService:
    """Get PortfolioSelectionService instance."""
    from caria.clients.fmp_client import FMPClient
    
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        LOGGER.warning("FMP_API_KEY not configured, PortfolioSelectionService may have limited functionality")
    
    fmp_client = FMPClient(api_key=api_key)
    return PortfolioSelectionService(fmp_client)

