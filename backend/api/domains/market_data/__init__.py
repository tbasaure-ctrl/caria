"""
Market Data Domain - Real-time Prices and Market Indicators.

Strict boundaries: This domain handles all market data operations.
Other domains should not directly access market data sources.
"""

from .routes import router as market_data_router

__all__ = ["market_data_router"]

