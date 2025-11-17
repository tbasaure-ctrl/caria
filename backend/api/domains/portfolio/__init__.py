"""
Portfolio Domain - Holdings, Analytics, Tactical Allocation, Monte Carlo.

Strict boundaries: This domain handles all portfolio-related operations.
Other domains should not directly manipulate portfolio data.
"""

from .routes import router as portfolio_router

__all__ = ["portfolio_router"]

