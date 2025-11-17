"""
Domain modules per audit document (4.1).
Modular monolith architecture with strict domain boundaries.

Domains:
- identity: Authentication, users, sessions
- portfolio: Holdings, analytics, tactical allocation, Monte Carlo
- social: Community posts, chat
- analysis: Regime detection, factors, valuation, model validation
- market_data: Real-time prices, market indicators
"""

__all__ = [
    "identity",
    "portfolio",
    "social",
    "analysis",
    "market_data",
]

