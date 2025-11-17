"""
Analysis Domain - Regime Detection, Factors, Valuation, Model Validation.

Strict boundaries: This domain handles all quantitative analysis.
Other domains should not directly access analysis models.
"""

from .routes import router as analysis_router

__all__ = ["analysis_router"]

