"""
Market Data Domain Routes.
Per audit document (4.1): Strict domain boundaries.
"""

from fastapi import APIRouter

# Import existing market data routes (already has /api/prices prefix)
from api.routes.prices import router as prices_router

# Combine into market data domain router
router = APIRouter()
router.include_router(prices_router)

