"""
Portfolio Domain Routes.
Per audit document (4.1): Strict domain boundaries.
"""

from fastapi import APIRouter

# Import existing portfolio routes (already have their prefixes)
from api.routes.holdings import router as holdings_router
from api.routes.portfolio_analytics import router as analytics_router
from api.routes.tactical_allocation import router as tactical_router
from api.routes.monte_carlo import router as monte_carlo_router

# Combine into portfolio domain router
# Note: Routes already have prefixes, so we include them as-is
router = APIRouter()
router.include_router(holdings_router)
router.include_router(analytics_router)
router.include_router(tactical_router)
router.include_router(monte_carlo_router)

