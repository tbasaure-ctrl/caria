"""
Analysis Domain Routes.
Per audit document (4.1): Strict domain boundaries.
"""

from fastapi import APIRouter

# Import existing analysis routes (already have their prefixes)
from api.routes.analysis import router as analysis_router
from api.routes.regime import router as regime_router
from api.routes.factors import router as factors_router
from api.routes.valuation import router as valuation_router
from api.routes.model_validation import router as validation_router
from api.routes.ux_tracking import router as ux_tracking_router
from api.routes.risk_reward import router as risk_reward_router
from api.routes.hidden_risk import router as hidden_risk_router

# Combine into analysis domain router
router = APIRouter()
router.include_router(analysis_router)
router.include_router(regime_router)
router.include_router(factors_router)
router.include_router(valuation_router)
router.include_router(validation_router)
router.include_router(ux_tracking_router)  # UX tracking and analytics
router.include_router(risk_reward_router)  # Risk-Reward Engine
router.include_router(hidden_risk_router)  # Hidden Risk AI Report
