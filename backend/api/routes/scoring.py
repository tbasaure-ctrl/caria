from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_current_user
from caria.models.auth import UserInDB

LOGGER = logging.getLogger("caria.api.scoring")

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
score_router = APIRouter(prefix="/api", tags=["analysis"])

# Use FMP scoring by default, or if USE_FMP_SCORING is set
USE_FMP_SCORING = os.getenv("USE_FMP_SCORING", "true").lower() == "true"

try:
    if USE_FMP_SCORING:
        try:
            from api.services.fmp_scoring_service import FMPScoringService
            scoring_service = FMPScoringService()
            LOGGER.info("Using FMP-based scoring service")
        except ImportError:
            # Fallback to regular ScoringService if FMP version doesn't exist
            LOGGER.warning("FMPScoringService not found, falling back to ScoringService")
            from api.services.scoring_service import ScoringService
            scoring_service = ScoringService()
            LOGGER.info("Using OpenBB-based scoring service (fallback)")
    else:
        from api.services.scoring_service import ScoringService
        scoring_service = ScoringService()
        LOGGER.info("Using OpenBB-based scoring service")
except Exception as e:
    LOGGER.error("Failed to initialize scoring service: %s", e)
    # Final fallback - use ScoringService
    try:
        from api.services.scoring_service import ScoringService
        scoring_service = ScoringService()
        LOGGER.warning("Using ScoringService as final fallback")
    except Exception as e2:
        LOGGER.error("Failed to initialize any scoring service: %s", e2)
        # Don't raise - allow app to start without scoring service
        scoring_service = None