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
        from api.services.fmp_scoring_service import FMPScoringService
        scoring_service = FMPScoringService()
        LOGGER.info("Using FMP-based scoring service")
    else:
        from api.services.scoring_service import ScoringService
        scoring_service = ScoringService()
        LOGGER.info("Using OpenBB-based scoring service")
except Exception as e:
    LOGGER.error("Failed to initialize scoring service: %s", e)
    # Try fallback
    try:
        from api.services.fmp_scoring_service import FMPScoringService
        scoring_service = FMPScoringService()
        LOGGER.warning("Fell back to FMP scoring service")
    except Exception as e2:
        LOGGER.error("Failed to initialize FMP scoring service: %s", e2)
        raise