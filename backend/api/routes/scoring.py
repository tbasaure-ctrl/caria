from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_current_user
from api.services.scoring_service import ScoringService
from caria.models.auth import UserInDB

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
score_router = APIRouter(prefix="/api", tags=["analysis"])

scoring_service = ScoringService()


class ScoringDetails(BaseModel):
    quality: dict
    valuation: dict
    momentum: dict


class ScoringResponse(BaseModel):
    ticker: str
    qualityScore: float
    valuationScore: float
    momentumScore: float
    qualitativeMoatScore: float | None = None
    cScore: float
    classification: str | None = None
    current_price: float
    fair_value: float | None
    valuation_upside_pct: float | None
    details: ScoringDetails


@router.get("/scoring/{ticker}", response_model=ScoringResponse)
def get_scoring(
    ticker: str,
    current_user: UserInDB = Depends(get_current_user),
) -> ScoringResponse:
    """Devuelve los puntajes Quality / Valuation / Momentum para un ticker."""
    try:
        result = scoring_service.get_scores(ticker.upper())
        return ScoringResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error calculando puntajes: {exc}") from exc


@score_router.get("/score/{ticker}", response_model=ScoringResponse)
def get_scoring_alias(
    ticker: str,
    current_user: UserInDB = Depends(get_current_user),
) -> ScoringResponse:
    return get_scoring(ticker=ticker, current_user=current_user)
