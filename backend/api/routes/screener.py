from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.services.scoring_service import ScoringService
from caria.models.auth import UserInDB

LOGGER = logging.getLogger("caria.api.screener")

router = APIRouter(prefix="/api/screener", tags=["analysis"])
_screener_scoring = ScoringService()


class CScoreScreenRequest(BaseModel):
    tickers: List[str] = Field(..., min_items=1, max_items=25)
    minScore: float | None = Field(default=None, description="Optional minimum C-Score filter (0-100)")


class ScreenerItem(BaseModel):
    ticker: str
    cScore: float
    classification: str | None
    qualityScore: float
    valuationScore: float
    momentumScore: float
    qualitativeMoatScore: float | None = None


class CScoreScreenResponse(BaseModel):
    results: List[ScreenerItem]


@router.post("/cscore", response_model=CScoreScreenResponse)
def screen_cscore(
    payload: CScoreScreenRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> CScoreScreenResponse:
    """Screen a list of tickers and rank them by C-Score."""
    results: List[ScreenerItem] = []
    for raw_ticker in payload.tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker:
            continue
        try:
            score = _screener_scoring.get_scores(ticker)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping %s in screener due to error: %s", ticker, exc)
            continue

        if payload.minScore is not None and score["cScore"] < payload.minScore:
            continue

        results.append(
            ScreenerItem(
                ticker=ticker,
                cScore=score["cScore"],
                classification=score.get("classification"),
                qualityScore=score["qualityScore"],
                valuationScore=score["valuationScore"],
                momentumScore=score["momentumScore"],
                qualitativeMoatScore=score.get("qualitativeMoatScore"),
            )
        )

    results.sort(key=lambda item: item.cScore, reverse=True)
    return CScoreScreenResponse(results=results)

