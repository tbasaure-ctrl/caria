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
    hiddenGemScore: float
    classification: str | None
    qualityScore: float
    valuationScore: float
    momentumScore: float
    current_price: float | None = None
    details: dict | None = None
    explanations: dict | None = None


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
                hiddenGemScore=score.get("hiddenGemScore", score["cScore"]),
                classification=score.get("classification"),
                qualityScore=score["qualityScore"],
                valuationScore=score["valuationScore"],
                momentumScore=score["momentumScore"],
                current_price=score.get("current_price"),
                details=score.get("details"),
                explanations=score.get("explanations"),
            )
        )

    results.sort(key=lambda item: item.cScore, reverse=True)
    return CScoreScreenResponse(results=results)


@router.get("/hidden-gems", response_model=CScoreScreenResponse)
def get_hidden_gems(
    limit: int = 10,
    current_user: UserInDB = Depends(get_current_user),
) -> CScoreScreenResponse:
    """
    Discover 'Hidden Gems' using FMP screener and our custom scoring model.
    1. Fetches candidates from FMP (profitable, non-huge, growing).
    2. Scores them with our detailed model.
    3. Returns the top ranked stocks.
    """
    # 1. Fetch Candidates from FMP
    # Criteria: Market Cap > 200M, Beta < 1.5, Dividend > 0 (optional), etc.
    # We want "hidden gems": maybe reasonable valuation?
    try:
        # FMP Screener params
        params = {
            "marketCapMoreThan": 200_000_000,
            "isEtf": "false",
            "isFund": "false",
            "limit": 50, # Get pool of 50
            "exchange": "NASDAQ,NYSE,AMEX"
        }
        candidates = _screener_scoring.fmp.get_stock_screener(params)
    except Exception as e:
        LOGGER.error(f"Failed to fetch from FMP screener: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch candidates from data provider")

    # 2. Score them
    results: List[ScreenerItem] = []
    # We take the first 'limit' * 2 to score, then sort.
    # Scoring is slow (multiple API calls), so we limit the pool we fully analyze.
    
    pool = candidates[: (limit * 2) if limit < 10 else 20] 
    
    for cand in pool:
        ticker = cand.get("symbol")
        if not ticker:
            continue
            
        try:
            score = _screener_scoring.get_scores(ticker)
            if score["cScore"] < 50: # Basic filter
                continue
                
            results.append(
                ScreenerItem(
                    ticker=ticker,
                    cScore=score["cScore"],
                    hiddenGemScore=score.get("hiddenGemScore", score["cScore"]),
                    classification=score.get("classification"),
                    qualityScore=score["qualityScore"],
                    valuationScore=score["valuationScore"],
                    momentumScore=score["momentumScore"],
                    current_price=score.get("current_price"),
                    details=score.get("details"),
                    explanations=score.get("explanations"),
                )
            )
        except Exception:
            continue

    # 3. Sort and limit
    results.sort(key=lambda item: item.hiddenGemScore, reverse=True)
    return CScoreScreenResponse(results=results[:limit])

