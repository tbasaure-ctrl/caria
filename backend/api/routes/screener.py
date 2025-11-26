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
        # FMP Screener params - relaxed for better results
        params = {
            "marketCapMoreThan": 50_000_000,      # Lowered from 200M to 50M
            "marketCapLowerThan": 10_000_000_000, # Add: < 10B (mid-caps)
            "isActivelyTrading": "true",          # Add: actively trading only
            "isEtf": "false",
            "isFund": "false",
            "limit": 100,                         # Increased from 50 to 100
            "exchange": "NASDAQ,NYSE,AMEX"
        }
        candidates = _screener_scoring.fmp.get_stock_screener(params)
    except Exception as e:
        LOGGER.error(f"Failed to fetch from FMP screener: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch candidates from data provider")

    # Fallback ticker list if FMP returns insufficient results
    if not candidates or len(candidates) < 5:
        LOGGER.warning(f"FMP returned only {len(candidates) if candidates else 0} candidates, using fallback list")

        # Fallback: predefined mid-cap value stocks
        fallback_tickers = [
            "INTC", "F", "WBA", "KSS", "NWL", "GPS", "M", "JWN",
            "AAL", "UAL", "LUV", "DAL", "HA", "ALK", "JBLU",
            "UAA", "FL", "DDS", "BBWI", "RL", "PVH"
        ]
        LOGGER.info(f"Using fallback list: {len(fallback_tickers)} stocks")
        candidates = [{"symbol": t} for t in fallback_tickers]

    # 2. Score them
    results: List[ScreenerItem] = []
    # We take a larger pool to score, then sort.
    # Scoring is slow (multiple API calls), so we limit the pool we fully analyze.

    pool_size = min(limit * 3, 50)  # 3x limit, max 50
    pool = candidates[:pool_size]
    LOGGER.info(f"Screening {len(pool)} candidates from {len(candidates)} total")

    min_threshold = 35  # Lowered from 50 to 35
    scored_count = 0
    filtered_count = 0

    for cand in pool:
        ticker = cand.get("symbol")
        if not ticker:
            continue

        try:
            score = _screener_scoring.get_scores(ticker)
            scored_count += 1

            LOGGER.debug(f"{ticker}: cScore={score['cScore']:.1f}, Q={score['qualityScore']:.1f}, V={score['valuationScore']:.1f}")

            if score["cScore"] < min_threshold:  # Lowered threshold
                filtered_count += 1
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
        except Exception as e:
            LOGGER.warning(f"Failed to score {ticker}: {e}")
            continue

    # 3. Sort and limit
    results.sort(key=lambda item: item.hiddenGemScore, reverse=True)
    LOGGER.info(f"Screener complete: {scored_count} scored, {filtered_count} filtered out, {len(results)} passed threshold")
    return CScoreScreenResponse(results=results[:limit])

