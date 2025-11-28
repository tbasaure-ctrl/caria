from __future__ import annotations

import logging
from typing import List, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.services.market_scanner import MarketScannerService
from api.services.scoring_service import ScoringService
from api.services.stock_screener_service import StockScreenerService
from api.services.social_screener_service import SocialScreenerService
from caria.models.auth import UserInDB

LOGGER = logging.getLogger("caria.api.screener")

router = APIRouter(prefix="/api/screener", tags=["analysis"])
_screener_scoring = ScoringService()
_market_scanner_service = MarketScannerService()
_stock_screener_service = StockScreenerService()
_social_screener_service = SocialScreenerService()


# --- OLDER MODELS (Keeping for compatibility) ---
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

# --- NEW MODELS (Fundamental & Social) ---

class FundamentalPick(BaseModel):
    symbol: str
    quality_score: float
    valuation_score: float
    momentum_score: float
    catalyst_score: float
    risk_penalty: float
    c_score: float

class SocialPick(BaseModel):
    symbol: str
    reddit_mentions: int
    stocktwits_bullish: float
    social_score: float
    sentiment_avg: float

class FundamentalScreenResponse(BaseModel):
    picks: List[FundamentalPick]
    timestamp: str

class SocialScreenResponse(BaseModel):
    picks: List[SocialPick]
    timestamp: str


# --- ENDPOINTS ---

@router.post("/cscore", response_model=CScoreScreenResponse)
def screen_cscore(
    payload: CScoreScreenRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> CScoreScreenResponse:
    """Legacy C-Score screener."""
    results: List[ScreenerItem] = []
    for raw_ticker in payload.tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker: continue
        try:
            score = _screener_scoring.get_scores(ticker)
            if payload.minScore is not None and score["cScore"] < payload.minScore:
                continue
            results.append(ScreenerItem(
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
            ))
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", ticker, exc)
            continue
    results.sort(key=lambda item: item.cScore, reverse=True)
    return CScoreScreenResponse(results=results)


@router.post("/run-fundamental", response_model=FundamentalScreenResponse)
def run_fundamental_screen(
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Run the new Fundamental Stock Screener (C-Score v2).
    Filters ~200 large cap stocks and ranks top 3 by Quality, Value, Momentum, Catalyst.
    """
    try:
        picks = _stock_screener_service.run_screen()
        # Convert to model
        formatted_picks = []
        for p in picks:
            formatted_picks.append(FundamentalPick(
                symbol=p['symbol'],
                quality_score=p['quality'],
                valuation_score=p['valuation'],
                momentum_score=p['momentum'],
                catalyst_score=p['catalyst'],
                risk_penalty=p['risk'],
                c_score=p['c_score']
            ))
        
        return FundamentalScreenResponse(
            picks=formatted_picks,
            timestamp="" # You might want to add timestamp to result
        )
    except Exception as e:
        LOGGER.error(f"Fundamental screen failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-social", response_model=SocialScreenResponse)
def run_social_screen(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Run the new Social Sentiment Screener (Under the Radar).
    Detects non-Mag7 stocks with rising social interest on Reddit/StockTwits.
    """
    try:
        picks = _social_screener_service.run_screen()
        formatted_picks = []
        for p in picks:
            formatted_picks.append(SocialPick(
                symbol=p['symbol'],
                reddit_mentions=p['reddit_mentions'],
                stocktwits_bullish=p['stocktwits_bullish'],
                social_score=p['social_score'],
                sentiment_avg=p['sentiment_avg']
            ))
            
        return SocialScreenResponse(
            picks=formatted_picks,
            timestamp=""
        )
    except Exception as e:
        LOGGER.error(f"Social screen failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Market Scanner (Event Driven) ---

class MarketSignal(BaseModel):
    ticker: str
    price: float
    change: float
    rvol: float
    market_cap: float | None = None
    signal_strength: str
    tag: str
    desc: str
    social_spike: dict | None = None

class MarketScannerResponse(BaseModel):
    momentum_signals: List[MarketSignal]
    accumulation_signals: List[MarketSignal]

@router.get("/market-opportunities", response_model=MarketScannerResponse)
def get_market_opportunities(
    current_user: UserInDB = Depends(get_current_user),
) -> MarketScannerResponse:
    """
    Professional Market Scanner: Event-Driven Social Screener
    """
    try:
        results = _market_scanner_service.get_professional_opportunities()
        momentum_signals = [MarketSignal(**signal) for signal in results.get("momentum_signals", [])]
        accumulation_signals = [MarketSignal(**signal) for signal in results.get("accumulation_signals", [])]
        return MarketScannerResponse(momentum_signals=momentum_signals, accumulation_signals=accumulation_signals)
    except Exception as e:
        LOGGER.exception(f"Error running market scanner: {e}")
        raise HTTPException(status_code=500, detail=f"Error running market scanner: {str(e)}")
