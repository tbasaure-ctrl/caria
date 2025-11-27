from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.services.market_scanner import MarketScannerService
from api.services.scoring_service import ScoringService
from api.services.under_the_radar_screener_service import UnderTheRadarScreenerService
from caria.models.auth import UserInDB

LOGGER = logging.getLogger("caria.api.screener")

router = APIRouter(prefix="/api/screener", tags=["analysis"])
_screener_scoring = ScoringService()
_under_the_radar_service = UnderTheRadarScreenerService()
_market_scanner_service = MarketScannerService()


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
        LOGGER.warning(f"FMP screener failed: {e}. Using fallback list...")
        # Don't raise error, use fallback instead
        candidates = None

    # Fallback ticker list if FMP returns insufficient results or fails
    if not candidates or len(candidates) < 5:
        LOGGER.info(f"Using fallback list (FMP returned {len(candidates) if candidates else 0} candidates)")

        # Fallback: predefined mid-cap value stocks (under the radar)
        fallback_tickers = [
            "INTC", "F", "WBA", "KSS", "NWL", "GPS", "M", "JWN",
            "AAL", "UAL", "LUV", "DAL", "HA", "ALK", "JBLU",
            "UAA", "FL", "DDS", "BBWI", "RL", "PVH", "TGT", "HD",
            "LOW", "NKE", "SBUX", "CMG", "DPZ", "YUM"
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


# ---------------------------------------------------------------------
# Under-the-Radar Screener
# ---------------------------------------------------------------------

class UnderTheRadarCandidate(BaseModel):
    ticker: str
    name: str
    sector: str
    social_spike: dict
    catalysts: dict
    quality_metrics: dict
    liquidity: dict
    explanation: str


class UnderTheRadarResponse(BaseModel):
    candidates: List[UnderTheRadarCandidate]
    message: str | None = None


@router.get("/under-the-radar", response_model=UnderTheRadarResponse)
def get_under_the_radar_screener(
    current_user: UserInDB = Depends(get_current_user),
) -> UnderTheRadarResponse:
    """
    Under-the-Radar Screener: Detects small-cap stocks with social momentum,
    recent catalysts, and quality metrics.
    
    Returns 0-3 high-conviction candidates per week.
    """
    try:
        candidates = _under_the_radar_service.screen()
        
        if not candidates:
            return UnderTheRadarResponse(
                candidates=[],
                message="No stocks passed all filters this week. The screener looks for: "
                        "social momentum spikes (2+ sources), recent catalysts, quality metrics "
                        "(ROCE improvement, FCF yield), and size/liquidity (50M-800M market cap, volume spike)."
            )
        
        return UnderTheRadarResponse(
            candidates=[UnderTheRadarCandidate(**c) for c in candidates],
            message=f"Found {len(candidates)} under-the-radar candidate(s)"
        )
    except Exception as e:
        LOGGER.exception(f"Error running under-the-radar screener: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error running screener: {str(e)}"
        )


# ---------------------------------------------------------------------
# Professional Market Scanner (Event-Driven Social Screener)
# ---------------------------------------------------------------------

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
    
    Invierte el flujo tradicional:
    1. Busca anomalías de precio primero (gainers, most active)
    2. Filtra mega-caps (>50B-100B)
    3. Valida con ruido social después (spike ratio)
    
    Devuelve dos tipos de señales:
    - momentum_signals: Precio subiendo fuerte (>5%) con volumen alto (>1.2x RVol)
    - accumulation_signals: Volumen muy alto (>1.8x RVol) pero precio comprimido (-1.5% a +2.5%)
    """
    try:
        results = _market_scanner_service.get_professional_opportunities()
        
        # Convertir a modelos Pydantic
        momentum_signals = [
            MarketSignal(**signal) for signal in results.get("momentum_signals", [])
        ]
        accumulation_signals = [
            MarketSignal(**signal) for signal in results.get("accumulation_signals", [])
        ]
        
        return MarketScannerResponse(
            momentum_signals=momentum_signals,
            accumulation_signals=accumulation_signals
        )
    except Exception as e:
        LOGGER.exception(f"Error running market scanner: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error running market scanner: {str(e)}"
        )

