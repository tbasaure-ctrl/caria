"""
Weekly Alpha Picks - Automated system that generates and stores 3 stocks per week.
Same picks shown to all users with investment thesis.
"""
from fastapi import APIRouter, Depends, Request, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging

from api.dependencies import get_current_user, get_optional_current_user
from api.services.alpha_service import AlphaService
from api.routes.factors import _guard_factor_service

router = APIRouter(prefix="/api/weekly-picks", tags=["weekly-picks"])
LOGGER = logging.getLogger("caria.api.weekly_picks")

# In-memory storage (in production, use database)
_weekly_picks_cache: Dict[str, Any] = {}
_WEEKLY_PICKS_KEY = "current_weekly_picks"

class WeeklyPick(BaseModel):
    ticker: str
    company_name: str
    sector: str
    cas_score: float
    scores: Dict[str, float]
    investment_thesis: str  # Brief investment thesis
    generated_date: str

class WeeklyPicksResponse(BaseModel):
    picks: List[WeeklyPick]
    week_start: str
    generated_date: str

def _generate_weekly_picks(request: Request) -> List[Dict[str, Any]]:
    """Generate weekly picks using AlphaService."""
    factor_service = _guard_factor_service(request)
    alpha_service = AlphaService(factor_service)
    
    try:
        picks = alpha_service.compute_alpha_picks(top_n_candidates=100)
        
        # Enhance with investment thesis
        enhanced_picks = []
        for pick in picks[:3]:  # Top 3 only
            thesis = _generate_investment_thesis(pick)
            enhanced_picks.append({
                **pick,
                "investment_thesis": thesis,
                "generated_date": datetime.utcnow().isoformat()
            })
        
        return enhanced_picks
    except Exception as e:
        LOGGER.exception(f"Error generating weekly picks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate weekly picks: {str(e)}")

def _generate_investment_thesis(pick: Dict[str, Any]) -> str:
    """Generate a brief investment thesis based on the pick's scores."""
    ticker = pick.get("ticker", "")
    scores = pick.get("scores", {})
    explanation = pick.get("explanation", "")
    
    quality = scores.get("quality", 50)
    valuation = scores.get("valuation", 50)
    momentum = scores.get("momentum", 50)
    catalyst = scores.get("catalyst", 50)
    
    thesis_parts = []
    
    # Primary strength
    if quality >= 70:
        thesis_parts.append(f"{ticker} demonstrates exceptional financial quality with strong profitability and operational efficiency.")
    elif valuation >= 70:
        thesis_parts.append(f"{ticker} trades at an attractive valuation relative to its fundamentals and peer group.")
    elif momentum >= 70:
        thesis_parts.append(f"{ticker} shows strong price momentum and positive market sentiment.")
    else:
        thesis_parts.append(f"{ticker} offers a balanced investment profile across key factors.")
    
    # Secondary factors
    if quality >= 60 and valuation >= 60:
        thesis_parts.append("The combination of solid fundamentals and reasonable pricing creates a compelling risk-reward opportunity.")
    elif momentum >= 60 and catalyst >= 60:
        thesis_parts.append("Recent momentum is supported by positive near-term catalysts that could drive further appreciation.")
    
    # Risk note
    if valuation < 40:
        thesis_parts.append("Note: Valuation appears stretched; consider position sizing accordingly.")
    
    return " ".join(thesis_parts) if thesis_parts else explanation

@router.get("/current", response_model=WeeklyPicksResponse)
def get_current_weekly_picks(
    request: Request,
    current_user = Depends(get_optional_current_user)
):
    """
    Get the current week's picks. Auto-generates if none exist or if week has changed.
    Same picks shown to all users.
    """
    # Check if we have cached picks for this week
    today = datetime.utcnow()
    week_start = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
    
    cached = _weekly_picks_cache.get(_WEEKLY_PICKS_KEY)
    cached_week = cached.get("week_start") if cached else None
    
    # Generate new picks if:
    # 1. No cached picks exist
    # 2. Week has changed
    # 3. Cache is older than 7 days
    should_regenerate = (
        not cached or 
        cached_week != week_start or
        (cached.get("generated_date") and 
         (today - datetime.fromisoformat(cached["generated_date"])).days >= 7)
    )
    
    if should_regenerate:
        LOGGER.info(f"Generating new weekly picks for week starting {week_start}")
        picks_data = _generate_weekly_picks(request)
        
        _weekly_picks_cache[_WEEKLY_PICKS_KEY] = {
            "week_start": week_start,
            "generated_date": datetime.utcnow().isoformat(),
            "picks": picks_data
        }
    
    cached = _weekly_picks_cache.get(_WEEKLY_PICKS_KEY, {})
    picks_data = cached.get("picks", [])
    
    return WeeklyPicksResponse(
        picks=[WeeklyPick(**p) for p in picks_data],
        week_start=cached.get("week_start", week_start),
        generated_date=cached.get("generated_date", datetime.utcnow().isoformat())
    )

@router.post("/regenerate")
def regenerate_weekly_picks(
    request: Request,
    current_user = Depends(get_current_user)
):
    """
    Force regenerate weekly picks (admin function).
    """
    LOGGER.info("Force regenerating weekly picks")
    picks_data = _generate_weekly_picks(request)
    
    today = datetime.utcnow()
    week_start = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
    
    _weekly_picks_cache[_WEEKLY_PICKS_KEY] = {
        "week_start": week_start,
        "generated_date": datetime.utcnow().isoformat(),
        "picks": picks_data
    }
    
    return {
        "message": "Weekly picks regenerated",
        "picks_count": len(picks_data),
        "week_start": week_start
    }
