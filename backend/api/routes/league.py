"""
API Routes for Caria Global Portfolio League.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel

from api.dependencies import get_current_user, get_db_connection
from caria.models.auth import UserInDB
from api.domains.social.league import LeagueService

router = APIRouter(prefix="/api/league", tags=["league"])

class LeagueEntry(BaseModel):
    rank: int
    user_id: UUID
    username: str
    score: float
    sharpe_ratio: Optional[float]
    cagr: Optional[float]
    max_drawdown: Optional[float]
    diversification_score: Optional[float]
    account_age_days: Optional[int]

class LeagueProfile(BaseModel):
    current: Optional[LeagueEntry]
    history: List[dict] # Simplified for now

@router.get("/leaderboard", response_model=List[LeagueEntry])
def get_leaderboard(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
):
    """Get the global leaderboard."""
    service = LeagueService()
    try:
        results = service.get_leaderboard(conn, limit, offset)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profile/{user_id}", response_model=LeagueProfile)
def get_user_profile(
    user_id: UUID,
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
):
    """Get a user's league profile."""
    service = LeagueService()
    try:
        profile = service.get_user_profile(conn, user_id)
        
        # Format current entry if exists
        current = None
        if profile.get('current'):
            # Fetch username separately or join in service (service join is better, let's assume service returns raw dict)
            # For MVP, we might need to fetch username if not in 'current'
            # But let's assume the frontend handles basic display or we add username to 'current' query in service
            # Update: Service 'get_user_profile' query for 'current' selects * from league_rankings, 
            # which doesn't have username. We need to fix that or fetch it.
            # Let's do a quick fix here or in service. Service is cleaner.
            # For now, let's just return what we have, frontend might need to fetch user details separately 
            # or we assume 'username' is not strictly required for the profile view of *self* or we fetch it.
            
            # Quick fetch of username
            with conn.cursor() as cur:
                cur.execute("SELECT username FROM users WHERE id = %s", (str(user_id),))
                u_row = cur.fetchone()
                username = u_row[0] if u_row else "Unknown"

            c_row = profile['current']
            current = LeagueEntry(
                rank=c_row['rank'],
                user_id=c_row['user_id'],
                username=username,
                score=float(c_row['score']),
                sharpe_ratio=float(c_row['sharpe_ratio']) if c_row['sharpe_ratio'] else None,
                cagr=float(c_row['cagr']) if c_row['cagr'] else None,
                max_drawdown=float(c_row['max_drawdown']) if c_row['max_drawdown'] else None,
                diversification_score=float(c_row['diversification_score']) if c_row['diversification_score'] else None,
                account_age_days=c_row['account_age_days']
            )
            
        return LeagueProfile(
            current=current,
            history=profile['history']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calculate", status_code=202)
def trigger_calculation(
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
):
    """Trigger daily score calculation (Admin only ideally, but open for demo)."""
    # In prod, check if current_user.is_superuser
    service = LeagueService()
    try:
        service.calculate_daily_scores(conn)
        return {"message": "Calculation triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
