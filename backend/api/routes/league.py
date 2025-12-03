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

class JoinLeagueRequest(BaseModel):
    is_anonymous: bool = False
    display_name: Optional[str] = None

@router.post("/join")
def join_league(
    request: JoinLeagueRequest,
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
):
    """Join the league with optional anonymity."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Check if table exists first
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'league_participants'
                );
            """)
            table_exists = cur.fetchone()[0]
            
            if not table_exists:
                logger.error("league_participants table does not exist. Migration required.")
                raise HTTPException(
                    status_code=500, 
                    detail="League feature not available. Database migration required. See LEAGUE_MIGRATION.md or run: POST /api/league/migrate?secret_key=YOUR_SECRET_KEY"
                )
            
            # Insert or update participant
            cur.execute("""
                INSERT INTO league_participants (user_id, is_anonymous, display_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET is_anonymous = EXCLUDED.is_anonymous,
                    display_name = EXCLUDED.display_name,
                    updated_at = CURRENT_TIMESTAMP
            """, (str(current_user.id), request.is_anonymous, request.display_name))
        conn.commit()
        return {"message": "Successfully joined the league", "is_anonymous": request.is_anonymous}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error joining league: {e}", exc_info=True)
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to join league: {str(e)}")

@router.get("/participation-status")
def get_participation_status(
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
):
    """Check if user has joined the league."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT is_anonymous, display_name, joined_at
                FROM league_participants
                WHERE user_id = %s
            """, (str(current_user.id),))
            row = cur.fetchone()
            if row:
                return {
                    "has_joined": True,
                    "is_anonymous": row[0],
                    "display_name": row[1],
                    "joined_at": row[2]
                }
            return {"has_joined": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/leaderboard", response_model=List[LeagueEntry])
def get_leaderboard(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
):
    """Get the global leaderboard (only participants)."""
    service = LeagueService()
    try:
        from psycopg2.extras import RealDictCursor
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get latest date
            cur.execute("SELECT MAX(date) as max_date FROM league_rankings")
            row = cur.fetchone()
            if not row or not row['max_date']:
                return []
            
            latest_date = row['max_date']
            
            # Join with participants and mask anonymous users
            cur.execute("""
                SELECT 
                    lr.rank,
                    u.id as user_id,
                    CASE 
                        WHEN lp.is_anonymous = TRUE THEN 
                            COALESCE(lp.display_name, 'Anonymous Investor')
                        ELSE u.username
                    END as username,
                    lr.score,
                    lr.sharpe_ratio,
                    lr.cagr,
                    lr.max_drawdown,
                    lr.diversification_score,
                    lr.account_age_days
                FROM league_rankings lr
                JOIN users u ON lr.user_id = u.id
                JOIN league_participants lp ON lr.user_id = lp.user_id
                WHERE lr.date = %s
                ORDER BY lr.rank ASC
                LIMIT %s OFFSET %s
            """, (latest_date, limit, offset))
            
            return cur.fetchall()
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

@router.post("/migrate")
def run_migration(
    secret_key: str = Query(..., description="Secret key to authorize migration"),
    conn = Depends(get_db_connection),
):
    """Run league_participants table migration. Requires secret key."""
    import os
    import logging
    logger = logging.getLogger(__name__)
    
    # Check secret key (use environment variable or default for development)
    expected_key = os.getenv("MIGRATION_SECRET_KEY", "dev-migration-key-change-in-prod")
    if secret_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid migration secret key")
    
    migration_sql = """
    -- Create league_participants table to track opt-in status and anonymity
    CREATE TABLE IF NOT EXISTS league_participants (
        user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        is_anonymous BOOLEAN DEFAULT FALSE,
        display_name TEXT, -- Optional custom name, otherwise use username or "Anonymous"
        joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Index for faster joins
    CREATE INDEX IF NOT EXISTS idx_league_participants_joined ON league_participants(joined_at);
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(migration_sql)
        conn.commit()
        logger.info("League participants migration applied successfully")
        return {
            "message": "Migration applied successfully",
            "table": "league_participants",
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Error applying migration: {e}", exc_info=True)
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")

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
