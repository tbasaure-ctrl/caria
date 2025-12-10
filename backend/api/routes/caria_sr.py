
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import date
from pydantic import BaseModel

from api.dependencies import get_db_connection

router = APIRouter(prefix="/caria-sr", tags=["caria-sr"])

class SRSeriesPoint(BaseModel):
    date: str
    e4: float
    sync: float
    sr: float
    regime: int

class SRStatus(BaseModel):
    ticker: str
    auc: float
    mean_normal: float
    mean_fragile: float
    last_date: str
    last_sr: float
    last_regime: int
    last_updated: str

@router.get("/series/{ticker}", response_model=List[SRSeriesPoint])
def get_sr_series(
    ticker: str,
    conn = Depends(get_db_connection)
):
    """Get the calculated SR series for a ticker."""
    try:
        from psycopg2.extras import RealDictCursor
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT date, e4, sync, sr, regime
                FROM caria_sr_daily
                WHERE ticker = %s
                ORDER BY date ASC;
            """, (ticker.upper(),))
            rows = cur.fetchall()
            
        if not rows:
            # raise HTTPException(status_code=404, detail="Ticker not found (run job first)")
            return [] # Return empty list if no data yet to handle gracefully

        return [
            {
                "date": r["date"].isoformat(),
                "e4": r["e4"],
                "sync": r["sync"],
                "sr": r["sr"],
                "regime": int(r["regime"])
            } for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{ticker}", response_model=SRStatus)
def get_sr_status(
    ticker: str,
    conn = Depends(get_db_connection)
):
    """Get current regime status and aggregate stats."""
    try:
        from psycopg2.extras import RealDictCursor
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Stats
            cur.execute("""
                SELECT auc, mean_normal, mean_fragile, last_updated
                FROM caria_sr_stats
                WHERE ticker = %s;
            """, (ticker.upper(),))
            stat = cur.fetchone()
            
            # Last available point
            cur.execute("""
                SELECT date, sr, regime
                FROM caria_sr_daily
                WHERE ticker = %s
                ORDER BY date DESC
                LIMIT 1;
            """, (ticker.upper(),))
            last = cur.fetchone()

        if not stat or not last:
            raise HTTPException(status_code=404, detail="Ticker not found or no data calculated")

        return {
            "ticker": ticker.upper(),
            "auc": stat["auc"],
            "mean_normal": stat["mean_normal"],
            "mean_fragile": stat["mean_fragile"],
            "last_date": last["date"].isoformat(),
            "last_sr": last["sr"],
            "last_regime": int(last["regime"]),
            "last_updated": stat["last_updated"].isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
