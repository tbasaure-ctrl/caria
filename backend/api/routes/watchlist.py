from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
from ..database import get_db_connection

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])

class WatchlistItemCreate(BaseModel):
    ticker: str

class WatchlistItem(BaseModel):
    id: str
    ticker: str
    company_name: str | None = None
    added_date: str
    current_price: float | None = None
    change_pct: float | None = None

def get_current_user_id():
    """Placeholder for authentication - returns default user ID"""
    # In production, this would extract the user ID from JWT token
    return "default_user"

@router.get("")
async def get_watchlist(user_id: str = Depends(get_current_user_id)):
    """Get user's watchlist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get watchlist items for user
        cursor.execute("""
            SELECT id, ticker, company_name, added_date, current_price, change_pct
            FROM watchlist
            WHERE user_id = %s
            ORDER BY added_date DESC
        """, (user_id,))
        
        rows = cursor.fetchall()
        watchlist = [
            {
                "id": str(row[0]),
                "ticker": row[1],
                "company_name": row[2],
                "added_date": row[3].isoformat() if row[3] else None,
                "current_price": float(row[4]) if row[4] else None,
                "change_pct": float(row[5]) if row[5] else None
            }
            for row in rows
        ]
        
        cursor.close()
        conn.close()
        
        return {"watchlist": watchlist}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("")
async def add_to_watchlist(item: WatchlistItemCreate, user_id: str = Depends(get_current_user_id)):
    """Add a ticker to user's watchlist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if already in watchlist
        cursor.execute("""
            SELECT id FROM watchlist
            WHERE user_id = %s AND ticker = %s
        """, (user_id, item.ticker.upper()))
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Ticker already in watchlist")
        
        # Insert new item
        cursor.execute("""
            INSERT INTO watchlist (user_id, ticker, added_date)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (user_id, item.ticker.upper(), datetime.now()))
        
        new_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        return {"id": str(new_id), "ticker": item.ticker.upper()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.delete("/{item_id}")
async def remove_from_watchlist(item_id: str, user_id: str = Depends(get_current_user_id)):
    """Remove a ticker from user's watchlist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM watchlist
            WHERE id = %s AND user_id = %s
            RETURNING ticker
        """, (item_id, user_id))
        
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Watchlist item not found")
        
        return {"message": "Removed from watchlist", "ticker": result[0]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
