"""
Market News API - GDELT-powered news feed
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.dependencies import get_current_user, get_db_connection
from caria.models.auth import UserInDB
from api.services.gdelt_service import gdelt_service

router = APIRouter(prefix="/api/news", tags=["news"])

class NewsArticle(BaseModel):
    id: int
    title: str
    source_domain: str
    url: str
    published_at: str
    tone: Optional[float]
    related_tickers: Optional[List[str]] = None

@router.get("/market", response_model=List[NewsArticle])
def get_market_news(
    tickers: Optional[str] = Query(None, description="Comma-separated list of tickers"),
    limit: int = Query(20, ge=1, le=100),
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
):
    """
    Get recent market news, optionally filtered by tickers.
    """
    try:
        ticker_list = None
        if tickers:
            ticker_list = [t.strip().upper() for t in tickers.split(',')]
        
        news = gdelt_service.get_market_news(conn, ticker_list, limit)
        return news
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fetch")
def trigger_news_fetch(
    query: str = Query(..., description="Search query (e.g., company name, sector)"),
    days: int = Query(7, ge=1, le=30),
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
):
    """
    Admin endpoint to manually trigger news fetch from GDELT.
    """
    try:
        from datetime import datetime, timezone, timedelta
        
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)
        
        articles = gdelt_service.fetch_news(query, start_dt, end_dt, max_records=100)
        inserted = gdelt_service.ingest_news(conn, articles)
        
        return {
            "message": f"Fetched {len(articles)} articles, inserted {inserted} new ones",
            "query": query,
            "days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
