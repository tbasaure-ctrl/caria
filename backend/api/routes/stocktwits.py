"""StockTwits social sentiment endpoints."""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Literal
import logging
import requests

router = APIRouter(prefix="/api/social", tags=["social"])
logger = logging.getLogger(__name__)

# StockTwits API endpoints (public, no auth required)
STOCKTWITS_TRENDING_URL = "https://api.stocktwits.com/api/2/trending.json"
STOCKTWITS_SYMBOL_URL = "https://api.stocktwits.com/api/2/streams/symbol/{}.json"


@router.get("/stocktwits")
async def get_stocktwits_trending(
    timeframe: Literal["hour", "day", "week"] = Query("day", description="Timeframe for trending (note: StockTwits API doesn't support timeframe filtering)")
):
    """
    Get trending stocks from StockTwits.
    
    Returns most mentioned tickers with sentiment analysis.
    StockTwits API provides trending symbols based on watchlist count and message volume.
    """
    
    try:
        # Fetch trending symbols from StockTwits
        response = requests.get(STOCKTWITS_TRENDING_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        symbols = data.get("symbols", [])
        
        if not symbols:
            logger.warning("StockTwits API returned no symbols")
            return {
                "stocks": [],
                "timeframe": timeframe,
                "mock_data": False
            }
        
        # Process trending symbols
        stocks = []
        for symbol_data in symbols[:20]:  # Get top 20 trending
            symbol = symbol_data.get("symbol", "").upper()
            if not symbol or len(symbol) > 5:
                continue
            
            watchlist_count = symbol_data.get("watchlist_count", 0)
            message_count = symbol_data.get("message_count", 0)
            
            # Calculate trending score based on watchlist and message activity
            # Normalize to 0-100 scale
            trending_score = min(100, int((watchlist_count * 0.6 + message_count * 0.4) / 10))
            
            # Determine sentiment based on message count and watchlist growth
            # Higher message count relative to watchlist suggests active discussion (bullish)
            if message_count > watchlist_count * 0.5:
                sentiment = "bullish"
            elif message_count < watchlist_count * 0.2:
                sentiment = "neutral"
            else:
                sentiment = "bullish" if watchlist_count > 100 else "neutral"
            
            # Try to get a recent message for context
            top_message = None
            try:
                symbol_response = requests.get(
                    STOCKTWITS_SYMBOL_URL.format(symbol),
                    timeout=5
                )
                if symbol_response.ok:
                    symbol_data_full = symbol_response.json()
                    messages = symbol_data_full.get("messages", [])
                    if messages:
                        top_message = messages[0].get("body", "")[:100]  # First 100 chars
            except Exception as e:
                logger.debug(f"Could not fetch message for {symbol}: {e}")
            
            stocks.append({
                "ticker": symbol,
                "mentions": message_count,
                "watchlist_count": watchlist_count,
                "sentiment": sentiment,
                "trending_score": trending_score,
                "top_message": top_message,
                "source": "stocktwits"
            })
        
        # Sort by trending score (combination of watchlist and messages)
        stocks.sort(key=lambda x: x["trending_score"], reverse=True)
        
        # Return top 10
        return {
            "stocks": stocks[:10],
            "timeframe": timeframe,
            "mock_data": False
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"StockTwits API request error: {e}")
        # Return mock data on error
        return {
            "stocks": [
                {
                    "ticker": "NVDA",
                    "mentions": 1523,
                    "watchlist_count": 12450,
                    "sentiment": "bullish",
                    "trending_score": 95,
                    "top_message": "NVDA breaking new highs on AI momentum",
                    "source": "stocktwits"
                },
                {
                    "ticker": "TSLA",
                    "mentions": 987,
                    "watchlist_count": 8920,
                    "sentiment": "neutral",
                    "trending_score": 78,
                    "top_message": "TSLA production update",
                    "source": "stocktwits"
                },
                {
                    "ticker": "AAPL",
                    "mentions": 756,
                    "watchlist_count": 6540,
                    "sentiment": "bullish",
                    "trending_score": 72,
                    "top_message": "Apple Vision Pro sales",
                    "source": "stocktwits"
                }
            ],
            "timeframe": timeframe,
            "mock_data": True,
            "error": f"StockTwits API error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Unexpected error accessing StockTwits API: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error accessing StockTwits API: {str(e)}"
        )
