"""Reddit social sentiment endpoints."""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Literal
import os
import logging

router = APIRouter(prefix="/api/social", tags=["social"])
logger = logging.getLogger(__name__)

# You'll need to install praw: pip install praw
# Add REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT to your .env
try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    logger.warning("praw not installed. Reddit endpoints will return mock data.")


@router.get("/reddit")
async def get_reddit_sentiment(
    timeframe: Literal["hour", "day", "week"] = Query("day", description="Timeframe for mentions")
):
    """
    Get trending stocks from Reddit (r/wallstreetbets, r/stocks, r/investing).

    Returns most mentioned tickers with sentiment analysis.
    """

    if not REDDIT_AVAILABLE:
        # Return mock data if praw not installed
        return {
            "stocks": [
                {
                    "ticker": "NVDA",
                    "mentions": 1247,
                    "sentiment": "bullish",
                    "trending_score": 92,
                    "top_post_title": "NVDA earnings beat expectations",
                    "subreddit": "wallstreetbets"
                },
                {
                    "ticker": "TSLA",
                    "mentions": 856,
                    "sentiment": "neutral",
                    "trending_score": 78,
                    "top_post_title": "Tesla production numbers released",
                    "subreddit": "stocks"
                },
                {
                    "ticker": "AAPL",
                    "mentions": 634,
                    "sentiment": "bullish",
                    "trending_score": 71,
                    "top_post_title": "Apple Vision Pro sales surging",
                    "subreddit": "investing"
                },
                {
                    "ticker": "SPY",
                    "mentions": 521,
                    "sentiment": "bearish",
                    "trending_score": 65,
                    "top_post_title": "Market correction incoming?",
                    "subreddit": "wallstreetbets"
                },
                {
                    "ticker": "AMD",
                    "mentions": 412,
                    "sentiment": "bullish",
                    "trending_score": 58,
                    "top_post_title": "AMD new chip announcement",
                    "subreddit": "stocks"
                }
            ],
            "timeframe": timeframe,
            "mock_data": True
        }

    try:
        # Check if credentials are available
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "Caria-Investment-App-v1.0")
        
        # Log credential status (without exposing secrets)
        logger.info(f"Reddit credentials check:")
        logger.info(f"  REDDIT_CLIENT_ID present: {bool(client_id)}")
        logger.info(f"  REDDIT_CLIENT_SECRET present: {bool(client_secret)}")
        logger.info(f"  REDDIT_USER_AGENT: {user_agent}")
        
        if not client_id or not client_secret:
            error_msg = "Reddit credentials not configured. REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set."
            logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
        
        # Initialize Reddit API (read-only mode for public data)
        # Use proper User Agent format recommended by Reddit
        # Format: <platform>:<app ID>:<version> (by /u/<reddit username>)
        formatted_user_agent = user_agent
        if not any(char in user_agent for char in ['/', ':', 'by']):
            # If user agent doesn't follow Reddit format, use a simple descriptive one
            formatted_user_agent = f"web:CariaInvestmentApp:v1.0 (by /u/caria_app)"
        
        logger.info(f"Initializing Reddit API with client_id: {client_id[:10]}..., user_agent: {formatted_user_agent}")
        
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=formatted_user_agent,
            check_for_async=False  # Disable async check for FastAPI compatibility
        )
        
        # Test connection by accessing a public subreddit
        try:
            test_subreddit = reddit.subreddit("test")
            _ = test_subreddit.display_name  # This will raise if auth fails
            logger.info("Reddit API connection successful")
        except Exception as auth_error:
            logger.error(f"Reddit API authentication failed: {auth_error}")
            raise
        
        # Subreddits to monitor
        subreddits = ["wallstreetbets", "stocks", "investing"]

        # Map timeframe to Reddit time filter
        time_filters = {
            "hour": "hour",
            "day": "day",
            "week": "week"
        }

        # Collect tickers from posts
        ticker_mentions = {}
        ticker_posts = {}

        for subreddit_name in subreddits:
            subreddit = reddit.subreddit(subreddit_name)

            # Get hot posts
            for submission in subreddit.hot(limit=100):
                # Simple ticker extraction (look for $TICKER or uppercase tickers)
                import re
                tickers = re.findall(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b', submission.title + " " + submission.selftext)

                for ticker_match in tickers:
                    ticker = ticker_match[0] or ticker_match[1]
                    if ticker and len(ticker) <= 5:  # Filter out non-ticker words
                        ticker_mentions[ticker] = ticker_mentions.get(ticker, 0) + 1

                        # Save top post for each ticker
                        if ticker not in ticker_posts or submission.score > ticker_posts[ticker]["score"]:
                            ticker_posts[ticker] = {
                                "title": submission.title,
                                "subreddit": subreddit_name,
                                "score": submission.score
                            }

        # Sort by mentions and get top stocks
        sorted_tickers = sorted(ticker_mentions.items(), key=lambda x: x[1], reverse=True)[:10]

        # Format response
        stocks = []
        for ticker, mentions in sorted_tickers:
            post_info = ticker_posts.get(ticker, {})

            # Simple sentiment (you can enhance this with NLP)
            sentiment = "neutral"
            if post_info.get("score", 0) > 100:
                sentiment = "bullish"
            elif post_info.get("score", 0) < 50:
                sentiment = "bearish"

            stocks.append({
                "ticker": ticker,
                "mentions": mentions,
                "sentiment": sentiment,
                "trending_score": min(100, mentions * 2),  # Simple trending score
                "top_post_title": post_info.get("title"),
                "subreddit": post_info.get("subreddit")
            })

        return {
            "stocks": stocks,
            "timeframe": timeframe,
            "mock_data": False
        }

    except Exception as e:
        error_str = str(e).lower()
        error_msg = str(e)
        
        # Log detailed error information
        logger.error(f"Reddit API error details:")
        logger.error(f"  Error type: {type(e).__name__}")
        logger.error(f"  Error message: {error_msg}")
        logger.error(f"  Client ID configured: {bool(client_id)}")
        logger.error(f"  Client Secret configured: {bool(client_secret)}")
        logger.error(f"  User Agent: {user_agent}")
        
        # Check for specific Reddit API errors
        if "401" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
            logger.error(f"Reddit API authentication failed (401/403): {e}")
            logger.error("Possible causes:")
            logger.error("  1. Client ID or Secret incorrect")
            logger.error("  2. User Agent format not accepted by Reddit")
            logger.error("  3. Reddit API rate limiting or blocking")
            logger.error("  4. Credentials expired or revoked")
            # Don't return mock data - raise the error so we can see it
            raise HTTPException(
                status_code=500,
                detail=f"Reddit API authentication failed: {error_msg}. Check logs for details."
            )
        elif "praw" in error_str or "reddit" in error_str:
            logger.error(f"Reddit/PRAW error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Reddit API error: {error_msg}"
            )
        else:
            logger.error(f"Unexpected error accessing Reddit API: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error accessing Reddit API: {error_msg}"
            )
        
        # Fallback mock data (should not reach here due to raises above)
        return {
            "stocks": [
                {
                    "ticker": "NVDA",
                    "mentions": 1247,
                    "sentiment": "bullish",
                    "trending_score": 92,
                    "top_post_title": "NVDA earnings beat expectations",
                    "subreddit": "wallstreetbets"
                },
                {
                    "ticker": "TSLA",
                    "mentions": 856,
                    "sentiment": "neutral",
                    "trending_score": 78,
                    "top_post_title": "Tesla production numbers released",
                    "subreddit": "stocks"
                },
                {
                    "ticker": "AAPL",
                    "mentions": 634,
                    "sentiment": "bullish",
                    "trending_score": 71,
                    "top_post_title": "Apple Vision Pro sales surging",
                    "subreddit": "investing"
                },
                {
                    "ticker": "SPY",
                    "mentions": 521,
                    "sentiment": "bearish",
                    "trending_score": 65,
                    "top_post_title": "Market correction incoming?",
                    "subreddit": "wallstreetbets"
                },
                {
                    "ticker": "AMD",
                    "mentions": 412,
                    "sentiment": "bullish",
                    "trending_score": 58,
                    "top_post_title": "AMD new chip announcement",
                    "subreddit": "stocks"
                }
            ],
            "timeframe": timeframe,
            "mock_data": True,
            "error": f"Reddit API unavailable: {str(e)}"
        }
