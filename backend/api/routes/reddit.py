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
        # Initialize Reddit API (read-only mode for public data)
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "Caria Investment App v1.0"),
            check_for_async=False  # Disable async check for FastAPI compatibility
        )

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
        logger.error(f"Reddit API error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch Reddit data: {str(e)}")
