import os
import re
import time
import requests
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import Counter
import logging

LOGGER = logging.getLogger("caria.api.services.social_screener")

# Try to import optional deps
try:
    import praw
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    # Download vader lexicon if needed
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    HAS_NLTK = True
except ImportError:
    LOGGER.warning("NLTK/PRAW not installed. Social screener running in limited mode.")
    HAS_NLTK = False
    sia = None

try:
    from pytrends.request import TrendReq
    pytrends = TrendReq(hl='en-US', tz=360)
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False
    pytrends = None

class SocialScreenerService:
    def __init__(self):
        self.reddit_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_agent = os.getenv('REDDIT_USER_AGENT', 'StockScreenerBot/1.0')
        self.fmp_key = os.getenv('FMP_API_KEY')
        self.db_url = os.getenv('DATABASE_URL') or os.getenv('NEON_DB_URL')
        self.base_url = 'https://financialmodelingprep.com/api/v3'
        self.st_base_url = 'https://api.stocktwits.com/api/2'
        self.mag7 = {'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA'}

    def _get_db_connection(self):
        if not self.db_url:
            password = os.getenv("POSTGRES_PASSWORD")
            if not password: return None
            return psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                user=os.getenv("POSTGRES_USER", "caria_user"),
                password=password,
                database=os.getenv("POSTGRES_DB", "caria"),
            )
        return psycopg2.connect(self.db_url)

    def fetch_fmp(self, endpoint: str, params: Dict = None) -> List[Dict]:
        if params is None: params = {}
        params['apikey'] = self.fmp_key
        try:
            res = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=5)
            return res.json() if res.status_code == 200 else []
        except:
            return []

    def is_valid_ticker(self, symbol: str) -> bool:
        data = self.fetch_fmp(f'/profile/{symbol}')
        # Market Cap < 50B for "Under Radar"
        return bool(data) and symbol not in self.mag7 and data[0].get('mktCap', 0) < 50000000000

    def extract_tickers(self, text: str) -> List[str]:
        ticker_pattern = r'\b\$?([A-Z]{2,5})\b'
        potential = re.findall(ticker_pattern, text.upper())
        # Basic filter of common words
        COMMON = {'THE', 'AND', 'FOR', 'YOU', 'ARE', 'THIS', 'THAT', 'WITH', 'HAVE', 'NOT'}
        return [t for t in set(potential) if t not in COMMON] 

    def get_reddit_mentions(self) -> Dict[str, int]:
        if not HAS_NLTK or not self.reddit_id:
            return {}
        
        try:
            reddit = praw.Reddit(client_id=self.reddit_id, client_secret=self.reddit_secret, user_agent=self.reddit_agent)
            ticker_counts = Counter()
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'pennystocks']
            
            for sub in subreddits:
                subreddit = reddit.subreddit(sub)
                for submission in subreddit.hot(limit=20): # Limit to 20 for speed
                    text = submission.title + ' ' + submission.selftext
                    tickers = self.extract_tickers(text)
                    ticker_counts.update(tickers)
            return dict(ticker_counts.most_common(30))
        except Exception as e:
            LOGGER.error(f"Reddit fetch failed: {e}")
            return {}

    def get_stocktwits_trending(self) -> Dict[str, float]:
        try:
            response = requests.get(f"{self.st_base_url}/streams/trending.json?limit=30", timeout=5)
            if response.status_code != 200: return {}
            
            data = response.json().get('messages', [])
            bullish_counts = Counter()
            total_counts = Counter()
            
            for msg in data:
                symbol = msg.get('entities', {}).get('symbols', [{}])[0].get('symbol', '')
                if symbol:
                    sentiment = 1 if msg.get('entities', {}).get('sentiment', {}).get('basic') == 'Bullish' else 0
                    bullish_counts[symbol] += sentiment
                    total_counts[symbol] += 1
            
            return {sym: bullish_counts[sym] / total_counts[sym] if total_counts[sym] else 0 for sym in total_counts}
        except:
            return {}

    def run_screen(self):
        # 1. Get trending tickers from Reddit & StockTwits
        reddit_data = self.get_reddit_mentions()
        st_data = self.get_stocktwits_trending()
        
        all_tickers = list(set(list(reddit_data.keys()) + list(st_data.keys())))
        
        results = []
        for ticker in all_tickers[:5]:
            # 2. Validate if it's a real stock and under radar
            if not self.is_valid_ticker(ticker):
                continue
                
            r_mentions = reddit_data.get(ticker, 0)
            st_bullish = st_data.get(ticker, 0)
            
            # 3. Calculate Score
            # Simple weighted score
            social_score = (r_mentions * 2) + (st_bullish * 20)
            sentiment = st_bullish # 0-1
            
            if social_score > 5:
                results.append({
                    'symbol': ticker,
                    'reddit_mentions': r_mentions,
                    'stocktwits_bullish': st_bullish,
                    'social_score': social_score,
                    'sentiment_avg': sentiment
                })
        
        results.sort(key=lambda x: x['social_score'], reverse=True)
        top3 = results[:3]
        
        self._save_results(top3)
        return top3

    def _save_results(self, results):
        conn = self._get_db_connection()
        if not conn: return
        
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS social_screening_results (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symbol VARCHAR(10),
                    reddit_mentions INT,
                    stocktwits_bullish FLOAT,
                    x_mentions INT,
                    social_score DECIMAL,
                    sentiment_avg DECIMAL,
                    rank INTEGER
                );
            """)
            
            timestamp = datetime.now()
            for i, row in enumerate(results):
                rank = i + 1
                cur.execute("""
                    INSERT INTO social_screening_results 
                    (timestamp, symbol, reddit_mentions, stocktwits_bullish, x_mentions, social_score, sentiment_avg, rank)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (timestamp, row['symbol'], row['reddit_mentions'], row['stocktwits_bullish'], 0, row['social_score'], row['sentiment_avg'], rank))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Social DB Save failed: {e}")

