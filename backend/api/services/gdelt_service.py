"""
GDELT Service - Fetch and ingest news from GDELT Doc API
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class GdeltService:
    """Service for fetching news from GDELT and storing in database."""
    
    def __init__(self):
        self.gdelt_available = self._check_gdeltdoc()
        
    def _check_gdeltdoc(self) -> bool:
        """Check if gdeltdoc is available."""
        try:
            from gdeltdoc import GdeltDoc
            return True
        except ImportError:
            logger.warning("gdeltdoc not installed. Run: pip install gdeltdoc")
            return False
    
    def fetch_news(
        self, 
        query: str, 
        start_dt: datetime, 
        end_dt: datetime,
        max_records: int = 250
    ) -> List[Dict[str, Any]]:
        """
        Fetch news from GDELT Doc API.
        
        Args:
            query: Search query (company name, topic, etc.)
            start_dt: Start datetime (UTC)
            end_dt: End datetime (UTC)
            max_records: Maximum number of articles to fetch
            
        Returns:
            List of article dictionaries
        """
        if not self.gdelt_available:
            logger.error("gdeltdoc not available")
            return []
        
        try:
            from gdeltdoc import GdeltDoc
            gd = GdeltDoc()
            
            # Format dates for GDELT API
            start_str = start_dt.strftime("%Y%m%d%H%M%S")
            end_str = end_dt.strftime("%Y%m%d%H%M%S")
            
            # Fetch articles
            articles = gd.article_search(
                query=query,
                startdatetime=start_str,
                enddatetime=end_str,
                maxrecords=max_records
            )
            
            logger.info(f"Fetched {len(articles) if articles else 0} articles for query: {query}")
            return articles if articles else []
            
        except Exception as e:
            logger.error(f"Error fetching GDELT news: {e}")
            return []
    
    def ingest_news(self, db_conn, articles: List[Dict[str, Any]]) -> int:
        """
        Ingest GDELT articles into database.
        
        Args:
            db_conn: Database connection
            articles: List of article dictionaries from GDELT
            
        Returns:
            Number of articles inserted
        """
        if not articles:
            return 0
        
        inserted = 0
        try:
            with db_conn.cursor() as cur:
                for article in articles:
                    try:
                        # Extract fields
                        gdelt_id = article.get('url') or article.get('documentidentifier', '')
                        if not gdelt_id:
                            continue
                        
                        title = article.get('title', '')
                        url = article.get('url', '')
                        outlet = article.get('domain', '')
                        
                        # Parse publication date
                        pub_date_str = article.get('seendate') or article.get('publishdate', '')
                        if pub_date_str:
                            try:
                                published_at = datetime.strptime(pub_date_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                            except:
                                published_at = datetime.now(timezone.utc)
                        else:
                            published_at = datetime.now(timezone.utc)
                        
                        # GDELT-specific fields
                        tone = None
                        if article.get('tone'):
                            try:
                                tone = float(article['tone'])
                            except:
                                pass
                        
                        themes = article.get('themes', '').split(';') if article.get('themes') else []
                        locations = article.get('locations', '').split(';') if article.get('locations') else []
                        language = article.get('language', 'en')
                        
                        # Insert into database
                        cur.execute("""
                            INSERT INTO gdelt_news_raw (
                                gdelt_id, source, source_domain, title, url,
                                published_at, language, tone, themes, locations, meta
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (gdelt_id) DO NOTHING
                            RETURNING id
                        """, (
                            gdelt_id,
                            'gdelt_doc',
                            outlet,
                            title,
                            url,
                            published_at,
                            language,
                            tone,
                            themes or None,
                            locations or None,
                            article
                        ))
                        
                        if cur.fetchone():
                            inserted += 1
                            
                    except Exception as e:
                        logger.warning(f"Error inserting article: {e}")
                        continue
                
                db_conn.commit()
                logger.info(f"Inserted {inserted} new articles into database")
                
        except Exception as e:
            logger.error(f"Error ingesting news: {e}")
            db_conn.rollback()
        
        return inserted
    
    def get_market_news(
        self, 
        db_conn, 
        tickers: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recent market news, optionally filtered by tickers.
        
        Args:
            db_conn: Database connection
            tickers: Optional list of tickers to filter by
            limit: Maximum number of articles
            
        Returns:
            List of news articles
        """
        try:
            from psycopg2.extras import RealDictCursor
            with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
                if tickers:
                    # Filter by tickers
                    cur.execute("""
                        SELECT DISTINCT
                            n.id,
                            n.title,
                            n.source_domain,
                            n.url,
                            n.published_at,
                            n.tone,
                            array_agg(DISTINCT nt.ticker) as related_tickers
                        FROM gdelt_news_raw n
                        JOIN gdelt_news_tickers nt ON n.id = nt.news_id
                        WHERE nt.ticker = ANY(%s)
                        AND n.published_at >= NOW() - INTERVAL '7 days'
                        GROUP BY n.id, n.title, n.source_domain, n.url, n.published_at, n.tone
                        ORDER BY n.published_at DESC
                        LIMIT %s
                    """, (tickers, limit))
                else:
                    # Get general market news
                    cur.execute("""
                        SELECT
                            id,
                            title,
                            source_domain,
                            url,
                            published_at,
                            tone
                        FROM gdelt_news_raw
                        WHERE published_at >= NOW() - INTERVAL '7 days'
                        ORDER BY published_at DESC
                        LIMIT %s
                    """, (limit,))
                
                return cur.fetchall()
                
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []

# Singleton instance
gdelt_service = GdeltService()
