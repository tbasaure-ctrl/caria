"""
Automated News Fetching Scheduler
Fetches GDELT news on a schedule to keep the database updated.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
import asyncio
import threading

logger = logging.getLogger(__name__)

class NewsScheduler:
    """Scheduler for automated GDELT news fetching."""
    
    def __init__(self, gdelt_service, db_connection_func):
        """
        Initialize the news scheduler.
        
        Args:
            gdelt_service: Instance of GdeltService
            db_connection_func: Function that returns a database connection
        """
        self.gdelt_service = gdelt_service
        self.db_connection_func = db_connection_func
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        
    def fetch_market_news_job(self):
        """
        Background job that fetches market news from GDELT.
        Runs every 6 hours to keep news fresh.
        """
        logger.info("Starting automated news fetch job")
        
        try:
            # Get database connection
            conn = self.db_connection_func()
            
            # Define queries for different market segments
            queries = [
                "stock market",
                "earnings",
                "IPO",
                "mergers acquisitions",
                "Federal Reserve",
                "inflation",
                "unemployment",
                "GDP",
            ]
            
            # Fetch news for the last 24 hours
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=1)
            
            total_inserted = 0
            
            for query in queries:
                try:
                    logger.info(f"Fetching news for query: {query}")
                    articles = self.gdelt_service.fetch_news(
                        query=query,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        max_records=50  # Limit per query to avoid rate limits
                    )
                    
                    if articles:
                        inserted = self.gdelt_service.ingest_news(conn, articles)
                        total_inserted += inserted
                        logger.info(f"Inserted {inserted} articles for query: {query}")
                    
                    # Small delay between queries to avoid rate limiting
                    import time
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error fetching news for query '{query}': {e}")
                    continue
            
            conn.close()
            logger.info(f"News fetch job completed. Total inserted: {total_inserted}")
            
        except Exception as e:
            logger.error(f"Error in news fetch job: {e}")
    
    def run_periodic(self, interval_hours: int = 6):
        """
        Run the news fetch job periodically in a background thread.
        
        Args:
            interval_hours: Hours between each fetch (default: 6)
        """
        import time
        
        self.is_running = True
        
        # Run initial fetch immediately
        self.fetch_market_news_job()
        
        while self.is_running:
            try:
                # Wait for the specified interval
                time.sleep(interval_hours * 3600)
                
                if self.is_running:
                    self.fetch_market_news_job()
                
            except Exception as e:
                logger.error(f"Error in periodic news scheduler: {e}")
                # Wait 1 hour before retrying on error
                time.sleep(3600)
    
    def start(self, interval_hours: int = 6):
        """
        Start the scheduler in a background thread.
        
        Args:
            interval_hours: Hours between each fetch (default: 6)
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info(f"Starting news scheduler (interval: {interval_hours} hours)")
        
        # Start periodic fetching in background thread
        self._thread = threading.Thread(
            target=self.run_periodic,
            args=(interval_hours,),
            daemon=True,
            name="NewsScheduler"
        )
        self._thread.start()
    
    def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping news scheduler")
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)

