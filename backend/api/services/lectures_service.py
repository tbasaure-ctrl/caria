"""
Service to fetch recommended lectures/articles from external sources.
Sources:
- Abnormal Returns (Finance/Investing)
- Morning Brew (General Business/Tech)
"""
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional

LOGGER = logging.getLogger("caria.api.services.lectures")

class LecturesService:
    def __init__(self):
        self.sources = [
            {
                "name": "Abnormal Returns",
                "url": "https://abnormalreturns.com/",
                "type": "finance",
                "logo": "https://abnormalreturns.com/favicon.ico"
            },
            {
                "name": "Morning Brew",
                "url": "https://www.morningbrew.com/daily",
                "type": "business",
                "logo": "https://www.morningbrew.com/favicon.ico"
            }
        ]
        # Simple in-memory cache: {date_str: [articles]}
        self._cache = {}

    def get_daily_recommendations(self) -> List[Dict[str, str]]:
        """
        Get recommended articles for today.
        Returns a list of dicts with title, url, source, date.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Check cache first
        if today in self._cache:
            return self._cache[today]
        
        articles = []
        
        # Fetch from Abnormal Returns
        try:
            ar_articles = self._scrape_abnormal_returns()
            articles.extend(ar_articles)
        except Exception as e:
            LOGGER.error(f"Error fetching Abnormal Returns: {e}")
            
        # Fetch from Morning Brew
        try:
            mb_articles = self._scrape_morning_brew()
            articles.extend(mb_articles)
        except Exception as e:
            LOGGER.error(f"Error fetching Morning Brew: {e}")
            
        # Update cache if we found something
        if articles:
            self._cache[today] = articles
            
        return articles

    def _scrape_abnormal_returns(self) -> List[Dict[str, str]]:
        """Scrape latest links from Abnormal Returns."""
        try:
            resp = requests.get("https://abnormalreturns.com/", timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            articles = []
            # Logic depends on their current HTML structure. 
            # Usually they have a daily link post.
            # Looking for the first main post content which usually contains the links.
            
            # This is a heuristic. Adjust based on actual site structure.
            # Assuming standard WordPress structure
            main_content = soup.find('div', class_='entry-content')
            if not main_content:
                # Try finding the latest post link first
                latest_post = soup.find('h2', class_='entry-title')
                if latest_post and latest_post.find('a'):
                    post_url = latest_post.find('a')['href']
                    # Fetch the specific post
                    resp_post = requests.get(post_url, timeout=10)
                    soup_post = BeautifulSoup(resp_post.text, 'html.parser')
                    main_content = soup_post.find('div', class_='entry-content')

            if main_content:
                # Extract links
                for link in main_content.find_all('a'):
                    url = link.get('href')
                    title = link.get_text().strip()
                    
                    if url and title and len(title) > 10 and "http" in url:
                        # Filter out internal links or garbage
                        if "abnormalreturns.com" not in url and "twitter.com" not in url:
                            articles.append({
                                "title": title,
                                "url": url,
                                "source": "Abnormal Returns",
                                "date": datetime.now().isoformat()
                            })
                            if len(articles) >= 5: # Limit to 5
                                break
            return articles
        except Exception as e:
            LOGGER.warning(f"Scraping Abnormal Returns failed: {e}")
            return []

    def _scrape_morning_brew(self) -> List[Dict[str, str]]:
        """Scrape latest stories from Morning Brew."""
        try:
            # Morning Brew main page usually has "Latest" or similar
            resp = requests.get("https://www.morningbrew.com/daily", timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            articles = []
            # Look for story cards or links
            # Heuristic: look for h3 or links with specific classes
            # This is fragile and might need adjustment
            
            for link in soup.find_all('a', href=True):
                url = link['href']
                # Check if it looks like a story
                if "/stories/" in url:
                    title = link.get_text().strip()
                    if title and len(title) > 10:
                        full_url = f"https://www.morningbrew.com{url}" if url.startswith("/") else url
                        articles.append({
                            "title": title,
                            "url": full_url,
                            "source": "Morning Brew",
                            "date": datetime.now().isoformat()
                        })
                        if len(articles) >= 5:
                            break
                            
            return articles
        except Exception as e:
            LOGGER.warning(f"Scraping Morning Brew failed: {e}")
            return []

# Singleton
_lectures_service = LecturesService()

def get_lectures_service():
    return _lectures_service
