"""
Professional curation service for recommended lectures/articles.
Sources include: Abnormal Returns, The Motley Fool, CFA Institute, Farnam Street.
All items go through a strict quality filter (domain allow-list, title length,
and banned keyword screening).
"""

import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, List, Optional
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger("caria.api.services.lectures")

ALLOWED_DOMAINS = {
    "abnormalreturns.com",
    "fool.com",
    "blogs.cfainstitute.org",
    "fs.blog",
    "awealthofcommonsense.com",
    "collaborativefund.com",
    "humbledollar.com",
}

BANNED_KEYWORDS = [
    "sponsored", "ad:", "crypto", "nft", "giveaway", "contest",
]

MAX_ARTICLES = 10


class LecturesService:
    def __init__(self) -> None:
        self._cache: Dict[str, List[Dict[str, str]]] = {}

    def get_daily_recommendations(self) -> List[Dict[str, str]]:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today in self._cache:
            return self._cache[today]

        articles: List[Dict[str, str]] = []
        fetchers = [
            self._scrape_abnormal_returns,
            self._fetch_motley_fool_rss,
            self._fetch_cfa_institute_rss,
            self._fetch_farnam_street_rss,
            self._fetch_awealthofcommonsense_rss,
            self._fetch_collaborative_fund_rss,
            self._fetch_humble_dollar_rss,
        ]

        for fetcher in fetchers:
            try:
                articles.extend(fetcher())
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Error fetching %s: %s", fetcher.__name__, exc)

        filtered = self._apply_quality_filter(articles)
        balanced = self._rebalance_sources(filtered)
        final_list = balanced[:MAX_ARTICLES]
        self._cache[today] = final_list
        return final_list

    def _apply_quality_filter(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        unique_titles = set()
        filtered: List[Dict[str, str]] = []

        for article in articles:
            title = article.get("title", "").strip()
            url = article.get("url", "")
            domain = urlparse(url).netloc.replace("www.", "")

            if not title or len(title) < 40:
                continue
            if any(bad in title.lower() for bad in BANNED_KEYWORDS):
                continue
            if not any(domain.endswith(allowed) for allowed in ALLOWED_DOMAINS):
                continue
            title_key = title.lower()
            if title_key in unique_titles:
                continue

            unique_titles.add(title_key)
            filtered.append(article)

        return filtered

    def _rebalance_sources(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not articles:
            return []

        buckets: Dict[str, Deque[Dict[str, str]]] = defaultdict(deque)
        for article in articles:
            source = article.get("source", "Other")
            buckets[source].append(article)

        ordered_sources = sorted(buckets.keys(), key=lambda s: (-len(buckets[s]), s))

        balanced: List[Dict[str, str]] = []
        while any(buckets.values()):
            for source in ordered_sources:
                if buckets[source]:
                    balanced.append(buckets[source].popleft())

        return balanced

    def _scrape_abnormal_returns(self) -> List[Dict[str, str]]:
        resp = requests.get("https://abnormalreturns.com/", timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        articles: List[Dict[str, str]] = []
        main_content = soup.find("div", class_="entry-content")
        if not main_content:
            latest_post = soup.find("h2", class_="entry-title")
            if latest_post and latest_post.find("a"):
                post_url = latest_post.find("a")["href"]
                resp_post = requests.get(post_url, timeout=10)
                resp_post.raise_for_status()
                soup_post = BeautifulSoup(resp_post.text, "html.parser")
                main_content = soup_post.find("div", class_="entry-content")

        if main_content:
            for link in main_content.find_all("a"):
                url = link.get("href")
                title = link.get_text().strip()
                if url and title and "http" in url:
                    if "abnormalreturns.com" in url or "twitter.com" in url:
                        continue
                    articles.append(
                        {
                            "title": title,
                            "url": url,
                            "source": "Abnormal Returns",
                            "date": datetime.utcnow().isoformat(),
                        }
                    )
                if len(articles) >= 5:
                    break
        return articles

    def _fetch_rss_entries(self, feed_url: str, source: str) -> List[Dict[str, str]]:
        resp = requests.get(feed_url, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        items: List[Dict[str, str]] = []

        for item in root.findall("./channel/item"):
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            pub_date = item.findtext("pubDate") or datetime.utcnow().isoformat()
            if title and link:
                items.append(
                    {
                        "title": title.strip(),
                        "url": link.strip(),
                        "source": source,
                        "date": pub_date,
                    }
                )
            if len(items) >= 5:
                break
        return items

    def _fetch_motley_fool_rss(self) -> List[Dict[str, str]]:
        return self._fetch_rss_entries(
            "https://www.fool.com/investing-news/?format=rss",
            "The Motley Fool",
        )

    def _fetch_cfa_institute_rss(self) -> List[Dict[str, str]]:
        return self._fetch_rss_entries(
            "https://blogs.cfainstitute.org/investor/feed/",
            "CFA Institute",
        )

    def _fetch_farnam_street_rss(self) -> List[Dict[str, str]]:
        return self._fetch_rss_entries(
            "https://fs.blog/feed/",
            "Farnam Street",
        )

    def _fetch_awealthofcommonsense_rss(self) -> List[Dict[str, str]]:
        return self._fetch_rss_entries(
            "https://awealthofcommonsense.com/feed/",
            "A Wealth of Common Sense",
        )

    def _fetch_collaborative_fund_rss(self) -> List[Dict[str, str]]:
        return self._fetch_rss_entries(
            "https://www.collaborativefund.com/blog/rss/",
            "Collaborative Fund",
        )

    def _fetch_humble_dollar_rss(self) -> List[Dict[str, str]]:
        return self._fetch_rss_entries(
            "https://humbledollar.com/feed/",
            "Humble Dollar",
        )


_lectures_service = LecturesService()


def get_lectures_service() -> LecturesService:
    return _lectures_service
