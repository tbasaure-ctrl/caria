"""
Under-the-Radar Screener Service

Detects small-cap stocks with:
1. Social momentum spikes (Reddit + Stocktwits)
2. Recent catalysts (8-K, earnings, insider buys)
3. Quality filters (ROCE proxy, efficiency, FCF yield)
4. Size & liquidity filters (50M-800M market cap, volume)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import requests
from caria.ingestion.clients.fmp_client import FMPClient

LOGGER = logging.getLogger("caria.api.under_the_radar_screener")

# Stocktwits API (public, no auth required for trending)
STOCKTWITS_TRENDING_URL = "https://api.stocktwits.com/api/2/trending.json"
STOCKTWITS_SYMBOL_URL = "https://api.stocktwits.com/api/2/streams/symbol/{}.json"


class UnderTheRadarScreenerService:
    """Lightweight screener for under-the-radar stocks."""

    def __init__(self, fmp_client: FMPClient | None = None):
        self.fmp = fmp_client or FMPClient()
        # Cache for social data to avoid repeated calls
        self._social_cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamp: datetime | None = None

    # ---------------------------------------------------------------------
    # Step 1: Social Momentum Detection
    # ---------------------------------------------------------------------

    def _get_stocktwits_trending(self) -> dict[str, dict[str, Any]]:
        """Fetch trending tickers from Stocktwits."""
        try:
            resp = requests.get(STOCKTWITS_TRENDING_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            trending = {}
            for item in data.get("symbols", []):
                symbol = item.get("symbol", "").upper()
                if symbol and len(symbol) <= 5:
                    trending[symbol] = {
                        "watchlist_count": item.get("watchlist_count", 0),
                        "message_count": item.get("message_count", 0),
                        "source": "stocktwits",
                    }
            return trending
        except Exception as e:
            LOGGER.warning(f"Error fetching Stocktwits trending: {e}")
            return {}

    def _get_stocktwits_symbol_stats(self, ticker: str) -> dict[str, Any] | None:
        """Get message volume stats for a specific ticker."""
        try:
            url = STOCKTWITS_SYMBOL_URL.format(ticker)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # Get message count from stream
            messages = data.get("messages", [])
            return {
                "current_messages": len(messages),
                "source": "stocktwits",
            }
        except Exception as e:
            LOGGER.debug(f"Error fetching Stocktwits stats for {ticker}: {e}")
            return None

    def _get_reddit_mentions(
        self, subreddits: list[str] = None
    ) -> dict[str, dict[str, Any]]:
        """Get ticker mentions from Reddit using existing infrastructure."""
        if subreddits is None:
            subreddits = ["wallstreetbets", "stocks", "investing", "pennystocks"]

        mentions: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "sources": set()}
        )

        try:
            # Use praw if available, otherwise return empty
            import praw
            import os

            client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
            client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
            user_agent = os.getenv("REDDIT_USER_AGENT", "Caria-Investment-App-v1.0")

            if not client_id or not client_secret:
                LOGGER.debug("Reddit credentials not available, skipping Reddit mentions")
                return {}

            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                check_for_async=False,
            )

            # Get last 24 hours of posts
            for subreddit_name in subreddits:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    # Get hot posts (last 24h equivalent)
                    for submission in subreddit.hot(limit=100):
                        # Extract tickers from title and selftext
                        text = (submission.title + " " + getattr(submission, "selftext", "")).upper()
                        # Match $TICKER or standalone tickers (2-5 uppercase letters)
                        ticker_matches = re.findall(
                            r'\$([A-Z]{2,5})\b|\b([A-Z]{2,5})\b', text
                        )
                        for match in ticker_matches:
                            ticker = (match[0] or match[1]).upper()
                            if ticker and len(ticker) >= 2 and len(ticker) <= 5:
                                # Filter out common words
                                if ticker not in ["THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE"]:
                                    mentions[ticker]["count"] += 1
                                    mentions[ticker]["sources"].add(subreddit_name)
                except Exception as e:
                    LOGGER.warning(f"Error fetching from r/{subreddit_name}: {e}")
                    continue

            # Convert sets to lists for JSON serialization
            result = {}
            for ticker, data in mentions.items():
                result[ticker] = {
                    "count": data["count"],
                    "sources": list(data["sources"]),
                    "source": "reddit",
                }
            return result

        except ImportError:
            LOGGER.debug("praw not available, skipping Reddit mentions")
            return {}
        except Exception as e:
            LOGGER.warning(f"Error fetching Reddit mentions: {e}")
            return {}

    def _detect_social_spikes(
        self, lookback_days: int = 30
    ) -> list[dict[str, Any]]:
        """
        Detect tickers with social momentum spikes.
        Returns tickers that show spikes in at least 2 independent sources.
        """
        LOGGER.info("Detecting social momentum spikes...")

        # Get current social data
        stocktwits_trending = self._get_stocktwits_trending()
        reddit_mentions = self._get_reddit_mentions()

        # For now, we'll use current data and estimate spikes
        # In production, you'd want to store historical data to compare
        candidates: dict[str, dict[str, Any]] = {}

        # Process Stocktwits trending (these are already trending)
        for ticker, data in stocktwits_trending.items():
            if ticker not in candidates:
                candidates[ticker] = {
                    "ticker": ticker,
                    "sources": [],
                    "spike_metrics": {},
                }
            candidates[ticker]["sources"].append("stocktwits")
            candidates[ticker]["spike_metrics"]["stocktwits_watchlist"] = data.get(
                "watchlist_count", 0
            )
            candidates[ticker]["spike_metrics"]["stocktwits_messages"] = data.get(
                "message_count", 0
            )

        # Process Reddit mentions
        for ticker, data in reddit_mentions.items():
            if ticker not in candidates:
                candidates[ticker] = {
                    "ticker": ticker,
                    "sources": [],
                    "spike_metrics": {},
                }
            candidates[ticker]["sources"].append("reddit")
            candidates[ticker]["spike_metrics"]["reddit_mentions"] = data.get("count", 0)
            candidates[ticker]["spike_metrics"]["reddit_sources"] = data.get("sources", [])

        # Filter: only keep tickers with spikes in at least 2 sources
        filtered = []
        for ticker, data in candidates.items():
            if len(data["sources"]) >= 2:
                # Calculate spike percentage (simplified - in production, compare to historical)
                spike_score = sum(data["spike_metrics"].values())
                data["spike_score"] = spike_score
                filtered.append(data)

        # Sort by spike score
        filtered.sort(key=lambda x: x.get("spike_score", 0), reverse=True)
        LOGGER.info(f"Found {len(filtered)} tickers with social spikes in 2+ sources")

        return filtered[:20]  # Top 20 candidates

    # ---------------------------------------------------------------------
    # Step 2: Catalyst Filter
    # ---------------------------------------------------------------------

    def _check_catalysts(self, ticker: str) -> dict[str, Any] | None:
        """
        Check for recent catalysts in the last 30 days.
        Returns catalyst info or None if no meaningful catalyst found.
        """
        try:
            catalysts = {
                "found": False,
                "flags": [],
                "details": {},
            }

            # Check for recent 8-K filings (FMP has SEC filings endpoint)
            try:
                # FMP doesn't have direct 8-K endpoint, but we can check news
                # For now, we'll use news and key metrics to infer catalysts
                news = self._get_recent_news(ticker, days=30)
                if news:
                    catalyst_keywords = [
                        "acquisition",
                        "fda approval",
                        "contract award",
                        "license agreement",
                        "phase 3",
                        "merger",
                        "partnership",
                        "approval",
                        "breakthrough",
                        "milestone",
                    ]
                    for article in news:
                        title_lower = article.get("title", "").lower()
                        text_lower = article.get("text", "").lower()
                        for keyword in catalyst_keywords:
                            if keyword in title_lower or keyword in text_lower:
                                catalysts["found"] = True
                                catalysts["flags"].append(f"News: {keyword}")
                                catalysts["details"]["news_catalyst"] = article.get("title", "")
                                break
            except Exception as e:
                LOGGER.debug(f"Error checking news for {ticker}: {e}")

            # Check for revenue growth > 50% YoY in latest quarter
            try:
                income = self.fmp.get_income_statement(ticker, period="quarter")
                if len(income) >= 2:
                    latest = income[0]
                    year_ago = income[1] if len(income) > 1 else income[0]

                    latest_revenue = latest.get("revenue", 0) or 0
                    year_ago_revenue = year_ago.get("revenue", 0) or 0

                    if year_ago_revenue > 0:
                        revenue_growth = (
                            (latest_revenue - year_ago_revenue) / year_ago_revenue
                        ) * 100
                        if revenue_growth > 50:
                            catalysts["found"] = True
                            catalysts["flags"].append(f"Revenue growth: {revenue_growth:.1f}% YoY")
                            catalysts["details"]["revenue_growth_pct"] = revenue_growth

                    # Check gross margins improving
                    latest_gross = latest.get("grossProfit", 0) or 0
                    year_ago_gross = year_ago.get("grossProfit", 0) or 0
                    if latest_revenue > 0 and year_ago_revenue > 0:
                        latest_margin = (latest_gross / latest_revenue) * 100
                        year_ago_margin = (year_ago_gross / year_ago_revenue) * 100
                        if latest_margin > year_ago_margin:
                            catalysts["found"] = True
                            catalysts["flags"].append(
                                f"Gross margin improving: {latest_margin:.1f}% vs {year_ago_margin:.1f}%"
                            )
                            catalysts["details"]["gross_margin_improving"] = True
            except Exception as e:
                LOGGER.debug(f"Error checking earnings for {ticker}: {e}")

            # Check for insider buys (Form 4) - FMP may have insider transactions
            try:
                # Try different endpoint names for insider trading
                insider_data = None
                for endpoint in ["insider-trading", "insider_trading", "insider-transactions"]:
                    try:
                        insider_data = self.fmp._get(endpoint, {"symbol": ticker, "limit": 10})
                        if insider_data:
                            break
                    except:
                        continue
                
                if insider_data and isinstance(insider_data, list):
                    recent_buys = []
                    for transaction in insider_data[:10]:
                        transaction_type = str(transaction.get("transactionCode", "") or transaction.get("type", "")).upper()
                        # Transaction codes: P = Purchase, S = Sale
                        if "P" in transaction_type or "BUY" in transaction_type:
                            transaction_date = transaction.get("transactionDate", "") or transaction.get("date", "")
                            if transaction_date:
                                try:
                                    tx_date = datetime.strptime(transaction_date, "%Y-%m-%d")
                                    days_ago = (datetime.now() - tx_date).days
                                    if days_ago <= 15:
                                        recent_buys.append(transaction)
                                except:
                                    pass
                    if len(recent_buys) >= 2:  # Cluster of insider buys
                        catalysts["found"] = True
                        catalysts["flags"].append(f"Insider buys: {len(recent_buys)} in last 15 days")
                        catalysts["details"]["insider_buys"] = len(recent_buys)
            except Exception as e:
                LOGGER.debug(f"Error checking insider transactions for {ticker}: {e}")

            if catalysts["found"]:
                return catalysts
            return None

        except Exception as e:
            LOGGER.warning(f"Error checking catalysts for {ticker}: {e}")
            return None

    def _get_recent_news(self, ticker: str, days: int = 30) -> list[dict[str, Any]]:
        """Get recent news for a ticker."""
        try:
            # FMP has news endpoint - try different endpoint names
            news_data = None
            for endpoint in ["stock_news", "stock-news", "news"]:
                try:
                    news_data = self.fmp._get(endpoint, {"tickers": ticker, "limit": 10})
                    if news_data:
                        break
                except:
                    continue
            
            if not news_data:
                return []
                
            if isinstance(news_data, list):
                # Filter by date
                cutoff_date = datetime.now() - timedelta(days=days)
                filtered = []
                for article in news_data:
                    published_date = article.get("publishedDate", "") or article.get("date", "")
                    if published_date:
                        try:
                            pub_date = datetime.strptime(published_date, "%Y-%m-%d %H:%M:%S")
                            if pub_date >= cutoff_date:
                                filtered.append(article)
                        except:
                            # Try alternative format
                            try:
                                pub_date = datetime.strptime(published_date, "%Y-%m-%d")
                                if pub_date >= cutoff_date:
                                    filtered.append(article)
                            except:
                                pass
                return filtered
            return []
        except Exception as e:
            LOGGER.debug(f"Error fetching news for {ticker}: {e}")
            return []

    # ---------------------------------------------------------------------
    # Step 3: Quality Filter
    # ---------------------------------------------------------------------

    def _calculate_quality_metrics(self, ticker: str) -> dict[str, Any] | None:
        """
        Calculate quality metrics using minimal history.
        Returns metrics dict or None if insufficient data.
        """
        try:
            # Get latest financials
            income = self.fmp.get_income_statement(ticker, period="quarter")
            balance = self.fmp.get_balance_sheet(ticker, period="quarter")
            cash_flow = self.fmp.get_cash_flow(ticker, period="quarter")

            if len(income) < 2 or len(balance) < 2:
                return None

            latest_inc = income[0]
            year_ago_inc = income[1] if len(income) > 1 else income[0]

            latest_bal = balance[0]
            year_ago_bal = balance[1] if len(balance) > 1 else balance[0]

            # Calculate gross margin and SGA ratio
            revenue = latest_inc.get("revenue", 0) or 0
            cogs = latest_inc.get("costOfRevenue", 0) or 0
            sga = latest_inc.get("sellingGeneralAndAdministrativeExpenses", 0) or 0

            if revenue == 0:
                return None

            gross_margin = ((revenue - cogs) / revenue) * 100 if revenue > 0 else 0
            sga_ratio = (sga / revenue) * 100 if revenue > 0 else 0

            # Operational efficiency proxy
            eficiencia = gross_margin / sga_ratio if sga_ratio > 0 else 0

            # Invested capital proxy
            total_assets = latest_bal.get("totalAssets", 0) or 0
            current_liabilities = latest_bal.get("totalCurrentLiabilities", 0) or 0
            invested_capital = total_assets - current_liabilities

            if invested_capital <= 0:
                return None

            # ROCE proxy
            operating_income = latest_inc.get("operatingIncome", 0) or 0
            tax_rate = 0.21  # Assume 21% corporate tax rate
            roce_proxy = (
                (operating_income * (1 - tax_rate)) / invested_capital
            ) * 100

            # Year-ago ROCE for delta
            year_ago_op_income = year_ago_inc.get("operatingIncome", 0) or 0
            year_ago_assets = year_ago_bal.get("totalAssets", 0) or 0
            year_ago_liabilities = year_ago_bal.get("totalCurrentLiabilities", 0) or 0
            year_ago_ic = year_ago_assets - year_ago_liabilities
            year_ago_roce = (
                ((year_ago_op_income * (1 - tax_rate)) / year_ago_ic) * 100
                if year_ago_ic > 0
                else 0
            )

            delta_roce = roce_proxy - year_ago_roce

            # FCF yield
            latest_cf = cash_flow[0] if cash_flow else {}
            op_cash_flow = latest_cf.get("operatingCashFlow", 0) or 0
            capex = abs(latest_cf.get("capitalExpenditure", 0) or 0)
            fcf = op_cash_flow - capex

            # Get market cap from quote
            quote = self.fmp.get_realtime_price(ticker)
            market_cap = None
            if quote:
                market_cap = quote.get("marketCap")
                if not market_cap:
                    price = quote.get("price", 0) or 0
                    shares = quote.get("sharesOutstanding", 0) or 0
                    market_cap = price * shares

            fcf_yield = (fcf / market_cap * 100) if market_cap and market_cap > 0 else 0

            # Net debt / EBITDA
            total_debt = latest_bal.get("totalDebt", 0) or 0
            cash_equiv = latest_bal.get("cashAndCashEquivalents", 0) or 0
            net_debt = total_debt - cash_equiv

            ebitda = latest_inc.get("ebitda", 0) or 0
            if ebitda == 0:
                # Calculate EBITDA proxy
                ebitda = operating_income + (
                    latest_inc.get("depreciationAndAmortization", 0) or 0
                )

            net_debt_ebitda = (net_debt / ebitda) if ebitda > 0 else float("inf")

            return {
                "eficiencia": eficiencia,
                "roce_proxy": roce_proxy,
                "delta_roce": delta_roce,
                "fcf_yield": fcf_yield,
                "net_debt_ebitda": net_debt_ebitda,
                "gross_margin": gross_margin,
                "sga_ratio": sga_ratio,
                "invested_capital": invested_capital,
                "market_cap": market_cap,
            }

        except Exception as e:
            LOGGER.warning(f"Error calculating quality metrics for {ticker}: {e}")
            return None

    def _passes_quality_filter(self, metrics: dict[str, Any]) -> bool:
        """Check if metrics pass quality filter."""
        eficiencia = metrics.get("eficiencia", 0)
        delta_roce = metrics.get("delta_roce", 0)
        fcf_yield = metrics.get("fcf_yield", 0)
        net_debt_ebitda = metrics.get("net_debt_ebitda", float("inf"))

        # Hard filters
        if eficiencia <= 3.0:
            return False
        if delta_roce <= 8.0:  # Must improve by at least 8 percentage points
            return False
        # FCF yield must be > 10% OR positive (turning from negative)
        # Accept if: fcf_yield > 10% OR (fcf_yield > 0 and improving)
        if fcf_yield <= 0:
            return False  # Must be positive
        if fcf_yield < 10:
            # If less than 10%, we'd need to check if it's improving from negative
            # For simplicity, require > 10% for now
            return False
        if net_debt_ebitda >= 1.5:  # Must be < 1.5x or falling fast
            return False

        return True

    # ---------------------------------------------------------------------
    # Step 4: Size & Liquidity Filter
    # ---------------------------------------------------------------------

    def _passes_size_liquidity_filter(
        self, ticker: str, metrics: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Check size and liquidity requirements.
        Returns liquidity data if passes, None otherwise.
        """
        try:
            market_cap = metrics.get("market_cap", 0)
            if not market_cap or market_cap < 50_000_000 or market_cap > 800_000_000:
                return None

            # Get quote for volume data
            quote = self.fmp.get_realtime_price(ticker)
            if not quote:
                return None

            # Get price history for volume analysis
            price_history = self.fmp.get_price_history(ticker)
            if not price_history or len(price_history) < 20:
                return None

            # Calculate average daily volume (last 20 days)
            recent_volumes = [
                day.get("volume", 0) or 0
                for day in price_history[:20]
                if day.get("volume")
            ]
            if not recent_volumes:
                return None

            avg_volume = sum(recent_volumes) / len(recent_volumes)
            current_volume = recent_volumes[0] if recent_volumes else 0

            # Filters
            if avg_volume < 300_000:  # Average daily volume > 300k
                return None

            volume_spike = (current_volume / avg_volume) if avg_volume > 0 else 0
            if volume_spike < 10:  # Current volume > 10x average
                return None

            # Get shares outstanding for free float estimate
            shares_outstanding = quote.get("sharesOutstanding", 0) or 0
            # Rough free float estimate (assume 80% of shares outstanding)
            free_float = shares_outstanding * 0.8

            if free_float >= 30_000_000:  # Free float < 30M shares
                return None

            return {
                "market_cap": market_cap,
                "avg_volume": avg_volume,
                "current_volume": current_volume,
                "volume_spike": volume_spike,
                "free_float_est": free_float,
            }

        except Exception as e:
            LOGGER.warning(f"Error checking size/liquidity for {ticker}: {e}")
            return None

    # ---------------------------------------------------------------------
    # Main Screening Logic
    # ---------------------------------------------------------------------

    def screen(self) -> list[dict[str, Any]]:
        """
        Run the complete screening process.
        Returns 0-3 high-conviction candidates.
        """
        LOGGER.info("Starting Under-the-Radar Screener...")

        # Step 1: Social momentum
        social_candidates = self._detect_social_spikes()
        if not social_candidates:
            LOGGER.info("No social momentum candidates found")
            return []

        LOGGER.info(f"Step 1 complete: {len(social_candidates)} social candidates")

        # Step 2-4: Filter each candidate
        final_candidates = []

        for candidate in social_candidates[:20]:  # Process top 20
            ticker = candidate["ticker"]

            try:
                # Step 2: Catalyst filter
                catalysts = self._check_catalysts(ticker)
                if not catalysts:
                    LOGGER.debug(f"{ticker}: No catalysts found")
                    continue

                # Step 3: Quality filter
                quality_metrics = self._calculate_quality_metrics(ticker)
                if not quality_metrics:
                    LOGGER.debug(f"{ticker}: Insufficient quality data")
                    continue

                if not self._passes_quality_filter(quality_metrics):
                    LOGGER.debug(f"{ticker}: Failed quality filter")
                    continue

                # Step 4: Size & liquidity filter
                liquidity_data = self._passes_size_liquidity_filter(ticker, quality_metrics)
                if not liquidity_data:
                    LOGGER.debug(f"{ticker}: Failed size/liquidity filter")
                    continue

                # Get company profile for name and sector
                try:
                    profile_data = self.fmp._get(f"profile/{ticker}", None)
                    if isinstance(profile_data, list) and len(profile_data) > 0:
                        profile = profile_data[0]
                        company_name = profile.get("companyName", ticker)
                        sector = profile.get("sector", "Unknown")
                    elif isinstance(profile_data, dict):
                        company_name = profile_data.get("companyName", ticker)
                        sector = profile_data.get("sector", "Unknown")
                    else:
                        company_name = ticker
                        sector = "Unknown"
                except Exception as e:
                    LOGGER.debug(f"Error fetching profile for {ticker}: {e}")
                    company_name = ticker
                    sector = "Unknown"

                # Build explanation
                explanation_parts = []
                explanation_parts.append(
                    f"Social spike in {', '.join(candidate['sources'])}"
                )
                explanation_parts.append(f"Catalyst: {', '.join(catalysts['flags'][:2])}")
                explanation_parts.append(
                    f"Quality: ROCE +{quality_metrics['delta_roce']:.1f}pp, FCF yield {quality_metrics['fcf_yield']:.1f}%"
                )
                explanation = ". ".join(explanation_parts)

                final_candidate = {
                    "ticker": ticker,
                    "name": company_name,
                    "sector": sector,
                    "social_spike": {
                        "sources": candidate["sources"],
                        "metrics": candidate["spike_metrics"],
                    },
                    "catalysts": catalysts,
                    "quality_metrics": {
                        "eficiencia": quality_metrics["eficiencia"],
                        "roce_proxy": quality_metrics["roce_proxy"],
                        "delta_roce": quality_metrics["delta_roce"],
                        "fcf_yield": quality_metrics["fcf_yield"],
                        "net_debt_ebitda": quality_metrics["net_debt_ebitda"],
                    },
                    "liquidity": liquidity_data,
                    "explanation": explanation,
                }

                final_candidates.append(final_candidate)
                LOGGER.info(f"✅ {ticker} passed all filters!")

                # Limit to 3 candidates
                if len(final_candidates) >= 3:
                    break

            except Exception as e:
                LOGGER.warning(f"Error processing {ticker}: {e}")
                continue

        LOGGER.info(f"Screening complete: {len(final_candidates)} candidates found")
        return final_candidates
