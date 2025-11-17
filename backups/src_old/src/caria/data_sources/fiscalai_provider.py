"""FiscalAI provider for earnings transcripts, insider trades, and SEC filings."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from prefect import task

LOGGER = logging.getLogger("caria.data_sources.fiscalai")

FISCALAI_BASE_URL = "https://api.fiscaldata.com/v1"


class FiscalAIProvider:
    """Provider for FiscalAI alternative data."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def fetch_earnings_transcripts(
        self,
        ticker: str,
        start_date: str | None = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Fetch earnings call transcripts with sentiment analysis."""
        endpoint = f"{FISCALAI_BASE_URL}/earnings-transcripts"
        params = {
            "ticker": ticker,
            "limit": limit,
        }
        if start_date:
            params["from_date"] = start_date

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                LOGGER.warning(f"No earnings transcripts found for {ticker}")
                return pd.DataFrame()

            records = []
            for item in data["results"]:
                records.append({
                    "ticker": ticker,
                    "date": item.get("call_date"),
                    "quarter": item.get("quarter"),
                    "year": item.get("year"),
                    "sentiment_score": item.get("sentiment_score", 0.0),
                    "key_topics": ", ".join(item.get("topics", [])),
                    "ceo_tone": item.get("ceo_tone"),
                    "forward_guidance": item.get("guidance"),
                    "questions_count": item.get("qa_count", 0),
                    "transcript_length": len(item.get("transcript", "")),
                    "source": "fiscalai_earnings",
                })

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            return df

        except requests.exceptions.RequestException as e:
            LOGGER.error(f"FiscalAI API error for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_insider_trades(
        self,
        ticker: str,
        start_date: str | None = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Fetch insider trading activity."""
        endpoint = f"{FISCALAI_BASE_URL}/insider-trades"
        params = {
            "ticker": ticker,
            "limit": limit,
        }
        if start_date:
            params["from_date"] = start_date

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                LOGGER.warning(f"No insider trades found for {ticker}")
                return pd.DataFrame()

            records = []
            for item in data["results"]:
                records.append({
                    "ticker": ticker,
                    "date": item.get("transaction_date"),
                    "insider_name": item.get("insider_name"),
                    "title": item.get("title"),
                    "transaction_type": item.get("transaction_type"),  # Buy/Sell
                    "shares": item.get("shares", 0),
                    "price": item.get("price", 0.0),
                    "value": item.get("value", 0.0),
                    "shares_owned_after": item.get("shares_owned_after", 0),
                    "source": "fiscalai_insider",
                })

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])

            # Derive sentiment features
            df["is_buy"] = df["transaction_type"].str.lower().str.contains("buy|purchase")
            df["is_c_suite"] = df["title"].str.lower().str.contains("ceo|cfo|coo|president")

            return df

        except requests.exceptions.RequestException as e:
            LOGGER.error(f"FiscalAI API error for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_sec_filings(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Fetch SEC filings (10-K, 10-Q, 8-K, etc)."""
        if not filing_types:
            filing_types = ["10-K", "10-Q", "8-K", "DEF 14A"]

        endpoint = f"{FISCALAI_BASE_URL}/sec-filings"
        params = {
            "ticker": ticker,
            "filing_types": ",".join(filing_types),
            "limit": limit,
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                LOGGER.warning(f"No SEC filings found for {ticker}")
                return pd.DataFrame()

            records = []
            for item in data["results"]:
                records.append({
                    "ticker": ticker,
                    "date": item.get("filing_date"),
                    "filing_type": item.get("filing_type"),
                    "filing_url": item.get("url"),
                    "key_risks_mentioned": item.get("risk_factors_count", 0),
                    "management_discussion_length": item.get("mda_word_count", 0),
                    "litigation_mentions": item.get("litigation_count", 0),
                    "source": "fiscalai_sec",
                })

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            return df

        except requests.exceptions.RequestException as e:
            LOGGER.error(f"FiscalAI API error for {ticker}: {e}")
            return pd.DataFrame()


@task(name="fetch-fiscalai-data")
def fetch_fiscalai_data(
    api_key: str,
    tickers: list[str],
    endpoints: list[str],
    start_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch data from FiscalAI for multiple tickers and endpoints."""

    provider = FiscalAIProvider(api_key=api_key)
    results = {}

    for endpoint in endpoints:
        LOGGER.info(f"Fetching {endpoint} for {len(tickers)} tickers...")
        frames = []

        for ticker in tickers:
            LOGGER.debug(f"  Processing {ticker}...")

            if endpoint == "earnings-call-transcripts":
                df = provider.fetch_earnings_transcripts(ticker, start_date=start_date)
            elif endpoint == "insider-trades":
                df = provider.fetch_insider_trades(ticker, start_date=start_date)
            elif endpoint == "sec-filings":
                df = provider.fetch_sec_filings(ticker)
            else:
                LOGGER.warning(f"Unknown endpoint: {endpoint}")
                continue

            if not df.empty:
                frames.append(df)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            results[endpoint] = combined
            LOGGER.info(f"  {endpoint}: {len(combined)} records")
        else:
            LOGGER.warning(f"  {endpoint}: No data fetched")

    return results
