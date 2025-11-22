"""
Thin wrapper around the OpenBB SDK with simple caching and defensive parsing.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional

DEFAULT_OPENBB_EXTENSIONS = ",".join(
    [
        "benzinga@1.5.0",
        "bls@1.2.0",
        "cftc@1.2.0",
        "commodity@1.4.0",
        "congress_gov@1.1.0",
        "crypto@1.5.0",
        "currency@1.5.0",
        "derivatives@1.5.0",
        "econdb@1.4.0",
        "economy@1.5.0",
        "equity@1.5.0",
        "etf@1.5.0",
        "federal_reserve@1.5.0",
        "fixedincome@1.5.0",
        "fmp@1.5.1",
        "fred@1.5.0",
        "imf@1.2.0",
        "index@1.5.0",
        "intrinio@1.5.0",
        "news@1.5.0",
        "oecd@1.5.0",
        "polygon@1.5.0",
        "regulators@1.5.0",
        "sec@1.5.0",
        "tiingo@1.5.0",
        "tradingeconomics@1.5.0",
        "us_eia@1.2.0",
        "uscongress@1.1.0",
        "yfinance@1.5.0",
    ]
)

os.environ.setdefault("OPENBB_EXTENSION_LIST", DEFAULT_OPENBB_EXTENSIONS)
os.environ.setdefault("OPENBB_FORCE_EXTENSION_BUILD", "true")
os.environ.setdefault("OPENBB_USER_DATA_PATH", "/tmp/openbb")

PRICE_PROVIDER = os.getenv("OPENBB_PRICE_PROVIDER", "yfinance")
FUNDAMENTAL_PROVIDER = os.getenv("OPENBB_FUNDAMENTAL_PROVIDER", "yfinance")

from openbb import obb
from psycopg2.extras import Json

from api.dependencies import open_db_connection
LOGGER = logging.getLogger("caria.api.services.openbb")

CACHE_TTL = timedelta(hours=6)


def _unwrap_results(result: Any) -> List[Dict[str, Any]]:
    if result is None:
        return []

    if hasattr(result, "results"):
        payload = result.results
    elif isinstance(result, dict) and "results" in result:
        payload = result["results"]
    else:
        payload = result

    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


class _OpenBBCache:
    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self.store.get(key)
        if not entry:
            return None
        if datetime.utcnow() - entry["ts"] > CACHE_TTL:
            return None
        return entry["value"]

    def set(self, key: str, value: Any) -> None:
        self.store[key] = {"value": value, "ts": datetime.utcnow()}


class OpenBBPersistentCache:
    """Neon-backed cache so OpenBB snapshots survive cold starts."""

    def __init__(self) -> None:
        self._table_ready = False

    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        conn = None
        try:
            conn = open_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS openbb_cache (
                        cache_key TEXT PRIMARY KEY,
                        payload JSONB NOT NULL,
                        fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            conn.commit()
            self._table_ready = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Unable to prepare openbb_cache table: %s", exc)
        finally:
            if conn:
                conn.close()

    def get(self, key: str, max_age_hours: int = 24) -> Optional[Any]:
        self._ensure_table()
        conn = None
        try:
            conn = open_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT payload, fetched_at FROM openbb_cache WHERE cache_key = %s",
                    (key,),
                )
                row = cursor.fetchone()
            if not row:
                return None
            payload, fetched_at = row
            if fetched_at and datetime.utcnow() - fetched_at > timedelta(hours=max_age_hours):
                return None
            return payload
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Persistent OpenBB cache miss for %s: %s", key, exc)
            return None
        finally:
            if conn:
                conn.close()

    def set(self, key: str, value: Any) -> None:
        self._ensure_table()
        conn = None
        try:
            conn = open_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO openbb_cache (cache_key, payload, fetched_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (cache_key)
                    DO UPDATE SET payload = EXCLUDED.payload, fetched_at = NOW()
                    """,
                    (key, Json(value)),
                )
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to persist OpenBB cache for %s: %s", key, exc)
        finally:
            if conn:
                conn.close()


class OpenBBClient:
    def __init__(self) -> None:
        self.cache = _OpenBBCache()
        self.persistent_cache = OpenBBPersistentCache()

    def get_price_history(self, symbol: str, start_date: str = "2010-01-01") -> List[Dict[str, Any]]:
        cache_key = f"price:{symbol}:{start_date}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        persisted = self.persistent_cache.get(cache_key, max_age_hours=24)
        if persisted is not None:
            self.cache.set(cache_key, persisted)
            return persisted

        response = obb.equity.price.historical(symbol=symbol, provider=PRICE_PROVIDER, start_date=start_date)
        data = _unwrap_results(response)
        data.sort(key=lambda x: x.get("date") or x.get("timestamp") or "")
        self.cache.set(cache_key, data)
        self.persistent_cache.set(cache_key, data)
        return data

    def get_multiples(self, symbol: str) -> List[Dict[str, Any]]:
        cache_key = f"multiples:{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        persisted = self.persistent_cache.get(cache_key, max_age_hours=12)
        if persisted is not None:
            self.cache.set(cache_key, persisted)
            return persisted

        response = obb.equity.fundamental.multiples(symbol=symbol, provider=FUNDAMENTAL_PROVIDER)
        data = _unwrap_results(response)
        self.cache.set(cache_key, data)
        self.persistent_cache.set(cache_key, data)
        return data

    def get_financials(self, symbol: str) -> Dict[str, List[Dict[str, Any]]]:
        cache_key = f"financials:{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        persisted = self.persistent_cache.get(cache_key, max_age_hours=48)
        if persisted is not None:
            self.cache.set(cache_key, persisted)
            return persisted

        response = obb.equity.fundamental.financials(symbol=symbol, provider=FUNDAMENTAL_PROVIDER)
        payload = getattr(response, "results", response)

        financials = {
            "income_statement": _unwrap_results(payload.get("income_statement") if isinstance(payload, dict) else []),
            "balance_sheet": _unwrap_results(payload.get("balance_sheet") if isinstance(payload, dict) else []),
            "cash_flow": _unwrap_results(payload.get("cash_flow") if isinstance(payload, dict) else []),
        }
        self.cache.set(cache_key, financials)
        self.persistent_cache.set(cache_key, financials)
        return financials

    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        cache_key = f"profile:{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        persisted = self.persistent_cache.get(cache_key, max_age_hours=168)
        if persisted is not None:
            self.cache.set(cache_key, persisted)
            return persisted

        response = obb.equity.profile(symbol=symbol, provider="yahoo")
        data = _unwrap_results(response)
        profile = data[0] if data else {}
        self.cache.set(cache_key, profile)
        self.persistent_cache.set(cache_key, profile)
        return profile

    def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        history = self.get_price_history(symbol)
        multiples = self.get_multiples(symbol)
        financials = self.get_financials(symbol)
        profile = self.get_company_profile(symbol)
        latest_price = history[-1]["close"] if history else None
        return {
            "symbol": symbol.upper(),
            "price_history": history,
            "multiples": multiples,
            "financials": financials,
            "profile": profile,
            "latest_price": latest_price,
        }


openbb_client = OpenBBClient()

