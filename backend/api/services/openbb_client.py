"""
Thin wrapper around the OpenBB SDK with simple caching and defensive parsing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional

from openbb import obb

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


class OpenBBClient:
    def __init__(self) -> None:
        self.cache = _OpenBBCache()

    def get_price_history(self, symbol: str, start_date: str = "2010-01-01") -> List[Dict[str, Any]]:
        cache_key = f"price:{symbol}:{start_date}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        response = obb.equity.price.historical(symbol=symbol, provider="yahoo", start_date=start_date)
        data = _unwrap_results(response)
        data.sort(key=lambda x: x.get("date") or x.get("timestamp") or "")
        self.cache.set(cache_key, data)
        return data

    def get_multiples(self, symbol: str) -> List[Dict[str, Any]]:
        cache_key = f"multiples:{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        response = obb.equity.fundamental.multiples(symbol=symbol, provider="yahoo")
        data = _unwrap_results(response)
        self.cache.set(cache_key, data)
        return data

    def get_financials(self, symbol: str) -> Dict[str, List[Dict[str, Any]]]:
        cache_key = f"financials:{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        response = obb.equity.fundamental.financials(symbol=symbol, provider="yahoo")
        payload = getattr(response, "results", response)

        financials = {
            "income_statement": _unwrap_results(payload.get("income_statement") if isinstance(payload, dict) else []),
            "balance_sheet": _unwrap_results(payload.get("balance_sheet") if isinstance(payload, dict) else []),
            "cash_flow": _unwrap_results(payload.get("cash_flow") if isinstance(payload, dict) else []),
        }
        self.cache.set(cache_key, financials)
        return financials

    def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        history = self.get_price_history(symbol)
        multiples = self.get_multiples(symbol)
        financials = self.get_financials(symbol)
        latest_price = history[-1]["close"] if history else None
        return {
            "symbol": symbol.upper(),
            "price_history": history,
            "multiples": multiples,
            "financials": financials,
            "latest_price": latest_price,
        }


openbb_client = OpenBBClient()

