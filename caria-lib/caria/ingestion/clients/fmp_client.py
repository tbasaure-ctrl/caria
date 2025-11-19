"""Cliente básico para Financial Modeling Prep (FMP)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import requests


LOGGER = logging.getLogger("caria.ingestion.fmp_client")


class FMPClient:
    base_url = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            LOGGER.error("FMP_API_KEY no configurado. Verifica que el secret esté configurado en Cloud Run.")
            raise RuntimeError("FMP_API_KEY no configurado para FMPClient")
        LOGGER.debug(f"FMPClient inicializado con API key: {self.api_key[:4]}...")

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        params = params.copy() if params else {}
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        resp = requests.get(url, params=params, timeout=45)
        resp.raise_for_status()
        return resp.json()

    def get_income_statement(self, ticker: str, period: str = "quarter") -> list[dict[str, Any]]:
        return self._get(f"income-statement/{ticker}", {"period": period})

    def get_balance_sheet(self, ticker: str, period: str = "quarter") -> list[dict[str, Any]]:
        return self._get(f"balance-sheet-statement/{ticker}", {"period": period})

    def get_cash_flow(self, ticker: str, period: str = "quarter") -> list[dict[str, Any]]:
        return self._get(f"cash-flow-statement/{ticker}", {"period": period})

    def get_key_metrics(self, ticker: str, period: str = "quarter") -> list[dict[str, Any]]:
        return self._get(f"key-metrics/{ticker}", {"period": period})

    def get_financial_growth(self, ticker: str, period: str = "quarter") -> list[dict[str, Any]]:
        return self._get(f"financial-growth/{ticker}", {"period": period})

    def get_financial_ratios(self, ticker: str, period: str = "quarter") -> list[dict[str, Any]]:
        return self._get(f"financial-ratios/{ticker}", {"period": period})

    def get_price_history(self, ticker: str, start_date: str | None = None, end_date: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
        data = self._get(f"historical-price-full/{ticker}", params)
        if isinstance(data, dict):
            return data.get("historical", [])
        return []

    @lru_cache(maxsize=1)
    def get_sp500_constituents(self) -> list[str]:
        companies = self._get("sp500_constituent", None)
        return [item["symbol"] for item in companies]

    def get_delisted_companies(self, limit: int = 50) -> list[str]:
        data = self._get("delisted-companies", {"limit": limit})
        results: list[str] = []
        if isinstance(data, dict):
            items = data.get("companies", []) or data.get("data", [])
        else:
            items = data
        for item in items or []:
            if isinstance(item, dict) and item.get("symbol"):
                results.append(item["symbol"])
        return results[:limit]

    def get_top_performers(self, limit: int = 50) -> list[str]:
        data = self._get("stock/gainers", None)
        items: list[Any]
        if isinstance(data, dict):
            items = data.get("mostGainerStock") or data.get("gainers", [])
        else:
            items = data if isinstance(data, list) else []

        symbols: list[str] = []
        for item in items:
            if isinstance(item, dict):
                ticker = item.get("ticker") or item.get("symbol")
                if ticker:
                    symbols.append(ticker)
            elif isinstance(item, str):
                symbols.append(item)
        return symbols[:limit]

    def get_realtime_price(self, ticker: str) -> dict[str, Any] | None:
        """Obtiene precio en tiempo real de un ticker.
        
        Args:
            ticker: Símbolo del ticker
            
        Returns:
            Dict con precio, cambio, cambio porcentual, etc. o None si falla
        """
        try:
            data = self._get(f"quote/{ticker}", None)
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict):
                return data
            return None
        except Exception as exc:
            LOGGER.warning("Error obteniendo precio en tiempo real para %s: %s", ticker, exc)
            return None

    def get_realtime_prices_batch(self, tickers: list[str]) -> dict[str, dict[str, Any]]:
        """Obtiene precios en tiempo real de múltiples tickers.
        
        Args:
            tickers: Lista de símbolos
            
        Returns:
            Dict con ticker como key y datos de precio como value
        """
        if not tickers:
            return {}
        
        # FMP permite hasta 3 tickers por request en el endpoint quote
        # Para más tickers, usar endpoint quote-short o hacer requests en batch
        results: dict[str, dict[str, Any]] = {}
        
        # Dividir en chunks de 3
        chunk_size = 3
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            tickers_str = ",".join(chunk)
            
            try:
                data = self._get(f"quote/{tickers_str}", None)
                if isinstance(data, list):
                    for item in data:
                        ticker = item.get("symbol") or item.get("ticker")
                        if ticker:
                            results[ticker] = item
                elif isinstance(data, dict):
                    ticker = data.get("symbol") or data.get("ticker")
                    if ticker:
                        results[ticker] = data
            except Exception as exc:
                LOGGER.warning("Error obteniendo precios batch para %s: %s", tickers_str, exc)
        
        return results

