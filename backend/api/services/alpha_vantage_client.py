"""
Alpha Vantage API Client

Provides access to Alpha Vantage data as a fallback/alternative data source.
Supports:
- Real-time stock quotes
- Historical price data
- Commodity prices (WTI, Brent, etc.)
- Economic indicators (GDP, CPI, etc.)
- Crypto prices
- News Sentiment
"""

import os
import logging
import requests
from typing import Any, Dict, List, Optional
from datetime import datetime

LOGGER = logging.getLogger("caria.services.alpha_vantage_client")

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

class AlphaVantageClient:
    """
    Client for Alpha Vantage API.
    Provides financial data as fallback when FMP/OpenBB fails.
    """

    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
        if not self.api_key:
            LOGGER.warning("⚠️ ALPHA_VANTAGE_API_KEY not found. Alpha Vantage features unavailable.")
        else:
            LOGGER.info("✅ Alpha Vantage API key configured")
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage is configured and available."""
        return bool(self.api_key)

    def _make_request(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make a request to the Alpha Vantage API."""
        if not self.api_key:
            return None
        
        params["apikey"] = self.api_key
        
        try:
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API error messages
            if "Error Message" in data:
                LOGGER.error(f"Alpha Vantage error: {data['Error Message']}")
                return None
            if "Note" in data:
                LOGGER.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return None
            if "Information" in data:
                LOGGER.warning(f"Alpha Vantage info: {data['Information']}")
                return None
                
            return data
        except requests.exceptions.RequestException as e:
            LOGGER.error(f"Alpha Vantage request failed: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for a single symbol.
        Returns: {price, change, change_percent, volume, ...}
        """
        data = self._make_request({
            "function": "GLOBAL_QUOTE",
            "symbol": symbol
        })
        
        if not data or "Global Quote" not in data:
            return None
        
        quote = data["Global Quote"]
        if not quote:
            return None
            
        return {
            "symbol": symbol,
            "price": float(quote.get("05. price", 0)),
            "change": float(quote.get("09. change", 0)),
            "changesPercentage": float(quote.get("10. change percent", "0%").replace("%", "")),
            "previousClose": float(quote.get("08. previous close", 0)),
            "volume": int(quote.get("06. volume", 0)),
            "open": float(quote.get("02. open", 0)),
            "high": float(quote.get("03. high", 0)),
            "low": float(quote.get("04. low", 0)),
        }

    def get_bulk_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get quotes for multiple symbols.
        Alpha Vantage doesn't have native batch quotes, so we call individually.
        Note: This can be slow and consume API rate limits.
        """
        results = {}
        for symbol in symbols:
            try:
                quote = self.get_quote(symbol)
                if quote:
                    results[symbol] = quote
            except Exception as e:
                LOGGER.warning(f"Failed to get quote for {symbol}: {e}")
        return results

    def get_daily_prices(
        self, 
        symbol: str, 
        outputsize: str = "compact"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get daily historical prices.
        outputsize: "compact" (100 data points) or "full" (20+ years)
        """
        data = self._make_request({
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize
        })
        
        if not data or "Time Series (Daily)" not in data:
            return None
        
        time_series = data["Time Series (Daily)"]
        results = []
        
        for date_str, values in time_series.items():
            results.append({
                "date": date_str,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "adjusted_close": float(values.get("5. adjusted close", 0)),
                "volume": int(values.get("6. volume", 0)),
                "dividend": float(values.get("7. dividend amount", 0)),
                "split_coefficient": float(values.get("8. split coefficient", 1)),
            })
        
        # Sort by date ascending
        results.sort(key=lambda x: x["date"])
        return results

    def get_intraday_prices(
        self, 
        symbol: str, 
        interval: str = "5min",
        outputsize: str = "compact"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get intraday prices.
        interval: 1min, 5min, 15min, 30min, 60min
        """
        data = self._make_request({
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize
        })
        
        time_series_key = f"Time Series ({interval})"
        if not data or time_series_key not in data:
            return None
        
        time_series = data[time_series_key]
        results = []
        
        for timestamp, values in time_series.items():
            results.append({
                "timestamp": timestamp,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": int(values.get("5. volume", 0)),
            })
        
        results.sort(key=lambda x: x["timestamp"])
        return results

    # =========================================================================
    # CRYPTO DATA
    # =========================================================================

    def get_crypto_price(self, symbol: str, market: str = "USD") -> Optional[Dict[str, Any]]:
        """
        Get real-time crypto exchange rate.
        """
        data = self._make_request({
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": symbol,
            "to_currency": market
        })

        if not data or "Realtime Currency Exchange Rate" not in data:
            return None
        
        rate = data["Realtime Currency Exchange Rate"]
        return {
            "symbol": f"{symbol}{market}",
            "price": float(rate.get("5. Exchange Rate", 0)),
            "bid": float(rate.get("8. Bid Price", 0)),
            "ask": float(rate.get("9. Ask Price", 0)),
            "date": rate.get("6. Last Refreshed", "")
        }

    # =========================================================================
    # NEWS & SENTIMENT
    # =========================================================================

    def get_news_sentiment(self, tickers: str = "", limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get news sentiment for tickers.
        tickers: comma separated string
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "limit": limit,
            "sort": "LATEST"
        }
        if tickers:
            params["tickers"] = tickers

        data = self._make_request(params)
        
        if not data or "feed" not in data:
            return None
            
        return data["feed"]

    # =========================================================================
    # COMMODITY DATA
    # =========================================================================
    
    def get_wti_oil_prices(self, interval: str = "monthly") -> Optional[List[Dict[str, Any]]]:
        """Get WTI crude oil prices."""
        data = self._make_request({
            "function": "WTI",
            "interval": interval
        })
        
        if not data or "data" not in data:
            return None
        
        return [
            {"date": item["date"], "value": float(item["value"]) if item["value"] != "." else None}
            for item in data["data"]
        ]

    def get_brent_oil_prices(self, interval: str = "monthly") -> Optional[List[Dict[str, Any]]]:
        """Get Brent crude oil prices."""
        data = self._make_request({
            "function": "BRENT",
            "interval": interval
        })
        
        if not data or "data" not in data:
            return None
        
        return [
            {"date": item["date"], "value": float(item["value"]) if item["value"] != "." else None}
            for item in data["data"]
        ]

    def get_natural_gas_prices(self, interval: str = "monthly") -> Optional[List[Dict[str, Any]]]:
        """Get Henry Hub natural gas prices."""
        data = self._make_request({
            "function": "NATURAL_GAS",
            "interval": interval
        })
        
        if not data or "data" not in data:
            return None
        
        return [
            {"date": item["date"], "value": float(item["value"]) if item["value"] != "." else None}
            for item in data["data"]
        ]

    def get_copper_prices(self, interval: str = "monthly") -> Optional[List[Dict[str, Any]]]:
        """Get global copper prices."""
        data = self._make_request({
            "function": "COPPER",
            "interval": interval
        })
        
        if not data or "data" not in data:
            return None
        
        return [
            {"date": item["date"], "value": float(item["value"]) if item["value"] != "." else None}
            for item in data["data"]
        ]

    def get_commodity_prices(self, interval: str = "monthly") -> Optional[List[Dict[str, Any]]]:
        """Get global all commodities index."""
        data = self._make_request({
            "function": "ALL_COMMODITIES",
            "interval": interval
        })
        
        if not data or "data" not in data:
            return None
        
        return [
            {"date": item["date"], "value": float(item["value"]) if item["value"] != "." else None}
            for item in data["data"]
        ]

    # =========================================================================
    # ECONOMIC INDICATORS
    # =========================================================================

    def get_real_gdp(self, interval: str = "quarterly") -> Optional[List[Dict[str, Any]]]:
        """Get US Real GDP data."""
        data = self._make_request({
            "function": "REAL_GDP",
            "interval": interval
        })
        
        if not data or "data" not in data:
            return None
        
        return [
            {"date": item["date"], "value": float(item["value"]) if item["value"] != "." else None}
            for item in data["data"]
        ]

    def get_cpi(self, interval: str = "monthly") -> Optional[List[Dict[str, Any]]]:
        """Get Consumer Price Index data."""
        data = self._make_request({
            "function": "CPI",
            "interval": interval
        })
        
        if not data or "data" not in data:
            return None
        
        return [
            {"date": item["date"], "value": float(item["value"]) if item["value"] != "." else None}
            for item in data["data"]
        ]

    def get_inflation(self) -> Optional[List[Dict[str, Any]]]:
        """Get annual inflation rates."""
        data = self._make_request({"function": "INFLATION"})
        
        if not data or "data" not in data:
            return None
        
        return [
            {"date": item["date"], "value": float(item["value"]) if item["value"] != "." else None}
            for item in data["data"]
        ]

    def get_federal_funds_rate(self, interval: str = "monthly") -> Optional[List[Dict[str, Any]]]:
        """Get Federal Funds Rate data."""
        data = self._make_request({
            "function": "FEDERAL_FUNDS_RATE",
            "interval": interval
        })
        
        if not data or "data" not in data:
            return None
        
        return [
            {"date": item["date"], "value": float(item["value"]) if item["value"] != "." else None}
            for item in data["data"]
        ]

    def get_unemployment(self) -> Optional[List[Dict[str, Any]]]:
        """Get unemployment rate data."""
        data = self._make_request({"function": "UNEMPLOYMENT"})
        
        if not data or "data" not in data:
            return None
        
        return [
            {"date": item["date"], "value": float(item["value"]) if item["value"] != "." else None}
            for item in data["data"]
        ]

    def get_treasury_yield(
        self, 
        interval: str = "monthly", 
        maturity: str = "10year"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get Treasury yield data.
        maturity: 3month, 2year, 5year, 7year, 10year, 30year
        """
        data = self._make_request({
            "function": "TREASURY_YIELD",
            "interval": interval,
            "maturity": maturity
        })
        
        if not data or "data" not in data:
            return None
        
        return [
            {"date": item["date"], "value": float(item["value"]) if item["value"] != "." else None}
            for item in data["data"]
        ]

    # =========================================================================
    # TECHNICAL INDICATORS
    # =========================================================================

    def get_sma(
        self, 
        symbol: str, 
        interval: str = "daily", 
        time_period: int = 50, 
        series_type: str = "close"
    ) -> Optional[List[Dict[str, Any]]]:
        """Get Simple Moving Average."""
        data = self._make_request({
            "function": "SMA",
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type
        })
        
        if not data or "Technical Analysis: SMA" not in data:
            return None
        
        results = []
        for date_str, values in data["Technical Analysis: SMA"].items():
            results.append({
                "date": date_str,
                "sma": float(values.get("SMA", 0))
            })
        
        results.sort(key=lambda x: x["date"])
        return results

    def get_rsi(
        self, 
        symbol: str, 
        interval: str = "daily", 
        time_period: int = 14, 
        series_type: str = "close"
    ) -> Optional[List[Dict[str, Any]]]:
        """Get Relative Strength Index."""
        data = self._make_request({
            "function": "RSI",
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type
        })
        
        if not data or "Technical Analysis: RSI" not in data:
            return None
        
        results = []
        for date_str, values in data["Technical Analysis: RSI"].items():
            results.append({
                "date": date_str,
                "rsi": float(values.get("RSI", 0))
            })
        
        results.sort(key=lambda x: x["date"])
        return results


# Singleton instance
alpha_vantage_client = AlphaVantageClient()
