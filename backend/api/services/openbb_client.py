import logging
from typing import Any, Dict, Optional
import os
from openbb import obb

LOGGER = logging.getLogger("caria.services.openbb_client")
DEFAULT_PROVIDER = os.getenv("OPENBB_PROVIDER", "fmp")

class OpenBBClient:
    """
    Unified client for OpenBB data integration.
    Replaces FMP and Yahoo wrappers.
    """

    def __init__(self):
        # Initialize any necessary configurations here
        self.provider = DEFAULT_PROVIDER

    def get_price_history(self, symbol: str, start_date: str = "2010-01-01") -> Any:
        """
        Fetch historical price data.
        Uses the configured provider (default fmp).
        """
        try:
            result = obb.equity.price.historical(symbol=symbol, provider=self.provider, start_date=start_date)
            # Verify result contains data
            if result and hasattr(result, 'to_df'):
                df = result.to_df()
                if df.empty:
                    LOGGER.warning(f"Empty price history for {symbol} returned by provider {self.provider}")
                    return None
                return result
            LOGGER.warning(f"Price history for {symbol} returned unexpected format.")
            return None
        except Exception as e:
            LOGGER.error(f"Error fetching price history for {symbol}: {e}")
            return None

    def get_multiples(self, symbol: str, limit: int = 1, period: str = "annual") -> Any:
        """
        Fetch valuation multiples (Ratios).
        """
        try:
            # OpenBB v4 uses 'ratios' for multiples
            return obb.equity.fundamental.ratios(symbol=symbol, provider="fmp", limit=limit, period=period)
        except Exception as e:
            LOGGER.error(f"Error fetching multiples for {symbol}: {e}")
            return None

    def get_financials(self, symbol: str, limit: int = 1, period: str = "annual") -> Any:
        """
        Fetch financial statements (Income Statement as proxy for main financials).
        """
        try:
            return obb.equity.fundamental.income(symbol=symbol, provider="fmp", limit=limit, period=period)
        except Exception as e:
            LOGGER.error(f"Error fetching financials for {symbol}: {e}")
            return None

    def get_key_metrics(self, symbol: str, limit: int = 1, period: str = "annual") -> Any:
        """
        Fetch key metrics (e.g. FCF per share).
        """
        try:
            return obb.equity.fundamental.metrics(symbol=symbol, provider="fmp", limit=limit, period=period)
        except Exception as e:
            LOGGER.error(f"Error fetching key metrics for {symbol}: {e}")
            return None

    def get_growth(self, symbol: str, limit: int = 1, period: str = "annual") -> Any:
        """
        Fetch growth metrics.
        """
        try:
            # Fetch cash growth for FCF growth
            return obb.equity.fundamental.cash_growth(symbol=symbol, provider="fmp", limit=limit, period=period)
        except Exception as e:
            LOGGER.error(f"Error fetching growth for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        """
        try:
            # Try quote first
            quote = obb.equity.price.quote(symbol=symbol, provider="fmp")
            if quote and hasattr(quote, 'to_df'):
                df = quote.to_df()
                if not df.empty:
                    # FMP quote has 'last_price'
                    return float(df.iloc[0].get('last_price', df.iloc[0].get('price', 0)))
            
            # Fallback to historical
            hist = self.get_price_history(symbol)
            if hist and hasattr(hist, 'to_df'):
                df = hist.to_df()
                if not df.empty:
                    return float(df.iloc[-1]['close'])
            return 0.0
        except Exception as e:
            LOGGER.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0

    def get_current_prices(self, symbols: list[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get latest prices for multiple symbols.
        Returns dict: {symbol: {price: float, change: float, ...}}
        """
        results = {}
        try:
            # Join symbols for batch request if provider supports it
            # FMP supports comma separated
            symbols_str = ",".join(symbols)
            quote = obb.equity.price.quote(symbol=symbols_str, provider="fmp")
            
            if quote and hasattr(quote, 'to_df'):
                df = quote.to_df()
                if not df.empty:
                    # Iterate over rows
                    for _, row in df.iterrows():
                        sym = row.get('symbol')
                        if sym:
                            results[sym] = {
                                "symbol": sym,
                                "price": float(row.get('last_price', row.get('price', 0))),
                                "change": float(row.get('change', 0)),
                                "changesPercentage": float(row.get('change_percent', row.get('changesPercentage', 0))),
                                "previousClose": float(row.get('prev_close', row.get('previousClose', 0)))
                            }
            
            # Fill missing with individual calls or 0
            for sym in symbols:
                if sym not in results:
                    price = self.get_current_price(sym)
                    results[sym] = {
                        "symbol": sym,
                        "price": price,
                        "change": 0,
                        "changesPercentage": 0
                    }
            
            return results
        except Exception as e:
            LOGGER.error(f"Error fetching batch prices: {e}")
            return {s: {"symbol": s, "price": 0, "change": 0, "changesPercentage": 0} for s in symbols}

    def _obbject_to_list(self, obb_object: Any) -> list:
        """Convert OBBject to list of dictionaries."""
        if not obb_object:
            return []
        try:
            if hasattr(obb_object, 'to_df'):
                df = obb_object.to_df()
                if not df.empty:
                    # Replace NaN with None for JSON serialization
                    return df.replace({float('nan'): None}).to_dict(orient='records')
            if hasattr(obb_object, 'results'):
                if isinstance(obb_object.results, list):
                    return [
                        res.model_dump() if hasattr(res, 'model_dump') else (res if isinstance(res, dict) else res.__dict__)
                        for res in obb_object.results
                    ]
        except Exception as e:
            LOGGER.warning(f"Error converting OBBject to list: {e}")
        return []

    def get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        """
        Unified aggregator for all metrics.
        Returns a dictionary with prices, multiples, and financials as processed lists/dicts.
        """
        try:
            # Get raw OBBject results
            price_history_raw = self.get_price_history(symbol)
            multiples_raw = self.get_multiples(symbol, limit=8)  # Get more history for median calc
            income_raw = self.get_financials(symbol, limit=6)
            key_metrics_raw = self.get_key_metrics(symbol)
            growth_raw = self.get_growth(symbol)
            current_price = self.get_current_price(symbol)
            
            # Get cash flow statement separately
            try:
                cash_flow_raw = obb.equity.fundamental.cash(symbol=symbol, provider="fmp", limit=8)
            except Exception:
                cash_flow_raw = None
            
            # Get company profile for moat analysis
            try:
                profile_raw = obb.equity.profile(symbol=symbol, provider="fmp")
                profile = self._obbject_to_list(profile_raw)
                profile = profile[0] if profile else {}
            except Exception:
                profile = {}

            # Convert to processed lists
            price_history = self._obbject_to_list(price_history_raw)
            multiples = self._obbject_to_list(multiples_raw)
            income_statement = self._obbject_to_list(income_raw)
            cash_flow = self._obbject_to_list(cash_flow_raw)
            key_metrics = self._obbject_to_list(key_metrics_raw)
            growth = self._obbject_to_list(growth_raw)
            
            # Extract latest price from history if current_price failed
            if not current_price and price_history:
                current_price = price_history[-1].get('close', 0)

            return {
                "symbol": symbol,
                "price_history": price_history,
                "latest_price": current_price,
                "multiples": multiples,
                "financials": {
                    "income_statement": income_statement,
                    "cash_flow": cash_flow,
                },
                "key_metrics": key_metrics[0] if key_metrics else {},
                "growth": growth[0] if growth else {},
                "profile": profile,
            }
        except Exception as e:
            LOGGER.error(f"Error aggregating data for {symbol}: {e}")
            return {}

# Singleton instance for easy import
openbb_client = OpenBBClient()
