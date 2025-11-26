import logging
from typing import Any, Dict, Optional, List
import os
import yfinance as yf
from openbb import obb

LOGGER = logging.getLogger("caria.services.openbb_client")
DEFAULT_PROVIDER = os.getenv("OPENBB_PROVIDER", "fmp")

class OpenBBClient:
    """
    Unified client for OpenBB data integration.
    Replaces FMP and Yahoo wrappers.
    Includes robust yfinance fallback for when keys are missing.
    """

    def __init__(self):
        self.provider = DEFAULT_PROVIDER
        # Ensure FMP API key is set if available
        self._configure_fmp_key()
    
    def _configure_fmp_key(self):
        """Configure FMP API key in OpenBB if available."""
        try:
            fmp_key = os.getenv("FMP_API_KEY", "").strip()
            if fmp_key:
                obb.user.credentials.fmp_api_key = fmp_key
                LOGGER.info("✅ FMP API key configured in OpenBB")
            else:
                LOGGER.warning("⚠️ FMP_API_KEY not found in environment. FMP features may be limited.")
        except Exception as e:
            LOGGER.warning(f"Could not configure FMP key: {e}")

    def get_price_history(self, symbol: str, start_date: str = "2010-01-01") -> Any:
        """
        Fetch historical price data.
        Uses the configured provider (default fmp), falls back to yfinance.
        """
        # Try OpenBB first
        try:
            result = obb.equity.price.historical(symbol=symbol, provider=self.provider, start_date=start_date)
            if result and hasattr(result, 'to_df'):
                df = result.to_df()
                if not df.empty:
                    return result
        except Exception as e:
            LOGGER.warning(f"OpenBB/FMP history failed for {symbol}: {e}")

        # Fallback to yfinance
        try:
            LOGGER.info(f"Falling back to yfinance for {symbol} history")
            # yfinance returns a DF directly
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date)
            if not hist.empty:
                # Wrap it in a simple object to match OpenBB interface if needed, 
                # or just rely on the caller handling a dataframe-like object.
                # For now, we return a wrapper that has to_df
                class YFinanceResult:
                    def __init__(self, df): self._df = df
                    def to_df(self): return self._df
                return YFinanceResult(hist)
        except Exception as e:
            LOGGER.error(f"yfinance history failed for {symbol}: {e}")
        
        return None

    def get_multiples(self, symbol: str, limit: int = 1, period: str = "annual") -> Any:
        """
        Fetch valuation multiples (Ratios).
        """
        try:
            return obb.equity.fundamental.ratios(symbol=symbol, provider="fmp", limit=limit, period=period)
        except Exception as e:
            LOGGER.error(f"Error fetching multiples for {symbol}: {e}")
            return None

    def get_financials(self, symbol: str, limit: int = 1, period: str = "annual") -> Any:
        """
        Fetch financial statements.
        """
        try:
            return obb.equity.fundamental.income(symbol=symbol, provider="fmp", limit=limit, period=period)
        except Exception as e:
            LOGGER.error(f"Error fetching financials for {symbol}: {e}")
            return None

    def get_key_metrics(self, symbol: str, limit: int = 1, period: str = "annual") -> Any:
        """
        Fetch key metrics.
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
            return obb.equity.fundamental.cash_growth(symbol=symbol, provider="fmp", limit=limit, period=period)
        except Exception as e:
            LOGGER.error(f"Error fetching growth for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        """
        # Try OpenBB/FMP Quote
        try:
            quote = obb.equity.price.quote(symbol=symbol, provider="fmp")
            if quote and hasattr(quote, 'to_df'):
                df = quote.to_df()
                if not df.empty:
                    val = df.iloc[0].get('last_price', df.iloc[0].get('price', 0))
                    if val > 0: return float(val)
        except Exception:
            pass
            
        # Fallback to yfinance
        try:
            ticker = yf.Ticker(symbol)
            # fast_info is faster than history
            price = ticker.fast_info.last_price
            if price and price > 0:
                return float(price)
            # fallback to history if fast_info fails
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            LOGGER.error(f"Error fetching current price for {symbol}: {e}")
            
        return 0.0

    def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get latest prices for multiple symbols.
        Returns dict: {symbol: {price: float, change: float, ...}}
        """
        results = {}
        
        # 1. Try Batch FMP via OpenBB
        try:
            symbols_str = ",".join(symbols)
            quote = obb.equity.price.quote(symbol=symbols_str, provider="fmp")
            
            if quote and hasattr(quote, 'to_df'):
                df = quote.to_df()
                if not df.empty:
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
        except Exception as e:
            LOGGER.warning(f"Batch FMP fetch failed: {e}")

        # 2. Fill missing with yfinance (Batch)
        missing_symbols = [s for s in symbols if s not in results]
        if missing_symbols:
            try:
                # yfinance download is efficient for batch
                # 'download' returns a MultiIndex DataFrame if >1 ticker
                data = yf.download(missing_symbols, period="1d", progress=False)
                
                # Check if we got data. If only 1 ticker, structure is different.
                if not data.empty:
                    if len(missing_symbols) == 1:
                        # Single ticker structure
                        sym = missing_symbols[0]
                        close = float(data['Close'].iloc[-1])
                        # approximate open as prev close for change calc if needed
                        open_p = float(data['Open'].iloc[-1])
                        change = close - open_p
                        change_p = (change / open_p) * 100 if open_p else 0
                        
                        results[sym] = {
                            "symbol": sym,
                            "price": close,
                            "change": change,
                            "changesPercentage": change_p,
                            "previousClose": open_p 
                        }
                    else:
                        # Multi-ticker
                        # 'Close' column has sub-columns for each ticker
                        for sym in missing_symbols:
                            try:
                                # Handle different yfinance versions (some use MultiIndex, some don't)
                                if isinstance(data['Close'], float):
                                     # Should not happen with multi-ticker
                                     pass
                                else:
                                    # Access column safely
                                    if sym in data['Close']:
                                        close = float(data['Close'][sym].iloc[-1])
                                        open_p = float(data['Open'][sym].iloc[-1])
                                        change = close - open_p
                                        change_p = (change / open_p) * 100 if open_p else 0
                                        
                                        results[sym] = {
                                            "symbol": sym,
                                            "price": close,
                                            "change": change,
                                            "changesPercentage": change_p,
                                            "previousClose": open_p
                                        }
                            except Exception:
                                continue
            except Exception as e:
                LOGGER.error(f"yfinance batch fetch failed: {e}")

        # 3. Final cleanup - fill remaining with 0s
        for sym in symbols:
            if sym not in results:
                results[sym] = {
                    "symbol": sym, 
                    "price": 0.0, 
                    "change": 0.0, 
                    "changesPercentage": 0.0,
                    "previousClose": 0.0
                }
            
        return results

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

openbb_client = OpenBBClient()
