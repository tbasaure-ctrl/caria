"""
Fundamentals Cache Service - Dynamic Universe Expansion

This service manages a growing cache of stock fundamentals:
- Loads initial 128 stocks from parquet files (static cache)
- Fetches new stocks from OpenBB on-demand (dynamic cache)
- Stores fetched data in PostgreSQL for future use
- Enables Alpha Picker universe to grow organically

Cache Strategy:
1. Check static parquet files first (instant)
2. Check PostgreSQL cache second (fast)
3. Fetch from OpenBB if not found (slow, then cached)
"""

import logging
import pandas as pd
import psycopg2
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from api.services.openbb_client import OpenBBClient
from api.dependencies import open_db_connection

LOGGER = logging.getLogger("caria.services.fundamentals_cache")


class FundamentalsCacheService:
    """
    Service for managing expandable fundamentals cache.
    Combines static parquet files with dynamic PostgreSQL cache.
    """
    
    def __init__(self):
        self.obb_client = OpenBBClient()
        self.static_cache_loaded = False
        self.static_tickers = set()
        
        # Paths to static parquet files (initial 128 stocks)
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.quality_path = self.base_dir / "data" / "silver" / "fundamentals" / "quality_signals.parquet"
        self.value_path = self.base_dir / "data" / "silver" / "fundamentals" / "value_signals.parquet"
        
        # Cache TTL (refresh after 24 hours)
        self.cache_ttl = timedelta(hours=24)
    
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Get fundamentals for a ticker. Uses cache-first strategy.
        
        Returns:
            {
                "ticker": str,
                "data": dict with quality/value metrics,
                "source": "static_cache" | "dynamic_cache" | "realtime_fetched",
                "fetched_at": datetime
            }
        """
        ticker = ticker.upper()
        
        # 1. Check static cache (parquet files - 128 stocks)
        static_data = self._get_from_static_cache(ticker)
        if static_data:
            return {
                "ticker": ticker,
                "data": static_data,
                "source": "static_cache",
                "fetched_at": datetime.now()
            }
        
        # 2. Check dynamic cache (PostgreSQL)
        dynamic_data = self._get_from_dynamic_cache(ticker)
        if dynamic_data:
            # Check if data is stale
            fetched_at = dynamic_data.get("fetched_at")
            if fetched_at and datetime.now() - fetched_at < self.cache_ttl:
                return {
                    "ticker": ticker,
                    "data": dynamic_data["data"],
                    "source": "dynamic_cache",
                    "fetched_at": fetched_at
                }
            else:
                LOGGER.info(f"Cache for {ticker} is stale, refetching...")
        
        # 3. Fetch from OpenBB (real-time)
        try:
            fresh_data = self._fetch_from_openbb(ticker)
            
            # Save to dynamic cache
            self._save_to_dynamic_cache(ticker, fresh_data)
            
            return {
                "ticker": ticker,
                "data": fresh_data,
                "source": "realtime_fetched",
                "fetched_at": datetime.now()
            }
        except Exception as e:
            LOGGER.error(f"Error fetching fundamentals for {ticker}: {e}")
            # If we have stale cache data, return it with warning
            if dynamic_data:
                LOGGER.warning(f"Returning stale cache data for {ticker}")
                return {
                    "ticker": ticker,
                    "data": dynamic_data["data"],
                    "source": "dynamic_cache_stale",
                    "fetched_at": dynamic_data.get("fetched_at"),
                    "error": str(e)
                }
            raise
    
    def _get_from_static_cache(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Load from parquet files (initial 128 stocks)."""
        try:
            if not self.quality_path.exists() or not self.value_path.exists():
                return None
            
            # Load parquet files
            quality_df = pd.read_parquet(self.quality_path)
            value_df = pd.read_parquet(self.value_path)
            
            # Filter for this ticker
            quality_data = quality_df[quality_df['ticker'] == ticker]
            value_data = value_df[value_df['ticker'] == ticker]
            
            if quality_data.empty or value_data.empty:
                return None
            
            # Get most recent data
            if 'date' in quality_data.columns:
                quality_data = quality_data.sort_values('date').tail(1)
            if 'date' in value_data.columns:
                value_data = value_data.sort_values('date').tail(1)
            
            # Combine into single dict
            result = {}
            
            # Quality metrics
            for col in ['roic', 'roiic', 'returnOn

Equity', 'returnOnAssets', 
                       'grossProfitMargin', 'netProfitMargin', 'freeCashFlowPerShare',
                       'freeCashFlowYield', 'capitalExpenditures', 'r_and_d']:
                if col in quality_data.columns:
                    result[col] = float(quality_data[col].iloc[0]) if pd.notna(quality_data[col].iloc[0]) else None
            
            # Value metrics
            for col in ['priceToBookRatio', 'priceToSalesRatio', 'enterpriseValue',
                       'marketCap', 'revenueGrowth', 'netIncomeGrowth', 
                       'operatingIncomeGrowth', 'totalDebt', 'cashAndCashEquivalents', 'net_debt']:
                if col in value_data.columns:
                    result[col] = float(value_data[col].iloc[0]) if pd.notna(value_data[col].iloc[0]) else None
            
            self.static_tickers.add(ticker)
            return result
            
        except Exception as e:
            LOGGER.debug(f"Error loading {ticker} from static cache: {e}")
            return None
    
    def _get_from_dynamic_cache(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Load from PostgreSQL cache."""
        try:
            conn = open_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    ticker, fetched_at, updated_at,
                    roic, roiic, return_on_equity, return_on_assets,
                    gross_profit_margin, net_profit_margin,
                    free_cashflow_per_share, free_cashflow_yield,
                    capital_expenditures, r_and_d,
                    price_to_book_ratio, price_to_sales_ratio,
                    enterprise_value, market_cap,
                    revenue_growth, net_income_growth, operating_income_growth,
                    total_debt, cash_and_equivalents, net_debt,
                    company_name, sector, industry
                FROM fundamentals_cache
                WHERE ticker = %s
                LIMIT 1;
            """, (ticker,))
            
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not row:
                return None
            
            # Map to dict
            col_names = [
                'ticker', 'fetched_at', 'updated_at',
                'roic', 'roiic', 'returnOnEquity', 'returnOnAssets',
                'grossProfitMargin', 'netProfitMargin',
                'freeCashFlowPerShare', 'freeCashFlowYield',
                'capitalExpenditures', 'r_and_d',
                'priceToBookRatio', 'priceToSalesRatio',
                'enterpriseValue', 'marketCap',
                'revenueGrowth', 'netIncomeGrowth', 'operatingIncomeGrowth',
                'totalDebt', 'cashAndCashEquivalents', 'net_debt',
                'company_name', 'sector', 'industry'
            ]
            
            data = dict(zip(col_names, row))
            
            return {
                "data": data,
                "fetched_at": data.pop('fetched_at'),
                "updated_at": data.pop('updated_at')
            }
            
        except Exception as e:
            LOGGER.debug(f"Error loading {ticker} from dynamic cache: {e}")
            return None
    
    def _fetch_from_openbb(self, ticker: str) -> Dict[str, Any]:
        """Fetch all fundamentals from OpenBB."""
        LOGGER.info(f"Fetching fundamentals for {ticker} from OpenBB...")
        
        result = {}
        
        try:
            # Get key metrics (quality data)
            metrics = self.obb_client.get_key_metrics(ticker, limit=1)
            if metrics:
                metrics_data = self._extract_first(metrics)
                if metrics_data:
                    # Map to our schema
                    result['roic'] = metrics_data.get('roic')
                    result['returnOnEquity'] = metrics_data.get('roe') or metrics_data.get('returnOnEquity')
                    result['returnOnAssets'] = metrics_data.get('roa') or metrics_data.get('returnOnAssets')
                    result['grossProfitMargin'] = metrics_data.get('grossProfitMargin')
                    result['netProfitMargin'] = metrics_data.get('netProfitMargin')
                    result['freeCashFlowPerShare'] = metrics_data.get('freeCashFlowPerShare')
                    result['freeCashFlowYield'] = metrics_data.get('freeCashFlowYield')
            
            # Get multiples (value data)
            multiples = self.obb_client.get_multiples(ticker, limit=1)
            if multiples:
                multiples_data = self._extract_first(multiples)
                if multiples_data:
                    result['priceToBookRatio'] = multiples_data.get('priceToBookRatio')
                    result['priceToSalesRatio'] = multiples_data.get('priceToSalesRatio')
                    result['enterpriseValue'] = multiples_data.get('enterpriseValue')
                    result['marketCap'] = multiples_data.get('marketCap')
            
            # Get growth metrics
            growth = self.obb_client.get_growth(ticker, limit=1)
            if growth:
                growth_data = self._extract_first(growth)
                if growth_data:
                    result['revenueGrowth'] = growth_data.get('revenueGrowth')
                    result['netIncomeGrowth'] = growth_data.get('netIncomeGrowth')
                    result['operatingIncomeGrowth'] = growth_data.get('operatingIncomeGrowth')
            
            # Get financials (for debt data)
            financials = self.obb_client.get_financials(ticker, limit=1)
            if financials:
                fin_data = self._extract_first(financials)
                if fin_data:
                    result['totalDebt'] = fin_data.get('totalDebt')
                    result['cashAndCashEquivalents'] = fin_data.get('cashAndCashEquivalents')
                    if result.get('totalDebt') and result.get('cashAndCashEquivalents'):
                        result['net_debt'] = result['totalDebt'] - result['cashAndCashEquivalents']
            
            # Get company info
            ticker_data = self.obb_client.get_ticker_data(ticker)
            if ticker_data:
                result['company_name'] = ticker_data.get('name') or ticker_data.get('longName')
                result['sector'] = ticker_data.get('sector')
                result['industry'] = ticker_data.get('industry')
            
            return result
            
        except Exception as e:
            LOGGER.error(f"Error fetching from OpenBB for {ticker}: {e}")
            raise
    
    def _extract_first(self, obb_object: Any) -> Optional[Dict]:
        """Extract first row/result from OpenBB object."""
        try:
            if hasattr(obb_object, 'to_dict'):
                data = obb_object.to_dict('records')
                return data[0] if data else None
            elif hasattr(obb_object, 'results'):
                results = obb_object.results
                if isinstance(results, list) and results:
                    result = results[0]
                    if hasattr(result, '__dict__'):
                        return vars(result)
                    return result
            return None
        except Exception as e:
            LOGGER.debug(f"Error extracting data: {e}")
            return None
    
    def _save_to_dynamic_cache(self, ticker: str, data: Dict[str, Any]):
        """Save fundamentals to PostgreSQL cache."""
        try:
            conn = open_db_connection()
            cursor = conn.cursor()
            
            # Upsert (INSERT ... ON CONFLICT UPDATE)
            cursor.execute("""
                INSERT INTO fundamentals_cache (
                    ticker, fetched_at, updated_at,
                    roic, roiic, return_on_equity, return_on_assets,
                    gross_profit_margin, net_profit_margin,
                    free_cashflow_per_share, free_cashflow_yield,
                    capital_expenditures, r_and_d,
                    price_to_book_ratio, price_to_sales_ratio,
                    enterprise_value, market_cap,
                    revenue_growth, net_income_growth, operating_income_growth,
                    total_debt, cash_and_equivalents, net_debt,
                    company_name, sector, industry
                ) VALUES (
                    %s, NOW(), NOW(),
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (ticker) 
                DO UPDATE SET
                    updated_at = NOW(),
                    roic = EXCLUDED.roic,
                    roiic = EXCLUDED.roiic,
                    return_on_equity = EXCLUDED.return_on_equity,
                    return_on_assets = EXCLUDED.return_on_assets,
                    gross_profit_margin = EXCLUDED.gross_profit_margin,
                    net_profit_margin = EXCLUDED.net_profit_margin,
                    free_cashflow_per_share = EXCLUDED.free_cashflow_per_share,
                    free_cashflow_yield = EXCLUDED.free_cashflow_yield,
                    capital_expenditures = EXCLUDED.capital_expenditures,
                    r_and_d = EXCLUDED.r_and_d,
                    price_to_book_ratio = EXCLUDED.price_to_book_ratio,
                    price_to_sales_ratio = EXCLUDED.price_to_sales_ratio,
                    enterprise_value = EXCLUDED.enterprise_value,
                    market_cap = EXCLUDED.market_cap,
                    revenue_growth = EXCLUDED.revenue_growth,
                    net_income_growth = EXCLUDED.net_income_growth,
                    operating_income_growth = EXCLUDED.operating_income_growth,
                    total_debt = EXCLUDED.total_debt,
                    cash_and_equivalents = EXCLUDED.cash_and_equivalents,
                    net_debt = EXCLUDED.net_debt,
                    company_name = EXCLUDED.company_name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    fetch_count = fundamentals_cache.fetch_count + 1;
            """, (
                ticker,
                data.get('roic'), data.get('roiic'), 
                data.get('returnOnEquity'), data.get('returnOnAssets'),
                data.get('grossProfitMargin'), data.get('netProfitMargin'),
                data.get('freeCashFlowPerShare'), data.get('freeCashFlowYield'),
                data.get('capitalExpenditures'), data.get('r_and_d'),
                data.get('priceToBookRatio'), data.get('priceToSalesRatio'),
                data.get('enterpriseValue'), data.get('marketCap'),
                data.get('revenueGrowth'), data.get('netIncomeGrowth'), 
                data.get('operatingIncomeGrowth'),
                data.get('totalDebt'), data.get('cashAndCashEquivalents'), 
                data.get('net_debt'),
                data.get('company_name'), data.get('sector'), data.get('industry')
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            LOGGER.info(f"✅ Saved {ticker} to dynamic cache")
            
        except Exception as e:
            LOGGER.error(f"Error saving {ticker} to cache: {e}")
            raise
    
    def get_all_cached_tickers(self) -> List[str]:
        """
        Get all tickers in cache (static + dynamic).
        Used by Alpha Picker to know the screening universe.
        """
        tickers = set()
        
        # Get static cache tickers
        try:
            if self.quality_path.exists():
                df = pd.read_parquet(self.quality_path)
                tickers.update(df['ticker'].unique())
        except Exception as e:
            LOGGER.debug(f"Error loading static tickers: {e}")
        
        # Get dynamic cache tickers
        try:
            conn = open_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT ticker FROM fundamentals_cache;")
            dynamic_tickers = [row[0] for row in cursor.fetchall()]
            tickers.update(dynamic_tickers)
            cursor.close()
            conn.close()
        except Exception as e:
            LOGGER.debug(f"Error loading dynamic tickers: {e}")
        
        return sorted(list(tickers))
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        try:
            static_count = 0
            if self.quality_path.exists():
                df = pd.read_parquet(self.quality_path)
                static_count = df['ticker'].nunique()
            
            conn = open_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fundamentals_cache;")
            dynamic_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return {
                "static_cache_count": static_count,
                "dynamic_cache_count": dynamic_count,
                "total_universe": static_count + dynamic_count,
                "growth": dynamic_count
            }
        except Exception as e:
            LOGGER.error(f"Error getting cache stats: {e}")
            return {
                "error": str(e)
            }


# Singleton instance
_cache_service = None

def get_fundamentals_cache_service() -> FundamentalsCacheService:
    """Get singleton instance of FundamentalsCacheService."""
    global _cache_service
    if _cache_service is None:
        _cache_service = FundamentalsCacheService()
    return _cache_service
