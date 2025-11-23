import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from .openbb_client import OpenBBClient

LOGGER = logging.getLogger("caria.services.simulation")

CRISIS_DATES = {
    "1929_depression": {"start": "1929-09-01", "end": "1932-06-01", "name": "Great Depression (1929)"},
    "1939_wwii": {"start": "1939-09-01", "end": "1945-09-01", "name": "WWII Start (1939)"},
    "1962_cuban_missile": {"start": "1962-10-16", "end": "1962-11-20", "name": "Cuban Missile Crisis (1962)"},
    "1963_jfk": {"start": "1963-11-22", "end": "1964-06-01", "name": "Kennedy Assassination (1963)"},
    "1987_black_monday": {"start": "1987-10-01", "end": "1988-03-01", "name": "Black Monday (1987)"},
    "2000_dot_com": {"start": "2000-03-10", "end": "2002-10-09", "name": "Dot Com Bubble (2000)"},
    "2001_911": {"start": "2001-09-11", "end": "2002-01-01", "name": "9/11 Attacks (2001)"},
    "2008_gfc": {"start": "2008-09-01", "end": "2009-03-09", "name": "Global Financial Crisis (2008)"},
    "2011_euro_debt": {"start": "2011-04-01", "end": "2012-07-01", "name": "European Debt Crisis (2011)"},
    "2018_trade_war": {"start": "2018-09-20", "end": "2018-12-24", "name": "2018 Trade War / Fed Tightening"},
    "2020_covid": {"start": "2020-02-19", "end": "2020-03-23", "name": "COVID-19 Crash (2020)"},
    "2022_inflation": {"start": "2022-01-03", "end": "2022-10-12", "name": "2022 Inflation Bear Market"},
}

class SimulationService:
    def __init__(self):
        self.obb_client = OpenBBClient()

    def simulate_crisis(self, portfolio: List[Dict[str, Any]], crisis_id: str) -> Dict[str, Any]:
        """
        Simulate portfolio performance during a specific historical crisis.
        
        Args:
            portfolio: List of dicts with 'ticker' and 'weight' (or 'quantity').
            crisis_id: Key from CRISIS_DATES.
            
        Returns:
            Dict with 'dates', 'portfolio_values', 'benchmark_values', 'metrics'.
        """
        if crisis_id not in CRISIS_DATES:
            raise ValueError(f"Invalid crisis_id: {crisis_id}")
            
        crisis = CRISIS_DATES[crisis_id]
        start_date = crisis["start"]
        end_date = crisis["end"]
        
        # 1. Fetch historical data for portfolio assets
        # Note: Many modern assets won't have data for 1929. We need a proxy strategy.
        # Strategy: 
        # - If asset existed, use its data.
        # - If not, map sector/beta to a proxy or use S&P 500 with a beta adjustment?
        # - For MVP: If asset data missing, assume it tracks SPY (S&P 500) perfectly (beta=1).
        
        # Fetch Benchmark (SPY or GSPC)
        # FMP 'historical-price-full/^GSPC'? OpenBB usually maps 'SPY' well.
        benchmark_symbol = "SPY" 
        # For very old dates, SPY (inception 1993) won't work. Need ^GSPC or proxy.
        # OpenBB/FMP might provide ^GSPC.
        if int(start_date[:4]) < 1993:
            benchmark_symbol = "^GSPC"

        benchmark_data = self.obb_client.get_price_history(benchmark_symbol, start_date=start_date, end_date=end_date)
        
        # Process Benchmark
        bench_df = pd.DataFrame()
        if benchmark_data and hasattr(benchmark_data, 'to_df'):
            bench_df = benchmark_data.to_df()
        
        if bench_df.empty:
             # Fallback if no benchmark data (unlikely for FMP ^GSPC)
             return {"error": "Could not fetch benchmark data for this period."}
             
        # Normalize benchmark to 100
        # Ensure index is datetime
        if 'date' in bench_df.columns:
            bench_df['date'] = pd.to_datetime(bench_df['date'])
            bench_df.set_index('date', inplace=True)
        elif not isinstance(bench_df.index, pd.DatetimeIndex):
             # Try to convert index if it's string
             try:
                bench_df.index = pd.to_datetime(bench_df.index)
             except:
                pass
        
        # Use 'close' column
        if 'close' not in bench_df.columns:
             return {"error": "Benchmark data missing 'close' column."}

        # Sort by date
        bench_df.sort_index(inplace=True)
        
        initial_bench = bench_df['close'].iloc[0]
        bench_df['normalized'] = (bench_df['close'] / initial_bench) * 100
        
        # 2. Construct Portfolio History
        # We need to combine asset histories based on weights.
        # If weights not provided, calculate from quantity * current_price (passed in portfolio?)
        # For simplicity, let's assume 'weight' is passed or we treat equal weight if missing.
        
        total_weight = sum(p.get('weight', 0) for p in portfolio)
        if total_weight == 0:
            # Assign equal weights
            weight = 1.0 / len(portfolio)
            for p in portfolio:
                p['weight'] = weight
        
        portfolio_series = pd.Series(0.0, index=bench_df.index)
        valid_assets = 0
        
        for asset in portfolio:
            ticker = asset['ticker']
            weight = asset.get('weight', 0)
            
            # Fetch history
            hist = self.obb_client.get_price_history(ticker, start_date=start_date, end_date=end_date)
            asset_df = pd.DataFrame()
            if hist and hasattr(hist, 'to_df'):
                asset_df = hist.to_df()
            
            if not asset_df.empty:
                if 'date' in asset_df.columns:
                    asset_df['date'] = pd.to_datetime(asset_df['date'])
                    asset_df.set_index('date', inplace=True)
                elif not isinstance(asset_df.index, pd.DatetimeIndex):
                    try:
                        asset_df.index = pd.to_datetime(asset_df.index)
                    except:
                        pass
                
                # Reindex to match benchmark dates (fill fwd/bwd)
                asset_df = asset_df.reindex(bench_df.index, method='ffill').fillna(method='bfill')
                
                # Normalize
                if 'close' in asset_df.columns and not asset_df['close'].isnull().all():
                    initial_price = asset_df['close'].iloc[0]
                    if initial_price > 0:
                        normalized = (asset_df['close'] / initial_price) * 100
                        portfolio_series += normalized * weight
                        valid_assets += 1
                    else:
                        # Fallback: asset tracks benchmark
                        portfolio_series += bench_df['normalized'] * weight
                else:
                    # Fallback: asset tracks benchmark
                    portfolio_series += bench_df['normalized'] * weight
            else:
                # Asset didn't exist or no data -> Assume it tracks benchmark (beta=1 assumption for missing data)
                # This is a simplification. A better approach would be sector proxy.
                portfolio_series += bench_df['normalized'] * weight
        
        # 3. Calculate Metrics
        # Max Drawdown
        rolling_max = portfolio_series.cummax()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Recovery Time (days to reach previous peak after max drawdown)
        # Simplified: Days from max drawdown to end or recovery
        
        return {
            "crisis_name": crisis["name"],
            "dates": bench_df.index.strftime('%Y-%m-%d').tolist(),
            "portfolio_values": portfolio_series.tolist(),
            "benchmark_values": bench_df['normalized'].tolist(),
            "metrics": {
                "max_drawdown": float(max_drawdown),
                "total_return": float((portfolio_series.iloc[-1] / 100) - 1),
                "benchmark_return": float((bench_df['normalized'].iloc[-1] / 100) - 1)
            }
        }

    def simulate_macro(self, portfolio: List[Dict[str, Any]], params: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate portfolio reaction to macro shocks (Sensitivity Analysis).
        Params: {'inflation': 2.0, 'rates': 0.5, 'gdp': -1.0} (percentage points change)
        """
        # Simplified Factor Model
        # We need Beta to SPY, and maybe sector sensitivities.
        # For MVP: Use Beta * Market Shock + Sector Shock
        
        # 1. Define Shocks based on params
        # Inflation up -> Tech down, Energy up, Consumer Discretionary down
        # Rates up -> Growth down, Financials up (sometimes), Utilities down
        # GDP down -> Cyclicals down, Defensives stable
        
        inflation_shock = params.get('inflation', 0)
        rates_shock = params.get('rates', 0)
        gdp_shock = params.get('gdp', 0)
        
        # Base Market Shock (S&P 500 approx reaction)
        # Rates +1% -> Market -5% approx
        # GDP -1% -> Market -3% approx
        market_shock_pct = (rates_shock * -5.0) + (gdp_shock * 3.0) + (inflation_shock * -2.0)
        
        results = []
        total_impact = 0.0
        
        for asset in portfolio:
            ticker = asset['ticker']
            weight = asset.get('weight', 0)
            
            # Get Beta (from OpenBB/FMP)
            # For speed, we might want to cache this or fetch in batch.
            # Here we'll fetch individually or use default 1.0
            try:
                metrics = self.obb_client.get_key_metrics(ticker)
                # FMP key metrics doesn't always have beta. Profile does.
                # Let's try to get beta from profile or calculate it?
                # For MVP, default beta = 1.2 for Tech, 0.8 for others?
                # Let's just use 1.0 if not found for now to keep it fast.
                beta = 1.0 
            except:
                beta = 1.0
                
            # Asset Impact = Beta * Market Shock
            # Refine with sector if possible.
            
            asset_impact = beta * market_shock_pct
            
            results.append({
                "ticker": ticker,
                "impact_pct": asset_impact,
                "contribution": asset_impact * weight
            })
            total_impact += asset_impact * weight
            
        return {
            "portfolio_impact_pct": total_impact,
            "market_impact_pct": market_shock_pct,
            "details": results
        }
