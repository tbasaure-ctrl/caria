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
        
        # Load relevant historical events for context
        events = self._load_historical_events(crisis_id)

        return {
            "crisis_name": crisis["name"],
            "dates": bench_df.index.strftime('%Y-%m-%d').tolist(),
            "portfolio_values": portfolio_series.tolist(),
            "benchmark_values": bench_df['normalized'].tolist(),
            "metrics": {
                "max_drawdown": float(max_drawdown),
                "total_return": float((portfolio_series.iloc[-1] / 100) - 1),
                "benchmark_return": float((bench_df['normalized'].iloc[-1] / 100) - 1)
            },
            "historical_events": events
        }

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
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
        
        # Load relevant historical events for context
        events = self._load_historical_events(crisis_id)

        return {
            "crisis_name": crisis["name"],
            "dates": bench_df.index.strftime('%Y-%m-%d').tolist(),
            "portfolio_values": portfolio_series.tolist(),
            "benchmark_values": bench_df['normalized'].tolist(),
            "metrics": {
                "max_drawdown": float(max_drawdown),
                "total_return": float((portfolio_series.iloc[-1] / 100) - 1),
                "benchmark_return": float((bench_df['normalized'].iloc[-1] / 100) - 1)
            },
            "historical_events": events
        }

    def run_monte_carlo(self, symbol: str, years: int = 5, n_paths: int = 1000, growth_rate: float = None, stage: str = None) -> Dict[str, Any]:
        """Run a Monte Carlo price simulation for a given ticker.
        Returns price paths, percentiles (p10, p50, p90) and the growth rate used.
        Stage can be one of: pre-revenue, stalwart, turnaround, etc., and adjusts the drift.
        """
        try:
            # 1. Get current price
            current_price = self.obb_client.get_current_price(symbol)
            if not current_price:
                return {"error": f"Could not fetch current price for {symbol}"}

            # 2. Determine growth rate
            if growth_rate is None:
                # Use last year growth from financials if available
                fin = self.obb_client.get_financials(symbol)
                if fin and hasattr(fin, 'to_df'):
                    df = fin.to_df()
                    # Try to get revenue growth or EPS growth as proxy
                    growth = None
                    if 'revenueGrowth' in df.columns:
                        growth = df['revenueGrowth'].iloc[-1]
                    elif 'netIncomeGrowth' in df.columns:
                        growth = df['netIncomeGrowth'].iloc[-1]
                    growth_rate = float(growth) if growth is not None else 0.05
                else:
                    growth_rate = 0.05
            # 3. Adjust drift based on stage
            drift = growth_rate
            if stage == "pre-revenue":
                # Use multiples EV/Sales as proxy, assume higher volatility
                drift = max(drift, 0.10)
            elif stage == "turnaround":
                drift = max(drift, 0.12)
            elif stage == "stalwart":
                drift = min(drift, 0.07)

            # 4. Volatility estimate – use historical price std dev over 1 year
            hist = self.obb_client.get_price_history(symbol, start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
            if not hist or not hasattr(hist, 'to_df'):
                sigma = 0.2
            else:
                df_hist = hist.to_df()
                if 'close' in df_hist.columns:
                    returns = df_hist['close'].pct_change().dropna()
                    sigma = returns.std()
                else:
                    sigma = 0.2

            dt = 1/252  # daily steps
            total_steps = years * 252
            paths = []
            for _ in range(n_paths):
                price = current_price
                path = [price]
                for _ in range(total_steps):
                    # Geometric Brownian Motion
                    rand = np.random.normal()
                    price = price * np.exp((drift - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
                    path.append(price)
                paths.append(path)

            # 5. Compute percentiles at final step
            final_prices = [p[-1] for p in paths]
            p10 = np.percentile(final_prices, 10)
            p50 = np.percentile(final_prices, 50)
            p90 = np.percentile(final_prices, 90)

            return {
                "symbol": symbol,
                "current_price": current_price,
                "growth_rate": drift,
                "volatility": sigma,
                "price_paths": paths,
                "percentiles": {"p10": p10, "p50": p50, "p90": p90}
            }
        except Exception as e:
            LOGGER.error(f"Monte Carlo simulation error for {symbol}: {e}")
            return {"error": str(e)}

    def _load_historical_events(self, crisis_id: str) -> List[Dict[str, Any]]:
        """Load and filter historical events for a specific crisis."""
        import json
        from pathlib import Path
        
        # Map crisis_id to keywords/tags
        keywords = {
            "1929_depression": ["1929", "depression", "crash"],
            "1939_wwii": ["1939", "war", "hitler", "germany"],
            "1962_cuban_missile": ["cuban", "missile", "kennedy", "1962"],
            "1987_black_monday": ["1987", "black monday", "crash"],
            "2000_dot_com": ["dot com", "bubble", "2000", "tech", "nasdaq"],
            "2001_911": ["9/11", "terrorist", "2001", "attacks"],
            "2008_gfc": ["2008", "financial crisis", "lehman", "subprime", "housing"],
            "2011_euro_debt": ["euro", "debt", "greece", "2011"],
            "2020_covid": ["covid", "coronavirus", "2020", "pandemic", "lockdown"],
            "2022_inflation": ["inflation", "fed", "2022", "rates"],
        }
        
        target_keywords = keywords.get(crisis_id, [])
        if not target_keywords:
            return []

        # Try multiple paths for robustness
        paths = [
            Path(r"c:\key\wise_adviser_cursor_context\notebooks\data\raw\wisdom\2025-11-08\historical_events_wisdom.jsonl"),
            Path(__file__).resolve().parents[3] / "data" / "raw" / "wisdom" / "2025-11-08" / "historical_events_wisdom.jsonl"
        ]
        
        events = []
        for path in paths:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                text = data.get("text", "").lower()
                                tags = data.get("historical_events", [])
                                source = data.get("source", "")
                                
                                # Check match
                                match = False
                                for k in target_keywords:
                                    if k in text or k in source.lower() or any(k in str(t).lower() for t in tags):
                                        match = True
                                        break
                                
                                if match:
                                    events.append({
                                        "date": "N/A", # Date extraction would require parsing text or metadata
                                        "headline": source,
                                        "description": text[:300] + "..." if len(text) > 300 else text
                                    })
                            except:
                                pass
                    break # Found the file, stop searching
                except Exception as e:
                    LOGGER.warning(f"Error reading historical events: {e}")
        
        return events[:10] # Return top 10 relevant events
