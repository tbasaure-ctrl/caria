"""
Simulation Service for Crisis Simulator, Macro Multiverse, and Monte Carlo simulations.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .openbb_client import OpenBBClient

LOGGER = logging.getLogger("caria.services.simulation")

# Historical crisis data with pre-computed benchmark returns for reliability
# Event types: "acute" (days-based), "recession" (months-based), "crash" (weeks-based)
CRISIS_DATA = {
    "1929_depression": {
        "start": "1929-01-01", 
        "end": "1933-12-31", 
        "name": "Great Depression",
        "benchmark_return": -0.86,
        "description": "The worst stock market crash in U.S. history",
        "event_type": "recession",
        "event_date": "1929-10-24",
        "pre_crisis_months": 6
    },
    "1939_wwii": {
        "start": "1939-01-01", 
        "end": "1945-12-31", 
        "name": "WWII Start",
        "benchmark_return": 0.07,
        "description": "World War II era volatility",
        "event_type": "recession",
        "event_date": "1939-09-01",
        "pre_crisis_months": 6
    },
    "1962_cuban_missile": {
        "start": "1962-08-01", 
        "end": "1963-01-01", 
        "name": "Cuban Missile Crisis",
        "benchmark_return": -0.07,
        "description": "Nuclear standoff between US and USSR",
        "event_type": "acute",
        "event_date": "1962-10-16",
        "pre_crisis_months": 2
    },
    "1963_jfk": {
        "start": "1963-10-01", 
        "end": "1964-02-01", 
        "name": "Kennedy Assassination",
        "benchmark_return": 0.15,
        "description": "Markets recovered quickly after initial shock",
        "event_type": "acute",
        "event_date": "1963-11-22",
        "pre_crisis_months": 2
    },
    "1987_black_monday": {
        "start": "1987-01-01", 
        "end": "1988-12-31", 
        "name": "Black Monday",
        "benchmark_return": -0.22,
        "description": "Single-day crash of 22.6%",
        "event_type": "acute",
        "event_date": "1987-10-19",
        "pre_crisis_months": 9
    },
    "2000_dot_com": {
        "start": "1999-01-01", 
        "end": "2002-12-31", 
        "name": "Dot Com Bubble",
        "benchmark_return": -0.49,
        "description": "Tech bubble burst - NASDAQ fell 78%",
        "event_type": "recession",
        "event_date": "2000-03-10",
        "pre_crisis_months": 12
    },
    "2001_911": {
        "start": "2001-07-01", 
        "end": "2002-03-01", 
        "name": "9/11 Attacks",
        "benchmark_return": -0.12,
        "description": "Terrorist attacks on World Trade Center. Market closed 9/11-9/14.",
        "event_type": "acute",
        "event_date": "2001-09-11",
        "market_closure": ["2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14"],
        "pre_crisis_months": 2
    },
    "2008_gfc": {
        "start": "2007-01-01", 
        "end": "2009-12-31", 
        "name": "Global Financial Crisis",
        "benchmark_return": -0.53,
        "description": "Lehman Brothers collapse (Sept 2008), global credit crisis",
        "event_type": "recession",
        "event_date": "2008-09-15",
        "pre_crisis_months": 12
    },
    "2011_euro_debt": {
        "start": "2011-01-01", 
        "end": "2012-12-31", 
        "name": "European Debt Crisis",
        "benchmark_return": -0.19,
        "description": "Greek debt crisis, European contagion fears",
        "event_type": "recession",
        "event_date": "2011-05-01",
        "pre_crisis_months": 4
    },
    "2018_trade_war": {
        "start": "2018-01-01", 
        "end": "2019-03-01", 
        "name": "Trade War",
        "benchmark_return": -0.20,
        "description": "US-China trade tensions and Fed rate hikes",
        "event_type": "crash",
        "event_date": "2018-10-01",
        "pre_crisis_months": 9
    },
    "2020_covid": {
        "start": "2019-10-01", 
        "end": "2020-06-01", 
        "name": "COVID-19 Crash",
        "benchmark_return": -0.34,
        "description": "Fastest 30% decline in history due to pandemic. Trough: March 23",
        "event_type": "crash",
        "event_date": "2020-02-19",
        "pre_crisis_months": 4
    },
    "2022_inflation": {
        "start": "2021-06-01", 
        "end": "2022-12-31", 
        "name": "2022 Inflation Bear Market",
        "benchmark_return": -0.25,
        "description": "Fed rate hikes to combat 40-year high inflation",
        "event_type": "recession",
        "event_date": "2022-01-03",
        "pre_crisis_months": 6
    },
}

# Sector sensitivity coefficients for macro simulation
# Format: {sector: {"inflation": coef, "rates": coef, "gdp": coef}}
SECTOR_SENSITIVITIES = {
    "Technology": {"inflation": -1.5, "rates": -2.0, "gdp": 1.5},
    "Healthcare": {"inflation": -0.5, "rates": -0.5, "gdp": 0.5},
    "Financials": {"inflation": 0.5, "rates": 1.5, "gdp": 1.0},
    "Consumer Discretionary": {"inflation": -1.0, "rates": -1.0, "gdp": 1.5},
    "Consumer Staples": {"inflation": 0.3, "rates": -0.3, "gdp": 0.3},
    "Energy": {"inflation": 1.0, "rates": 0.5, "gdp": 1.0},
    "Utilities": {"inflation": -0.3, "rates": -1.5, "gdp": 0.2},
    "Real Estate": {"inflation": -0.5, "rates": -2.5, "gdp": 0.5},
    "Materials": {"inflation": 0.8, "rates": -0.5, "gdp": 1.2},
    "Industrials": {"inflation": -0.5, "rates": -0.5, "gdp": 1.3},
    "Communication Services": {"inflation": -1.0, "rates": -1.0, "gdp": 1.0},
    "default": {"inflation": -0.5, "rates": -1.0, "gdp": 1.0},
}


class SimulationService:
    def __init__(self):
        self.obb_client = OpenBBClient()

    def simulate_crisis(self, portfolio: List[Dict[str, Any]], crisis_id: str) -> Dict[str, Any]:
        """
        Simulate portfolio performance during a specific historical crisis.
        
        Args:
            portfolio: List of dicts with 'ticker' and 'weight' (or 'quantity').
            crisis_id: Key from CRISIS_DATA.
            
        Returns:
            Dict with 'dates', 'portfolio_values', 'benchmark_values', 'metrics'.
        """
        if crisis_id not in CRISIS_DATA:
            return {"error": f"Invalid crisis_id: {crisis_id}. Valid options: {list(CRISIS_DATA.keys())}"}
            
        crisis = CRISIS_DATA[crisis_id]
        
        # We want to show context BEFORE the crisis start
        start_date = crisis["start"]
        end_date = crisis["end"]
        
        # Calculate total weight and normalize
        total_weight = sum(p.get('weight', 0) for p in portfolio)
        if total_weight == 0:
            # If weight is missing, try quantity * current_price or just equal weight
            weight = 1.0 / len(portfolio) if portfolio else 1.0
            for p in portfolio:
                p['weight'] = weight
            total_weight = 1.0
        
        # Normalize weights to sum to 1
        for p in portfolio:
            p['weight'] = p['weight'] / total_weight
        
        # Try to fetch actual historical data first
        benchmark_data = self._fetch_benchmark_data(start_date, end_date)
        
        if benchmark_data is not None and not benchmark_data.empty:
            # Use actual data
            return self._simulate_with_actual_data(portfolio, crisis, benchmark_data)
        else:
            # Fallback to synthetic simulation using pre-computed crisis returns
            LOGGER.info(f"Using synthetic simulation for {crisis_id} (no benchmark data available)")
            return self._simulate_synthetic(portfolio, crisis, start_date)

    def _fetch_benchmark_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch benchmark data, trying multiple symbols."""
        benchmark_symbols = ["SPY", "^GSPC", "VOO"]
        
        for symbol in benchmark_symbols:
            try:
                data = self.obb_client.get_price_history(symbol, start_date=start_date)
                if data and hasattr(data, 'to_df'):
                    df = data.to_df()
                    if not df.empty and 'close' in df.columns:
                        # Filter to date range
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                            df.set_index('date', inplace=True)
                        df.sort_index(inplace=True)
                        if len(df) > 5:  # Need at least some data points
                            return df
            except Exception as e:
                LOGGER.warning(f"Failed to fetch {symbol}: {e}")
                continue
        return None

    def _simulate_with_actual_data(self, portfolio: List[Dict[str, Any]], crisis: Dict, bench_df: pd.DataFrame) -> Dict[str, Any]:
        """Run simulation using actual historical data."""
        initial_bench = bench_df['close'].iloc[0]
        # Benchmark returns relative to start (%)
        bench_df['normalized'] = ((bench_df['close'] / initial_bench) - 1) * 100
        
        portfolio_series = pd.Series(0.0, index=bench_df.index)
        
        for asset in portfolio:
            ticker = asset['ticker']
            weight = asset.get('weight', 0)
            
            # Try to fetch asset history
            try:
                hist = self.obb_client.get_price_history(
                    ticker, 
                    start_date=bench_df.index.min().strftime('%Y-%m-%d')
                )
                asset_df = pd.DataFrame()
                if hist and hasattr(hist, 'to_df'):
                    asset_df = hist.to_df()
                
                if not asset_df.empty and 'close' in asset_df.columns:
                    if 'date' in asset_df.columns:
                        asset_df['date'] = pd.to_datetime(asset_df['date'])
                        asset_df.set_index('date', inplace=True)
                    
                    asset_df = asset_df.reindex(bench_df.index, method='ffill').bfill()
                    
                    if not asset_df['close'].isnull().all():
                        initial_price = asset_df['close'].iloc[0]
                        if initial_price and initial_price > 0:
                            normalized = ((asset_df['close'] / initial_price) - 1) * 100
                            portfolio_series += normalized * weight
                            continue
            except Exception as e:
                LOGGER.warning(f"Error fetching {ticker}: {e}")
            
            # Fallback: asset tracks benchmark
            portfolio_series += bench_df['normalized'] * weight
        
        # Calculate metrics
        # Drawdown is calculated from the normalized series (which is % change from start)
        # We need the relative peak to trough
        current_values = 1 + (portfolio_series / 100)
        rolling_max = current_values.cummax()
        drawdown_series = (current_values - rolling_max) / rolling_max
        max_drawdown = float(drawdown_series.min())

        # Recovery time (simplified: find trough and then find when it returns to previous peak)
        trough_idx = drawdown_series.idxmin()
        peak_before_trough = current_values.loc[:trough_idx].max()
        recovery_period = current_values.loc[trough_idx:][current_values >= peak_before_trough]
        
        recovery_time_months = "--"
        if not recovery_period.empty:
            days = (recovery_period.index[0] - trough_idx).days
            recovery_time_months = round(days / 30)

        # Volatility
        daily_returns = current_values.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        vol_label = "LOW"
        if volatility > 0.35: vol_label = "HIGH"
        elif volatility > 0.20: vol_label = "MEDIUM"

        # Convert dates to string list
        dates_list = [d.strftime('%b %d, %Y') if hasattr(d, 'strftime') else str(d) for d in bench_df.index]

        return {
            "crisis_name": crisis["name"],
            "description": crisis.get("description", ""),
            "dates": dates_list,
            "portfolio_values": portfolio_series.tolist(),
            "benchmark_values": bench_df['normalized'].tolist(),
            "metrics": {
                "max_drawdown": max_drawdown,
                "total_return": float((current_values.iloc[-1]) - 1),
                "benchmark_return": float((bench_df['close'].iloc[-1] / initial_bench) - 1),
                "recovery_time": recovery_time_months,
                "volatility": vol_label
            }
        }

    def _simulate_synthetic(self, portfolio: List[Dict[str, Any]], crisis: Dict, start_date: str = None) -> Dict[str, Any]:
        """Run synthetic simulation when actual data isn't available."""
        benchmark_return = crisis.get("benchmark_return", -0.30)
        event_type = crisis.get("event_type", "recession")
        
        # Duration adjustments for better visualization
        if event_type == "recession":
            days = 60  # months scaled to display points
        elif event_type == "crash":
            days = 45
        else: # acute
            days = 30
            
        sim_start = start_date if start_date else crisis["start"]
        dates = pd.date_range(start=sim_start, periods=days, freq='B')
        
        # Generate context-aware synthetic trajectory
        # 1. Pre-event phase (steady or rising)
        # 2. Crash phase (steep decline)
        # 3. Post-event phase (partial recovery)
        
        t = np.linspace(0, 1, days)
        event_pos = 0.2  # event happens at 20% of the way
        
        benchmark_values = []
        for i, val in enumerate(t):
            if val < event_pos:
                # Pre-event: slight positive drift
                ret = 1 + (val * 0.1)
            else:
                # Crash and Recovery
                # Normalize val to [0, 1] starting from event
                t_crash = (val - event_pos) / (1 - event_pos)
                # Crash part (negative sigmoid)
                crash = benchmark_return * (1 / (1 + np.exp(-15 * (t_crash - 0.1))))
                # Recovery part (gentle rising)
                recovery = abs(benchmark_return) * 0.3 * (t_crash ** 0.5)
                ret = 1.05 + crash + recovery
            
            benchmark_values.append((ret - 1) * 100)
            
        # Portfolio follows benchmark with some variation
        portfolio_values = [v * 0.9 + np.random.normal(0, 1) for v in benchmark_values]
        
        # Metrics
        max_drawdown = float(benchmark_return)
        recovery_time = "--"
        if event_type == "acute":
            recovery_time = "2"
        elif event_type == "recession":
            recovery_time = "18"
        else:
            recovery_time = "6"

        return {
            "crisis_name": crisis["name"],
            "description": crisis.get("description", ""),
            "dates": [d.strftime('%b %d, %Y') for d in dates],
            "portfolio_values": portfolio_values,
            "benchmark_values": benchmark_values,
            "metrics": {
                "max_drawdown": max_drawdown,
                "total_return": float(portfolio_values[-1] / 100),
                "benchmark_return": float(benchmark_return),
                "recovery_time": recovery_time,
                "volatility": "HIGH"
            },
            "note": "Synthetic simulation (actual data unavailable)"
        }

    def simulate_macro(self, portfolio: List[Dict[str, Any]], params: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate portfolio performance under macroeconomic scenario shocks.
        """
        inflation_shock = params.get('inflation', 0) / 100  # Convert to decimal
        rates_shock = params.get('rates', 0) / 100
        gdp_shock = params.get('gdp', 0) / 100
        
        # Normalize weights
        total_weight = sum(p.get('weight', 0) for p in portfolio)
        if total_weight == 0:
            weight = 1.0 / len(portfolio) if portfolio else 1.0
            for p in portfolio:
                p['weight'] = weight
            total_weight = 1.0
        
        # Calculate market impact (general S&P 500 sensitivity)
        market_sensitivity = SECTOR_SENSITIVITIES["default"]
        market_impact = (
            inflation_shock * market_sensitivity["inflation"] * 5 +
            rates_shock * market_sensitivity["rates"] * 10 +
            gdp_shock * market_sensitivity["gdp"] * 3
        )
        
        # Calculate per-asset impact
        details = []
        portfolio_impact = 0.0
        
        for asset in portfolio:
            ticker = asset['ticker']
            weight = asset.get('weight', 0) / total_weight
            
            # Try to get sector for the ticker
            sector = self._get_sector(ticker)
            sensitivity = SECTOR_SENSITIVITIES.get(sector, SECTOR_SENSITIVITIES["default"])
            
            # Calculate asset impact based on sector sensitivities
            asset_impact = (
                inflation_shock * sensitivity["inflation"] * 5 +
                rates_shock * sensitivity["rates"] * 10 +
                gdp_shock * sensitivity["gdp"] * 3
            )
            
            # Add some ticker-specific variation
            np.random.seed(hash(ticker) % (2**32))
            asset_impact += np.random.normal(0, abs(asset_impact) * 0.1)
            
            contribution = asset_impact * weight
            portfolio_impact += contribution
            
            details.append({
                "ticker": ticker,
                "sector": sector,
                "impact_pct": round(asset_impact * 100, 2),
                "contribution": round(contribution * 100, 2),
                "weight": round(weight * 100, 2)
            })
        
        return {
            "portfolio_impact_pct": round(portfolio_impact * 100, 2),
            "market_impact_pct": round(market_impact * 100, 2),
            "scenario": {
                "inflation_shock": params.get('inflation', 0),
                "rates_shock": params.get('rates', 0),
                "gdp_shock": params.get('gdp', 0)
            },
            "details": details,
            "interpretation": self._interpret_macro_result(portfolio_impact, market_impact, params)
        }

    def _get_sector(self, ticker: str) -> str:
        """Get sector for a ticker (simplified mapping)."""
        TICKER_SECTORS = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", 
            "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
            "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
            "ADBE": "Technology", "ORCL": "Technology", "CSCO": "Technology",
            "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
            "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
            "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
            "GS": "Financials", "MS": "Financials", "C": "Financials",
            "BRK.B": "Financials", "V": "Financials", "MA": "Financials",
            "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
            "HD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
            "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
            "KO": "Consumer Staples", "PEP": "Consumer Staples", 
            "PG": "Consumer Staples", "WMT": "Consumer Staples",
            "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
            "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
            "AMT": "Real Estate", "PLD": "Real Estate", "SPG": "Real Estate",
            "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
            "CAT": "Industrials", "DE": "Industrials", "UPS": "Industrials",
            "HON": "Industrials", "BA": "Industrials", "GE": "Industrials",
            "DIS": "Communication Services", "NFLX": "Communication Services",
            "CMCSA": "Communication Services", "T": "Communication Services",
            "VZ": "Communication Services",
        }
        return TICKER_SECTORS.get(ticker.upper(), "default")

    def _interpret_macro_result(self, portfolio_impact: float, market_impact: float, params: Dict) -> str:
        """Generate human-readable interpretation of macro simulation."""
        interpretations = []
        if portfolio_impact > market_impact:
            diff = (portfolio_impact - market_impact) * 100
            interpretations.append(f"Your portfolio would outperform the market by {diff:.1f}% in this scenario.")
        elif portfolio_impact < market_impact:
            diff = (market_impact - portfolio_impact) * 100
            interpretations.append(f"Your portfolio would underperform the market by {diff:.1f}% in this scenario.")
        else:
            interpretations.append("Your portfolio would perform in line with the market.")
        return " ".join(interpretations)

    def run_monte_carlo(self, symbol: str, years: int = 5, n_paths: int = 1000, 
                        growth_rate: float = None, stage: str = None) -> Dict[str, Any]:
        """
        Run a Monte Carlo price simulation for a given ticker.
        """
        try:
            current_price = self.obb_client.get_current_price(symbol)
            if not current_price:
                return {"error": f"Could not fetch current price for {symbol}"}
            if growth_rate is None:
                growth_rate = 0.05
            drift = growth_rate
            sigma = 0.2
            dt = 1/252
            total_steps = years * 252
            paths = []
            for _ in range(min(n_paths, 100)): # Limit paths for performance
                price = current_price
                path = [price]
                for _ in range(total_steps):
                    rand = np.random.normal()
                    price = price * np.exp((drift - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
                    path.append(price)
                paths.append(path)
            final_prices = [p[-1] for p in paths]
            return {
                "symbol": symbol,
                "current_price": current_price,
                "growth_rate": drift,
                "volatility": sigma,
                "price_paths": paths,
                "percentiles": {
                    "p10": float(np.percentile(final_prices, 10)),
                    "p50": float(np.percentile(final_prices, 50)),
                    "p90": float(np.percentile(final_prices, 90))
                }
            }
        except Exception as e:
            return {"error": str(e)}
