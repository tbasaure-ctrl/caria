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
CRISIS_DATA = {
    "1929_depression": {
        "start": "1929-09-01", 
        "end": "1932-06-01", 
        "name": "Great Depression (1929)",
        "benchmark_return": -0.86,  # S&P 500 approximate
        "description": "The worst stock market crash in U.S. history"
    },
    "1939_wwii": {
        "start": "1939-09-01", 
        "end": "1945-09-01", 
        "name": "WWII Start (1939)",
        "benchmark_return": 0.07,  # Markets recovered during war
        "description": "World War II era volatility"
    },
    "1962_cuban_missile": {
        "start": "1962-10-16", 
        "end": "1962-11-20", 
        "name": "Cuban Missile Crisis (1962)",
        "benchmark_return": -0.07,
        "description": "Nuclear standoff between US and USSR"
    },
    "1963_jfk": {
        "start": "1963-11-22", 
        "end": "1964-06-01", 
        "name": "Kennedy Assassination (1963)",
        "benchmark_return": 0.15,
        "description": "Markets recovered quickly after initial shock"
    },
    "1987_black_monday": {
        "start": "1987-10-01", 
        "end": "1988-03-01", 
        "name": "Black Monday (1987)",
        "benchmark_return": -0.22,
        "description": "Single-day crash of 22.6%"
    },
    "2000_dot_com": {
        "start": "2000-03-10", 
        "end": "2002-10-09", 
        "name": "Dot Com Bubble (2000)",
        "benchmark_return": -0.49,
        "description": "Tech bubble burst - NASDAQ fell 78%"
    },
    "2001_911": {
        "start": "2001-09-11", 
        "end": "2002-01-01", 
        "name": "9/11 Attacks (2001)",
        "benchmark_return": -0.12,
        "description": "Terrorist attacks on World Trade Center"
    },
    "2008_gfc": {
        "start": "2008-09-01", 
        "end": "2009-03-09", 
        "name": "Global Financial Crisis (2008)",
        "benchmark_return": -0.53,
        "description": "Lehman Brothers collapse, global credit crisis"
    },
    "2011_euro_debt": {
        "start": "2011-04-01", 
        "end": "2012-07-01", 
        "name": "European Debt Crisis (2011)",
        "benchmark_return": -0.19,
        "description": "Greek debt crisis, European contagion fears"
    },
    "2018_trade_war": {
        "start": "2018-09-20", 
        "end": "2018-12-24", 
        "name": "2018 Trade War / Fed Tightening",
        "benchmark_return": -0.20,
        "description": "US-China trade tensions and Fed rate hikes"
    },
    "2020_covid": {
        "start": "2020-02-19", 
        "end": "2020-03-23", 
        "name": "COVID-19 Crash (2020)",
        "benchmark_return": -0.34,
        "description": "Fastest 30% decline in history due to pandemic"
    },
    "2022_inflation": {
        "start": "2022-01-03", 
        "end": "2022-10-12", 
        "name": "2022 Inflation Bear Market",
        "benchmark_return": -0.25,
        "description": "Fed rate hikes to combat 40-year high inflation"
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
        start_date = crisis["start"]
        end_date = crisis["end"]
        
        # Calculate total weight and normalize
        total_weight = sum(p.get('weight', 0) for p in portfolio)
        if total_weight == 0:
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
            return self._simulate_synthetic(portfolio, crisis)

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
        bench_df['normalized'] = (bench_df['close'] / initial_bench) * 100
        
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
                            normalized = (asset_df['close'] / initial_price) * 100
                            portfolio_series += normalized * weight
                            continue
            except Exception as e:
                LOGGER.warning(f"Error fetching {ticker}: {e}")
            
            # Fallback: asset tracks benchmark
            portfolio_series += bench_df['normalized'] * weight
        
        # Calculate metrics
        rolling_max = portfolio_series.cummax()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        # Convert dates to string list
        dates_list = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in bench_df.index]

        return {
            "crisis_name": crisis["name"],
            "description": crisis.get("description", ""),
            "dates": dates_list,
            "portfolio_values": portfolio_series.tolist(),
            "benchmark_values": bench_df['normalized'].tolist(),
            "metrics": {
                "max_drawdown": max_drawdown,
                "total_return": float((portfolio_series.iloc[-1] / 100) - 1),
                "benchmark_return": float((bench_df['normalized'].iloc[-1] / 100) - 1)
            }
        }

    def _simulate_synthetic(self, portfolio: List[Dict[str, Any]], crisis: Dict) -> Dict[str, Any]:
        """Run synthetic simulation when actual data isn't available."""
        benchmark_return = crisis.get("benchmark_return", -0.30)
        
        # Generate synthetic timeline (30 trading days)
        days = 30
        dates = pd.date_range(start=crisis["start"], periods=days, freq='B')
        
        # Generate smooth benchmark decline using sigmoid
        t = np.linspace(0, 6, days)
        sigmoid = 1 / (1 + np.exp(-t + 3))
        benchmark_values = 100 * (1 + benchmark_return * sigmoid)
        
        # Portfolio follows benchmark with some noise
        portfolio_return = benchmark_return * 0.95  # Slightly better than benchmark
        portfolio_values = 100 * (1 + portfolio_return * sigmoid) + np.random.normal(0, 1, days)
        portfolio_values = np.maximum(portfolio_values, 10)  # Floor at 10
        
        max_drawdown = float(min(benchmark_return, portfolio_return))
        
        return {
            "crisis_name": crisis["name"],
            "description": crisis.get("description", ""),
            "dates": [d.strftime('%Y-%m-%d') for d in dates],
            "portfolio_values": portfolio_values.tolist(),
            "benchmark_values": benchmark_values.tolist(),
            "metrics": {
                "max_drawdown": max_drawdown,
                "total_return": float(portfolio_return),
                "benchmark_return": float(benchmark_return)
            },
            "note": "Synthetic simulation based on historical crisis returns (actual data unavailable)"
        }

    def simulate_macro(self, portfolio: List[Dict[str, Any]], params: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate portfolio performance under macroeconomic scenario shocks.
        
        Args:
            portfolio: List of dicts with 'ticker' and 'weight'.
            params: Dict with 'inflation', 'rates', 'gdp' shock values (in percentage points).
            
        Returns:
            Dict with portfolio impact, market impact, and per-asset breakdown.
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
        # Common sector mappings
        TICKER_SECTORS = {
            # Technology
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", 
            "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
            "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
            "ADBE": "Technology", "ORCL": "Technology", "CSCO": "Technology",
            # Healthcare
            "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
            "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
            # Financials
            "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
            "GS": "Financials", "MS": "Financials", "C": "Financials",
            "BRK.B": "Financials", "V": "Financials", "MA": "Financials",
            # Consumer
            "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
            "HD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
            "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
            "KO": "Consumer Staples", "PEP": "Consumer Staples", 
            "PG": "Consumer Staples", "WMT": "Consumer Staples",
            # Energy
            "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
            # Utilities
            "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
            # Real Estate
            "AMT": "Real Estate", "PLD": "Real Estate", "SPG": "Real Estate",
            # Materials
            "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
            # Industrials
            "CAT": "Industrials", "DE": "Industrials", "UPS": "Industrials",
            "HON": "Industrials", "BA": "Industrials", "GE": "Industrials",
            # Communication
            "DIS": "Communication Services", "NFLX": "Communication Services",
            "CMCSA": "Communication Services", "T": "Communication Services",
            "VZ": "Communication Services",
        }
        return TICKER_SECTORS.get(ticker.upper(), "default")

    def _interpret_macro_result(self, portfolio_impact: float, market_impact: float, params: Dict) -> str:
        """Generate human-readable interpretation of macro simulation."""
        interpretations = []
        
        # Portfolio vs Market
        if portfolio_impact > market_impact:
            diff = (portfolio_impact - market_impact) * 100
            interpretations.append(f"Your portfolio would outperform the market by {diff:.1f}% in this scenario.")
        elif portfolio_impact < market_impact:
            diff = (market_impact - portfolio_impact) * 100
            interpretations.append(f"Your portfolio would underperform the market by {diff:.1f}% in this scenario.")
        else:
            interpretations.append("Your portfolio would perform in line with the market.")
        
        # Scenario description
        if params.get('inflation', 0) > 2:
            interpretations.append("High inflation typically hurts growth stocks and benefits commodities/energy.")
        elif params.get('inflation', 0) < -2:
            interpretations.append("Deflation tends to hurt cyclicals but benefits quality growth stocks.")
        
        if params.get('rates', 0) > 1:
            interpretations.append("Rising rates pressure valuations, especially for high-multiple stocks.")
        elif params.get('rates', 0) < -1:
            interpretations.append("Falling rates generally support equity valuations.")
        
        if params.get('gdp', 0) < -2:
            interpretations.append("Recession fears would drive flight to quality and defensive sectors.")
        elif params.get('gdp', 0) > 2:
            interpretations.append("Strong growth expectations favor cyclical and growth stocks.")
        
        return " ".join(interpretations)

    def run_monte_carlo(self, symbol: str, years: int = 5, n_paths: int = 1000, 
                        growth_rate: float = None, stage: str = None) -> Dict[str, Any]:
        """
        Run a Monte Carlo price simulation for a given ticker.
        
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
                growth_rate = 0.05  # Default 5%
                try:
                    fin = self.obb_client.get_financials(symbol)
                    if fin and hasattr(fin, 'to_df'):
                        df = fin.to_df()
                        if 'revenueGrowth' in df.columns:
                            growth_rate = float(df['revenueGrowth'].iloc[-1] or 0.05)
                        elif 'netIncomeGrowth' in df.columns:
                            growth_rate = float(df['netIncomeGrowth'].iloc[-1] or 0.05)
                except Exception:
                    pass

            # 3. Adjust drift based on stage
            drift = growth_rate
            if stage == "pre-revenue":
                drift = max(drift, 0.10)
            elif stage == "turnaround":
                drift = max(drift, 0.12)
            elif stage == "stalwart":
                drift = min(drift, 0.07)

            # 4. Volatility estimate
            sigma = 0.2  # Default
            try:
                hist = self.obb_client.get_price_history(
                    symbol, 
                    start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                )
                if hist and hasattr(hist, 'to_df'):
                    df_hist = hist.to_df()
                    if 'close' in df_hist.columns:
                        returns = df_hist['close'].pct_change().dropna()
                        if len(returns) > 10:
                            sigma = max(returns.std() * np.sqrt(252), 0.1)  # Annualized, min 10%
            except Exception:
                pass

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
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "growth_rate": drift,
                "volatility": sigma,
                "price_paths": paths[:100],  # Limit to 100 paths for response size
                "percentiles": {
                    "p10": float(np.percentile(final_prices, 10)),
                    "p50": float(np.percentile(final_prices, 50)),
                    "p90": float(np.percentile(final_prices, 90))
                }
            }
        except Exception as e:
            LOGGER.error(f"Monte Carlo simulation error for {symbol}: {e}")
            return {"error": str(e)}
