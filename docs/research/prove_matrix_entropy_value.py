"""
Matrix Entropy Historical Proof: 35-Year Backtest (1990-2024)
=============================================================

Objective:
    Test if Matrix Entropy outperforms Volatility over a LONGER history
    that includes regimes where "buy the dip" FAILED (Dot-Com, GFC).

Key Difference:
    Uses individual large-cap stocks with 30+ year history instead of ETFs.
    This captures:
    - 1990-1991 Recession
    - 1997 Asian Crisis / 1998 LTCM
    - 2000-2002 Dot-Com Bust (NO V-recovery for 3 years)
    - 2007-2009 GFC
    - 2011 Euro Crisis
    - 2020 COVID

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.metrics import roc_auc_score
import os
import warnings

warnings.filterwarnings("ignore")

# Long-History Universe (Stocks with 30+ years of data)
# These are blue-chip stocks that existed before ETFs
LONG_HISTORY_UNIVERSE = [
    # Tech (Pre-Dot-Com)
    "AAPL", "MSFT", "INTC", "IBM", "ORCL",
    # Consumer
    "PG", "KO", "PEP", "JNJ", "WMT", "MCD",
    # Financials
    "JPM", "BAC", "WFC", "GS",
    # Energy
    "XOM", "CVX",
    # Industrials
    "GE", "MMM", "CAT", "BA",
    # Healthcare
    "MRK", "PFE", "ABT",
    # Materials
    "DOW",
]

# Using ^GSPC (S&P 500 Index) for market proxy - goes back to 1920s
# Using ^TNX (10Y Treasury Yield) for defensive proxy
START_DATE = "1990-01-01"
WINDOW = 63
ROLL_RANK = 252 * 2

def load_long_history_data(tickers, start_date):
    print(f"Loading {len(tickers)} assets from {start_date}...")
    try:
        df = yf.download(tickers, start=start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            px = df['Adj Close' if 'Adj Close' in df else 'Close']
        else:
            px = df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()
    
    # Also download S&P 500 Index and Treasury
    extras = yf.download(["^GSPC", "^TNX"], start=start_date, progress=False)
    if isinstance(extras.columns, pd.MultiIndex):
        extras = extras['Adj Close' if 'Adj Close' in extras else 'Close']
    
    return px.dropna(how='all'), extras.dropna(how='all')

def calculate_features(px, window):
    returns = px.pct_change()
    
    entropy_series = {}
    vol_series = {}
    
    # Use equal-weighted portfolio return as market proxy
    market_ret = returns.mean(axis=1)

    print("Calculating Rolling Features over 35 years...")
    for i in range(window, len(returns)):
        idx = returns.index[i]
        
        # Entropy
        window_ret = returns.iloc[i-window : i]
        # Only use columns with enough data in this window
        valid_cols = window_ret.dropna(axis=1, thresh=int(window*0.8)).columns
        if len(valid_cols) < 10:
            continue
            
        corr_matrix = window_ret[valid_cols].corr()
        corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        if not corr_matrix.empty and len(corr_matrix) >= 10:
            try:
                eigvals = np.linalg.eigvalsh(corr_matrix)
                eigvals = eigvals[eigvals > 1e-10]
                probs = eigvals / np.sum(eigvals)
                S = -np.sum(probs * np.log(probs))
                N = len(probs)
                S_norm = S / np.log(N)
            except:
                S_norm = np.nan
        else:
            S_norm = np.nan
        entropy_series[idx] = S_norm
        
        # Volatility (Energy) - Equal-weighted portfolio
        vol_series[idx] = market_ret.iloc[i-window : i].std() * np.sqrt(252)
            
    df = pd.DataFrame({
        'Entropy': pd.Series(entropy_series),
        'Vol_Equity': pd.Series(vol_series),
    })
    return df.dropna()

def backtest_strategies(market_data, features):
    # Use S&P 500 Index for returns
    # Use Treasury Yield as proxy for "defensive" (simplified)
    
    common = market_data.index.intersection(features.index)
    market = market_data.loc[common]
    feats = features.loc[common]
    
    # S&P 500 returns
    sp500_ret = market['^GSPC'].pct_change()
    
    # Treasury return proxy: (1/Yield) change (simplified - when yields go down, "price" goes up)
    # This is a crude proxy but captures the flight-to-safety
    if '^TNX' in market.columns:
        # Convert yield to approximate price return
        # When yield drops (flight to safety), this is "positive" for treasuries
        treasury_price_proxy = 1 / (market['^TNX'].replace(0, np.nan) + 1) # Avoid div by zero
        treasury_ret = treasury_price_proxy.pct_change().clip(-0.1, 0.1)
    else:
        treasury_ret = pd.Series(0, index=common) # Cash
    
    # Ranks (Expanding Window)
    ent_rank = feats['Entropy'].rolling(ROLL_RANK, min_periods=252).rank(pct=True)
    vol_rank = feats['Vol_Equity'].rolling(ROLL_RANK, min_periods=252).rank(pct=True)
    
    signals = pd.DataFrame(index=common)
    
    # Minsky (Pure): High Fragility (Low Entropy <20%) + Low Vol (<50%)
    is_fragile = ent_rank < 0.20
    is_complacent = vol_rank < 0.50
    signals['Minsky'] = is_fragile & is_complacent
    
    # Reactive: High Vol (>85%)
    signals['Reactive'] = vol_rank > 0.85
    
    # Returns
    res = pd.DataFrame({'Benchmark (S&P 500)': sp500_ret})
    
    for strat in ['Minsky', 'Reactive']:
        pos = signals[strat].shift(1).fillna(False)
        res[strat] = np.where(pos, treasury_ret, sp500_ret)
        
    return res.dropna()

def calculate_metrics(returns):
    metrics = {}
    for col in returns.columns:
        r = returns[col]
        cagr = (1 + r).prod() ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        sharpe = (cagr - 0.03) / vol  # Using 3% risk-free for historical
        
        cum = (1 + r).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        
        metrics[col] = {
            'CAGR': cagr,
            'Vol': vol,
            'Sharpe': sharpe,
            'MaxDD': max_dd
        }
    return pd.DataFrame(metrics).T

def analyze_by_decade(returns):
    """Break down performance by decade."""
    decades = {
        '1990s': ('1990', '1999'),
        '2000s': ('2000', '2009'),
        '2010s': ('2010', '2019'),
        '2020s': ('2020', '2024'),
    }
    
    results = []
    for name, (start, end) in decades.items():
        try:
            subset = returns.loc[start:end]
            if len(subset) > 100:
                metrics = calculate_metrics(subset)
                metrics['Decade'] = name
                results.append(metrics)
        except:
            pass
    
    if results:
        return pd.concat(results)
    return pd.DataFrame()

def plot_results(returns, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    cum_ret = (1 + returns).cumprod()
    
    plt.figure(figsize=(14, 7))
    plt.plot(cum_ret['Benchmark (S&P 500)'], 'gray', alpha=0.4, label='S&P 500')
    plt.plot(cum_ret['Minsky'], '#2ca02c', linewidth=2, label='Minsky (Structure)')
    plt.plot(cum_ret['Reactive'], 'blue', alpha=0.7, label='Reactive (Vol)')
    
    # Mark major crises
    crises = {
        'Dot-Com Peak': '2000-03-24',
        'GFC Peak': '2007-10-09',
        'COVID Crash': '2020-02-19',
    }
    for name, date in crises.items():
        try:
            plt.axvline(pd.Timestamp(date), color='red', linestyle='--', alpha=0.3)
        except:
            pass
    
    plt.title("35-Year Backtest: Matrix Entropy vs Volatility (1990-2024)", fontsize=14)
    plt.yscale('log')
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Proof_35Year_Backtest.png"))
    plt.close()
    
    # Drawdowns
    dd = (cum_ret - cum_ret.cummax()) / cum_ret.cummax()
    plt.figure(figsize=(14, 5))
    plt.plot(dd['Benchmark (S&P 500)'], 'gray', alpha=0.3, label='S&P 500 DD')
    plt.plot(dd['Minsky'], '#2ca02c', alpha=0.8, label='Minsky DD')
    plt.plot(dd['Reactive'], 'blue', alpha=0.5, label='Reactive DD')
    plt.title("Drawdowns (1990-2024)")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Proof_35Year_Drawdowns.png"))
    plt.close()

def main():
    print("=== 35-YEAR HISTORICAL BACKTEST ===")
    
    px, market = load_long_history_data(LONG_HISTORY_UNIVERSE, START_DATE)
    
    if px.empty or market.empty: 
        print("Data load failed.")
        return
    
    print(f"Loaded {len(px.columns)} stocks, {len(px)} days")
    print(f"Date range: {px.index.min().date()} to {px.index.max().date()}")
    
    feats = calculate_features(px, WINDOW)
    print(f"Features calculated for {len(feats)} days")
    
    results = backtest_strategies(market, feats)
    
    # Overall Metrics
    metrics = calculate_metrics(results)
    print("\n=== OVERALL PERFORMANCE (1990-2024) ===")
    print(metrics)
    
    # By Decade
    decade_metrics = analyze_by_decade(results)
    if not decade_metrics.empty:
        print("\n=== PERFORMANCE BY DECADE ===")
        print(decade_metrics)
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    metrics.to_csv(os.path.join(output_dir, "backtest_35year_metrics.csv"))
    decade_metrics.to_csv(os.path.join(output_dir, "backtest_35year_by_decade.csv"))
    
    plot_results(results, output_dir)
    print(f"\n=== PROOF COMPLETE. Artifacts in {output_dir} ===")

if __name__ == "__main__":
    main()
