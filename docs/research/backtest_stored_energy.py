"""
Stored Energy Strategy: The Final Backtest
==========================================

The "Compressed Spring" Strategy:
    - DANGER (0x): Stored Energy > 80th Percentile (Spring Compressed) + Low Volatility
    - NEUTRAL (1x): Normal state
    - SAFE (1.5x): Stored Energy < 30th Percentile (Spring Relaxed)

Key Insight:
    Time under fragility matters more than instantaneous fragility.

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
import warnings

warnings.filterwarnings("ignore")

LONG_HISTORY_UNIVERSE = [
    "AAPL", "MSFT", "INTC", "IBM", "ORCL",
    "PG", "KO", "PEP", "JNJ", "WMT", "MCD",
    "JPM", "BAC", "WFC", "GS",
    "XOM", "CVX",
    "GE", "MMM", "CAT", "BA",
    "MRK", "PFE", "ABT",
    "DOW",
]

START_DATE = "1980-01-01"
WINDOW = 63
ENERGY_WINDOW = 60
ROLL_RANK = 252 * 2
LEVERAGE = 1.5
BORROW_COST = 0.02

def load_data(tickers, start_date):
    print(f"Loading data...")
    try:
        df = yf.download(tickers, start=start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            px = df['Adj Close' if 'Adj Close' in df else 'Close']
        else:
            px = df
    except:
        return pd.DataFrame(), pd.DataFrame()
    
    extras = yf.download(["^GSPC"], start=start_date, progress=False)
    if isinstance(extras.columns, pd.MultiIndex):
        extras = extras['Adj Close' if 'Adj Close' in extras else 'Close']
    
    return px.dropna(how='all'), extras.dropna(how='all')

def calculate_features(px, window):
    returns = px.pct_change()
    entropy_series = {}
    vol_series = {}
    market_ret = returns.mean(axis=1)

    print("Calculating Entropy & Stored Energy...")
    for i in range(window, len(returns)):
        idx = returns.index[i]
        
        window_ret = returns.iloc[i-window : i]
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
        vol_series[idx] = market_ret.iloc[i-window : i].std() * np.sqrt(252)
            
    df = pd.DataFrame({
        'Entropy': pd.Series(entropy_series),
        'Vol': pd.Series(vol_series)
    })
    
    df['Fragility'] = 1 - df['Entropy']
    df['Stored_Energy'] = df['Fragility'].rolling(ENERGY_WINDOW).sum()
    
    return df.dropna()

def backtest_stored_energy_strategy(market, features):
    """Backtest: Dynamic leverage based on Stored Energy."""
    
    common = market.index.intersection(features.index)
    sp500_ret = market['^GSPC'].loc[common].pct_change()
    feats = features.loc[common]
    
    # Ranks for Stored Energy and Volatility
    se_rank = feats['Stored_Energy'].rolling(ROLL_RANK, min_periods=252).rank(pct=True)
    vol_rank = feats['Vol'].rolling(ROLL_RANK, min_periods=252).rank(pct=True)
    
    # DANGER: High Stored Energy (>80%) + Low Volatility (<50%)
    # "The spring is loaded and the room is quiet"
    danger = (se_rank > 0.80) & (vol_rank < 0.50)
    
    # SAFE: Low Stored Energy (<30%)
    # "The spring is relaxed"
    safe = se_rank < 0.30
    
    # Leverage
    leverage = pd.Series(1.0, index=common)
    leverage[danger] = 0.0
    leverage[safe] = LEVERAGE
    leverage = leverage.shift(1).fillna(1.0)
    
    # Borrow cost
    daily_cost = (BORROW_COST / 252) * (leverage - 1).clip(lower=0)
    
    # Strategy return
    strategy_ret = sp500_ret * leverage - daily_cost
    
    # Comparison strategies
    results = pd.DataFrame({
        'Benchmark (1x)': sp500_ret,
        'Stored Energy (0x-1.5x)': strategy_ret,
        'Always 1.5x': sp500_ret * 1.5 - (BORROW_COST / 252) * 0.5
    }).dropna()
    
    # Metrics
    metrics = {}
    for col in results.columns:
        r = results[col]
        cagr = (1 + r).prod() ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        sharpe = (cagr - 0.03) / vol
        cum = (1 + r).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        metrics[col] = {'CAGR': cagr, 'Vol': vol, 'Sharpe': sharpe, 'MaxDD': max_dd}
    
    metrics_df = pd.DataFrame(metrics).T
    
    # Time allocation
    time_danger = danger.mean()
    time_safe = safe.mean()
    time_neutral = 1 - time_danger - time_safe
    
    print(f"\n  Regime Allocation:")
    print(f"    DANGER (0x):   {time_danger:.1%}")
    print(f"    NEUTRAL (1x):  {time_neutral:.1%}")
    print(f"    SAFE (1.5x):   {time_safe:.1%}")
    
    return results, metrics_df, leverage, danger, safe

def plot_strategy(results, leverage, danger, safe, output_dir):
    """Create final visualization."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    cum_ret = (1 + results).cumprod()
    
    # Equity Curve
    ax1 = axes[0]
    ax1.semilogy(cum_ret.index, cum_ret['Benchmark (1x)'], 'gray', alpha=0.5, linewidth=1, label='Benchmark (1x)')
    ax1.semilogy(cum_ret.index, cum_ret['Always 1.5x'], 'blue', alpha=0.5, linewidth=1, label='Always 1.5x')
    ax1.semilogy(cum_ret.index, cum_ret['Stored Energy (0x-1.5x)'], '#2ca02c', linewidth=2.5, label='Stored Energy Strategy')
    
    ax1.set_ylabel('Growth of $1 (Log)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_title("The Compressed Spring Strategy: Stored Energy (1990-2024)", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Leverage / Regime
    ax2 = axes[1]
    lev_monthly = leverage.resample('M').mean()
    
    # Color by regime
    colors = ['red' if d else 'green' if s else 'gray' for d, s in zip(danger.resample('M').mean() > 0.5, safe.resample('M').mean() > 0.5)]
    ax2.bar(lev_monthly.index, lev_monthly, width=20, color='purple', alpha=0.6)
    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(0.0, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(1.5, color='green', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Leverage', fontsize=12)
    ax2.set_ylim(-0.1, 1.7)
    ax2.set_title("Dynamic Leverage: 0x (Danger) → 1.5x (Safe)", fontsize=11)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "Stored_Energy_Strategy_Final.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

def main():
    print("=== STORED ENERGY STRATEGY BACKTEST ===")
    
    px, market = load_data(LONG_HISTORY_UNIVERSE, START_DATE)
    if px.empty: return
    
    feats = calculate_features(px, WINDOW)
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    results, metrics, leverage, danger, safe = backtest_stored_energy_strategy(market, feats)
    
    print("\n=== FINAL PERFORMANCE ===")
    print(metrics)
    
    metrics.to_csv(os.path.join(output_dir, "backtest_stored_energy_final.csv"))
    
    plot_strategy(results, leverage, danger, safe, output_dir)
    
    print("\n=== BACKTEST COMPLETE ===")

if __name__ == "__main__":
    main()
