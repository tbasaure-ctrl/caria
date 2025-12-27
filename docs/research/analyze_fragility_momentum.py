"""
Fragility Momentum: Detecting the Acceleration of Risk
======================================================

Hypothesis:
    The LEVEL of fragility might lag (signal when crash is already happening).
    The CHANGE (momentum) or ACCELERATION of fragility could LEAD.

Signals:
    1. dF/dt (Velocity): Rate of change of Fragility (20-day ROC)
    2. d²F/dt²(Acceleration): Rate of change of Velocity (20-day ROC of ROC)

Strategy Test:
    - "Momentum Warning": Hedge when Fragility is ACCELERATING sharply upward.

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

START_DATE = "1990-01-01"
WINDOW = 63
ROC_WINDOW = 21  # 1 month for rate of change
ROLL_RANK = 252 * 2

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
    market_ret = returns.mean(axis=1)

    print("Calculating Entropy...")
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
            
    df = pd.DataFrame({
        'Entropy': pd.Series(entropy_series),
    })
    
    # Fragility = 1 - Entropy (Higher = More Fragile)
    df['Fragility'] = 1 - df['Entropy']
    
    # First Derivative: Rate of Change (Velocity)
    df['dF_dt'] = df['Fragility'].diff(ROC_WINDOW) / ROC_WINDOW
    
    # Second Derivative: Acceleration
    df['d2F_dt2'] = df['dF_dt'].diff(ROC_WINDOW) / ROC_WINDOW
    
    # Volatility for comparison
    vol_series = market_ret.rolling(WINDOW).std() * np.sqrt(252)
    df['Vol'] = vol_series
    
    return df.dropna()

def create_momentum_visualization(market, features, output_dir):
    """Create visualization with Fragility Momentum."""
    
    common = market.index.intersection(features.index)
    sp500 = market['^GSPC'].loc[common]
    feats = features.loc[common]
    
    # Rank the acceleration (for signal)
    accel_rank = feats['d2F_dt2'].rolling(ROLL_RANK, min_periods=252).rank(pct=True)
    
    # Signal: Acceleration in top 10% (Fragility is sharply accelerating)
    accel_warning = accel_rank > 0.90
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # --- Panel 1: S&P 500 with Acceleration Warnings ---
    ax1 = axes[0]
    ax1.semilogy(sp500.index, sp500.values, color='black', linewidth=1.5, label='S&P 500')
    
    # Shade Acceleration Warning periods
    signal_diff = accel_warning.astype(int).diff().fillna(0)
    starts = accel_warning.index[signal_diff == 1]
    ends = accel_warning.index[signal_diff == -1]
    
    for i, start in enumerate(starts):
        if i < len(ends):
            ax1.axvspan(start, ends[i], alpha=0.3, color='orange', label='_nolegend_')
    
    ax1.axvspan(pd.Timestamp('1900-01-01'), pd.Timestamp('1900-01-02'), 
                alpha=0.3, color='orange', label='Fragility ACCELERATING (d²F/dt² > 90th)')
    
    # Mark crises
    for name, date in [('Dot-Com', '2000-03-24'), ('GFC', '2007-10-09'), ('COVID', '2020-02-19')]:
        try:
            ax1.axvline(pd.Timestamp(date), color='red', linestyle='--', alpha=0.5)
        except:
            pass
    
    ax1.set_ylabel('S&P 500 (Log)', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.set_title("Fragility Momentum: Acceleration as Early Warning", fontsize=14, fontweight='bold')
    
    # --- Panel 2: Fragility Level vs First Derivative (Velocity) ---
    ax2 = axes[1]
    ax2.plot(feats.index, feats['Fragility'], color='green', alpha=0.7, label='Fragility Level')
    ax2.axhline(feats['Fragility'].quantile(0.80), color='green', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Fragility Level', fontsize=11, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax2b = ax2.twinx()
    ax2b.plot(feats.index, feats['dF_dt'], color='blue', alpha=0.6, label='Velocity (dF/dt)')
    ax2b.axhline(0, color='blue', linestyle='-', alpha=0.3)
    ax2b.set_ylabel('Velocity (dF/dt)', fontsize=11, color='blue')
    ax2b.tick_params(axis='y', labelcolor='blue')
    
    ax2.legend(loc='upper left')
    ax2b.legend(loc='upper right')
    
    # --- Panel 3: Acceleration (Second Derivative) ---
    ax3 = axes[2]
    ax3.fill_between(feats.index, 0, feats['d2F_dt2'], where=feats['d2F_dt2'] > 0, 
                     color='red', alpha=0.5, label='Accelerating Risk')
    ax3.fill_between(feats.index, 0, feats['d2F_dt2'], where=feats['d2F_dt2'] <= 0, 
                     color='green', alpha=0.5, label='Decelerating Risk')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.axhline(feats['d2F_dt2'].quantile(0.90), color='orange', linestyle='--', alpha=0.7, label='90th Percentile')
    ax3.set_ylabel('Acceleration (d²F/dt²)', fontsize=11)
    ax3.set_xlabel('')
    ax3.legend(loc='upper left')
    
    # Format x-axis
    ax3.xaxis.set_major_locator(mdates.YearLocator(5))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "Fragility_Momentum_Timeline.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path, feats, accel_warning

def backtest_momentum_strategy(market, features, accel_warning):
    """Backtest: Hedge when Fragility is ACCELERATING."""
    
    common = market.index.intersection(features.index)
    sp500_ret = market['^GSPC'].loc[common].pct_change()
    
    # Simple cash return (0) when hedging
    cash_ret = pd.Series(0, index=common)
    
    # Strategy: Hedge when Acceleration Warning is ON
    pos = accel_warning.shift(1).fillna(False)
    strategy_ret = np.where(pos, cash_ret, sp500_ret)
    
    # Calculate metrics
    results = pd.DataFrame({
        'Benchmark': sp500_ret,
        'Momentum_Warning': strategy_ret
    }).dropna()
    
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
    
    return pd.DataFrame(metrics).T

def main():
    print("=== FRAGILITY MOMENTUM ANALYSIS ===")
    
    px, market = load_data(LONG_HISTORY_UNIVERSE, START_DATE)
    if px.empty: return
    
    feats = calculate_features(px, WINDOW)
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    # Visualization
    _, feats_aligned, accel_warning = create_momentum_visualization(market, feats, output_dir)
    
    # Backtest
    metrics = backtest_momentum_strategy(market, feats, accel_warning)
    print("\n=== BACKTEST: Acceleration Warning Strategy ===")
    print(metrics)
    
    metrics.to_csv(os.path.join(output_dir, "backtest_momentum_metrics.csv"))
    
    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()
