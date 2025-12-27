"""
Dynamic Leverage Minsky Strategy: 0x to 1.5x Based on Structural Risk
=====================================================================

Strategy:
    - DANGER (0x): Minsky Zone (High Fragility + Low Vol) AND Fragility Accelerating
    - NEUTRAL (1x): Normal state
    - SAFE (1.5x): Low Fragility (High Entropy) AND Fragility Decelerating

Hypothesis:
    By scaling exposure with structural risk, we can:
    1. Avoid drawdowns during dangerous regimes
    2. Amplify returns during safe regimes
    3. Outperform on risk-adjusted basis

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

START_DATE = "1990-01-01"
WINDOW = 63
ROC_WINDOW = 21
ROLL_RANK = 252 * 2
LEVERAGE_MULTIPLIER = 1.5
BORROW_COST_ANNUAL = 0.02  # 2% annual cost for leverage

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
        vol_series[idx] = market_ret.iloc[i-window : i].std() * np.sqrt(252)
            
    df = pd.DataFrame({
        'Entropy': pd.Series(entropy_series),
        'Vol': pd.Series(vol_series)
    })
    
    # Fragility = 1 - Entropy
    df['Fragility'] = 1 - df['Entropy']
    
    # Derivatives
    df['dF_dt'] = df['Fragility'].diff(ROC_WINDOW)
    df['d2F_dt2'] = df['dF_dt'].diff(ROC_WINDOW)
    
    return df.dropna()

def calculate_leverage(features):
    """Calculate dynamic leverage based on structural signals."""
    
    # Ranks (Rolling)
    frag_rank = features['Fragility'].rolling(ROLL_RANK, min_periods=252).rank(pct=True)
    vol_rank = features['Vol'].rolling(ROLL_RANK, min_periods=252).rank(pct=True)
    accel = features['d2F_dt2']
    
    # Define Regimes
    # DANGER: High Fragility (>80%) + Low Vol (<50%) + Accelerating (>0)
    is_minsky_zone = (frag_rank > 0.80) & (vol_rank < 0.50)
    is_accelerating = accel > 0
    danger = is_minsky_zone & is_accelerating
    
    # SAFE: Low Fragility (<30%) + Decelerating
    is_healthy = frag_rank < 0.30
    is_decelerating = accel < 0
    safe = is_healthy & is_decelerating
    
    # Build Leverage Series
    leverage = pd.Series(1.0, index=features.index)  # Default: 1x
    leverage[danger] = 0.0  # Danger: 0x
    leverage[safe] = LEVERAGE_MULTIPLIER  # Safe: 1.5x
    
    # Shift by 1 day (signal = end of day, trade next day)
    leverage = leverage.shift(1).fillna(1.0)
    
    return leverage, danger, safe

def backtest_leveraged_strategy(market, features):
    """Backtest the dynamic leverage strategy."""
    
    common = market.index.intersection(features.index)
    sp500_ret = market['^GSPC'].loc[common].pct_change()
    
    leverage, danger, safe = calculate_leverage(features.loc[common])
    
    # Daily borrow cost for leverage (only when leverage > 1)
    daily_borrow_cost = (BORROW_COST_ANNUAL / 252) * (leverage - 1).clip(lower=0)
    
    # Strategy Return
    strategy_ret = sp500_ret * leverage - daily_borrow_cost
    
    # Calculate metrics
    results = pd.DataFrame({
        'Benchmark (1x)': sp500_ret,
        'Dynamic (0x-1.5x)': strategy_ret
    }).dropna()
    
    # Also calculate a simple 1.5x always (for comparison)
    always_levered = sp500_ret * 1.5 - (BORROW_COST_ANNUAL / 252) * 0.5
    results['Always 1.5x'] = always_levered
    
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
    
    # Calculate time in each regime
    time_danger = danger.loc[common].mean()
    time_safe = safe.loc[common].mean()
    time_neutral = 1 - time_danger - time_safe
    
    print(f"\n  Regime Allocation:")
    print(f"    DANGER (0x): {time_danger:.1%}")
    print(f"    NEUTRAL (1x): {time_neutral:.1%}")
    print(f"    SAFE (1.5x): {time_safe:.1%}")
    
    return results, metrics_df, leverage

def plot_leveraged_strategy(results, leverage, output_dir):
    """Create visualization."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Equity Curve
    ax1 = axes[0]
    cum_ret = (1 + results).cumprod()
    
    ax1.semilogy(cum_ret.index, cum_ret['Benchmark (1x)'], 'gray', alpha=0.5, linewidth=1, label='Benchmark (1x)')
    ax1.semilogy(cum_ret.index, cum_ret['Always 1.5x'], 'blue', alpha=0.5, linewidth=1, label='Always 1.5x')
    ax1.semilogy(cum_ret.index, cum_ret['Dynamic (0x-1.5x)'], '#2ca02c', linewidth=2, label='Dynamic Minsky (0x-1.5x)')
    
    ax1.set_ylabel('Growth of $1 (Log Scale)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_title("Dynamic Leverage Strategy: Scale Risk with Structure", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Leverage Over Time
    ax2 = axes[1]
    # Resample for cleaner plot
    lev_monthly = leverage.resample('M').mean()
    ax2.fill_between(lev_monthly.index, 0, lev_monthly, color='purple', alpha=0.5)
    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Leverage', fontsize=12)
    ax2.set_xlabel('')
    ax2.set_ylim(0, 1.7)
    ax2.set_title("Leverage Allocation Over Time", fontsize=11)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "Dynamic_Leverage_Strategy.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

def main():
    print("=== DYNAMIC LEVERAGE MINSKY STRATEGY ===")
    
    px, market = load_data(LONG_HISTORY_UNIVERSE, START_DATE)
    if px.empty: return
    
    feats = calculate_features(px, WINDOW)
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    results, metrics, leverage = backtest_leveraged_strategy(market, feats)
    
    print("\n=== PERFORMANCE COMPARISON ===")
    print(metrics)
    
    metrics.to_csv(os.path.join(output_dir, "backtest_leveraged_metrics.csv"))
    
    plot_leveraged_strategy(results, leverage, output_dir)
    
    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()
