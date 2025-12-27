"""
Accumulated Fragility: The "Stored Energy" Hypothesis
=====================================================

Insight from User:
    Fragility "accumulates energy" during low entropy periods.
    The longer the system stays fragile, the more potential energy builds up.

New Metric: Stored Energy
    SE(t) = Rolling Sum of Fragility over past N days
    (or exponentially weighted to give more weight to recent fragility)

Hypothesis:
    High "Stored Energy" = High probability of imminent volatility explosion.

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.metrics import roc_auc_score
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
ENERGY_WINDOW = 60  # 3 months of accumulation
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
    
    # STORED ENERGY: Integral of Fragility over time
    # Method 1: Simple Rolling Sum
    df['Stored_Energy'] = df['Fragility'].rolling(ENERGY_WINDOW).sum()
    
    # Method 2: Exponentially Weighted (more weight to recent fragility)
    df['Stored_Energy_EWM'] = df['Fragility'].ewm(span=ENERGY_WINDOW).sum()
    
    # Method 3: Only count "high fragility" days (above median)
    frag_above_median = (df['Fragility'] > df['Fragility'].median()).astype(float)
    df['Days_High_Frag'] = frag_above_median.rolling(ENERGY_WINDOW).sum()
    
    return df.dropna()

def analyze_energy_release(market, features):
    """Analyze what happens after high 'Stored Energy' periods."""
    
    common = market.index.intersection(features.index)
    sp500 = market['^GSPC'].loc[common]
    feats = features.loc[common]
    
    # Forward returns (next 21 days)
    fwd_ret = sp500.pct_change(21).shift(-21)
    
    # Future Volatility (next 21 days)
    sp500_ret = sp500.pct_change()
    fwd_vol = sp500_ret.rolling(21).std().shift(-21) * np.sqrt(252)
    
    df = pd.DataFrame({
        'Stored_Energy': feats['Stored_Energy'],
        'Current_Vol': feats['Vol'],
        'Fwd_Ret': fwd_ret,
        'Fwd_Vol': fwd_vol
    }).dropna()
    
    # Split by Stored Energy quartiles
    df['SE_Quartile'] = pd.qcut(df['Stored_Energy'], 4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High'])
    
    print("\n=== STORED ENERGY ANALYSIS ===")
    print("\nFuture Returns by Stored Energy Quartile:")
    summary = df.groupby('SE_Quartile').agg({
        'Fwd_Ret': ['mean', 'std'],
        'Fwd_Vol': 'mean',
        'Current_Vol': 'mean'
    })
    print(summary)
    
    # Key comparison: High SE vs Low SE
    high_se = df[df['SE_Quartile'] == 'Q4_High']
    low_se = df[df['SE_Quartile'] == 'Q1_Low']
    
    print("\n=== KEY FINDING ===")
    print(f"High Stored Energy (Q4):")
    print(f"  Current Vol: {high_se['Current_Vol'].mean():.2%}")
    print(f"  Future Vol:  {high_se['Fwd_Vol'].mean():.2%} (+{(high_se['Fwd_Vol'].mean() / high_se['Current_Vol'].mean() - 1):.0%} explosion)")
    print(f"  Future Ret:  {high_se['Fwd_Ret'].mean():.2%}")
    
    print(f"\nLow Stored Energy (Q1):")
    print(f"  Current Vol: {low_se['Current_Vol'].mean():.2%}")
    print(f"  Future Vol:  {low_se['Fwd_Vol'].mean():.2%}")
    print(f"  Future Ret:  {low_se['Fwd_Ret'].mean():.2%}")
    
    # AUC for predicting volatility explosion
    # Define "explosion" = future vol > 90th percentile
    vol_explosion = (fwd_vol > fwd_vol.quantile(0.90)).astype(int).loc[df.index]
    
    auc_se = roc_auc_score(vol_explosion, df['Stored_Energy'])
    auc_vol = roc_auc_score(vol_explosion, df['Current_Vol'])
    
    print(f"\n=== AUC for Predicting Volatility Explosion ===")
    print(f"  Current Volatility: {auc_vol:.4f}")
    print(f"  Stored Energy:      {auc_se:.4f}")
    
    if auc_se > auc_vol:
        print(f"  ✓ Stored Energy BEATS Current Vol by {auc_se - auc_vol:.4f}")
    
    return df

def create_energy_visualization(market, features, output_dir):
    """Visualize the energy accumulation and release."""
    
    common = market.index.intersection(features.index)
    sp500 = market['^GSPC'].loc[common]
    feats = features.loc[common]
    
    # Normalize Stored Energy for plotting
    se_norm = (feats['Stored_Energy'] - feats['Stored_Energy'].min()) / (feats['Stored_Energy'].max() - feats['Stored_Energy'].min())
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Panel 1: S&P 500
    ax1 = axes[0]
    ax1.semilogy(sp500.index, sp500.values, 'black', linewidth=1.5, label='S&P 500')
    ax1.set_ylabel('S&P 500 (Log)', fontsize=11)
    ax1.set_title("The Physics of Risk: Stored Energy in Structure", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    
    # Mark crises
    for name, date in [('Dot-Com', '2000-03-24'), ('GFC', '2007-10-09'), ('COVID', '2020-02-19')]:
        try:
            ax1.axvline(pd.Timestamp(date), color='red', linestyle='--', alpha=0.5)
        except:
            pass
    
    # Panel 2: Stored Energy (The Spring)
    ax2 = axes[1]
    ax2.fill_between(se_norm.index, 0, se_norm, 
                     where=se_norm > 0.7, color='red', alpha=0.7, label='High Stored Energy (Compressed Spring)')
    ax2.fill_between(se_norm.index, 0, se_norm, 
                     where=se_norm <= 0.7, color='blue', alpha=0.3, label='Normal/Low')
    ax2.set_ylabel('Stored Energy\n(Cumulative Fragility)', fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Danger Threshold')
    ax2.legend(loc='upper left', fontsize=9)
    
    # Panel 3: Volatility (The Release)
    ax3 = axes[2]
    ax3.fill_between(feats.index, 0, feats['Vol'], color='purple', alpha=0.5)
    ax3.set_ylabel('Realized Volatility\n(Energy Released)', fontsize=11)
    ax3.set_xlabel('')
    
    # Mark volatility spikes
    vol_spikes = feats['Vol'] > feats['Vol'].quantile(0.95)
    ax3.scatter(feats.index[vol_spikes], feats['Vol'][vol_spikes], color='red', s=20, zorder=5, label='Vol Explosion')
    ax3.legend(loc='upper left', fontsize=9)
    
    ax3.xaxis.set_major_locator(mdates.YearLocator(5))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "Stored_Energy_Analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

def main():
    print("=== STORED ENERGY ANALYSIS ===")
    
    px, market = load_data(LONG_HISTORY_UNIVERSE, START_DATE)
    if px.empty: return
    
    feats = calculate_features(px, WINDOW)
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    # Analyze
    df = analyze_energy_release(market, feats)
    
    # Visualize
    create_energy_visualization(market, feats, output_dir)
    
    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()
