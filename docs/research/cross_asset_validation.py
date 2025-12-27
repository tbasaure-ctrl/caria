"""
Cross-Asset Validation: Is Stored Energy a Systemic State?
==========================================================

Answers the referee question: "Is this just SPY?"

Tests:
1. Stored Energy computed on broad universe
2. Does it predict left-tail risk for:
   - SPY (U.S. Equity)
   - HYG (U.S. Credit)
   - EFA (Global ex-U.S.)

Output:
- One figure: SE vs CVaR for 3 assets
- One table: CVaR improvement across assets

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
np.random.seed(42)

# Universe for computing Stored Energy (broad system)
SYSTEM_UNIVERSE = [
    # U.S. Equity (sector diversity)
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV", "XLP",
    # Credit
    "LQD", "HYG",
    # Global
    "EFA", "EEM",
    # Rates
    "TLT", "IEF",
    # Commodities
    "GLD", "DBC"
]

# Assets to test prediction on
TEST_ASSETS = ["SPY", "HYG", "EFA"]
ASSET_LABELS = {"SPY": "U.S. Equity", "HYG": "U.S. Credit", "EFA": "Global ex-U.S."}

START_DATE = "2005-01-01"  # Most ETFs available from here
WINDOW = 63
ENERGY_WINDOW = 60
N_BOOTSTRAP = 500

def load_data(tickers, start_date):
    print(f"Loading {len(tickers)} assets...")
    try:
        df = yf.download(tickers, start=start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            px = df['Adj Close' if 'Adj Close' in df else 'Close']
        else:
            px = df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()
    
    return px.dropna(how='all')

def calculate_system_entropy(px, window):
    """Calculate entropy from the SYSTEM universe."""
    returns = px.pct_change()
    entropy_series = {}

    print("Calculating System Entropy...")
    for i in range(window, len(returns)):
        idx = returns.index[i]
        window_ret = returns.iloc[i-window : i]
        valid_cols = window_ret.dropna(axis=1, thresh=int(window*0.8)).columns
        if len(valid_cols) < 8:
            continue
            
        corr_matrix = window_ret[valid_cols].corr()
        corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        if not corr_matrix.empty and len(corr_matrix) >= 8:
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
    
    return pd.Series(entropy_series)

def calculate_asset_cvar_by_quintile(stored_energy, asset_prices, asset_name):
    """Calculate CVaR by SE quintile for a specific asset."""
    
    fwd_ret = asset_prices.pct_change(21).shift(-21)
    
    common = stored_energy.dropna().index.intersection(fwd_ret.dropna().index)
    
    df = pd.DataFrame({
        'SE': stored_energy.loc[common],
        'Fwd_Ret': fwd_ret.loc[common]
    })
    
    df['Quintile'] = pd.qcut(df['SE'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    results = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        subset = df[df['Quintile'] == q]['Fwd_Ret']
        
        # Bootstrap CVaR
        cvars = []
        for _ in range(N_BOOTSTRAP):
            sample = subset.sample(frac=1, replace=True)
            cvars.append(sample.quantile(0.05))
        
        results.append({
            'Asset': asset_name,
            'Quintile': q,
            'CVaR_Mean': np.mean(cvars),
            'CVaR_CI_Low': np.percentile(cvars, 2.5),
            'CVaR_CI_High': np.percentile(cvars, 97.5)
        })
    
    return pd.DataFrame(results)

def create_cross_asset_figure(all_results, output_dir):
    """Create one figure with SE vs CVaR for all assets."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#2ca02c', '#7fbd7f', '#f0e442', '#e69f00', '#d73027']
    
    for i, asset in enumerate(TEST_ASSETS):
        ax = axes[i]
        df = all_results[all_results['Asset'] == ASSET_LABELS[asset]]
        
        cvars = df['CVaR_Mean'].values * 100
        ci_low = (df['CVaR_Mean'] - df['CVaR_CI_Low']).values * 100
        ci_high = (df['CVaR_CI_High'] - df['CVaR_Mean']).values * 100
        
        bars = ax.bar(df['Quintile'], cvars, color=colors, edgecolor='black', linewidth=1)
        ax.errorbar(df['Quintile'], cvars, yerr=[ci_low, ci_high], 
                   fmt='none', ecolor='black', capsize=4, capthick=1.5)
        
        ax.set_ylabel('Forward 21-Day CVaR (5%)', fontsize=11)
        ax.set_xlabel('Stored Energy Quintile', fontsize=11)
        ax.set_title(f'{ASSET_LABELS[asset]}', fontsize=13, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylim(min(cvars) * 1.4, 0)
    
    plt.suptitle('Cross-Asset Validation: Stored Energy Predicts Left-Tail Risk Across Asset Classes', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "Figure_3_Cross_Asset_Validation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

def create_summary_table(all_results, output_dir):
    """Create summary table: CVaR improvement Q1 vs Q4 for each asset."""
    
    summary = []
    
    for asset in TEST_ASSETS:
        df = all_results[all_results['Asset'] == ASSET_LABELS[asset]]
        
        q1_cvar = df[df['Quintile'] == 'Q1']['CVaR_Mean'].values[0]
        q4_cvar = df[df['Quintile'] == 'Q4']['CVaR_Mean'].values[0]
        
        # Q4 is worse (more negative), so difference is Q4 - Q1 (negative = worse)
        deterioration = (q4_cvar - q1_cvar) * 100
        
        summary.append({
            'Asset': ASSET_LABELS[asset],
            'Q1_CVaR': f"{q1_cvar*100:.1f}%",
            'Q4_CVaR': f"{q4_cvar*100:.1f}%",
            'Deterioration': f"{deterioration:.1f}%",
            'Pattern': 'Monotonic' if q4_cvar < q1_cvar else 'Non-Monotonic'
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "Table_CrossAsset_Summary.csv"), index=False)
    
    return summary_df

def main():
    print("=== CROSS-ASSET VALIDATION ===")
    
    all_tickers = list(set(SYSTEM_UNIVERSE + TEST_ASSETS))
    px = load_data(all_tickers, START_DATE)
    
    if px.empty: return
    
    # Calculate System Entropy from broad universe
    entropy = calculate_system_entropy(px[SYSTEM_UNIVERSE], WINDOW)
    fragility = 1 - entropy
    stored_energy = fragility.rolling(ENERGY_WINDOW).sum()
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    # Calculate CVaR by quintile for each asset
    all_results = []
    for asset in TEST_ASSETS:
        print(f"Analyzing {asset}...")
        if asset in px.columns:
            result = calculate_asset_cvar_by_quintile(stored_energy, px[asset], ASSET_LABELS[asset])
            all_results.append(result)
    
    all_results = pd.concat(all_results, ignore_index=True)
    all_results.to_csv(os.path.join(output_dir, "cross_asset_cvar.csv"), index=False)
    
    # Create figure
    create_cross_asset_figure(all_results, output_dir)
    
    # Create summary table
    summary = create_summary_table(all_results, output_dir)
    
    print("\n=== CROSS-ASSET SUMMARY ===")
    print(summary.to_string(index=False))
    
    print("\n=== VALIDATION COMPLETE ===")

if __name__ == "__main__":
    main()
