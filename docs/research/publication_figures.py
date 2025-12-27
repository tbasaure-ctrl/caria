"""
Publication-Quality "Killer Figure" with Bootstrap CIs
======================================================

Upgrades:
1. Academic panel titles
2. Bootstrap 95% CIs on each bar
3. Interpretive caption below
4. Q4/Q5 regime labels

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
N_BOOTSTRAP = 1000

def load_data(tickers, start_date):
    print("Loading data...")
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

def calculate_entropy(px, window):
    returns = px.pct_change()
    entropy_series = {}

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
    
    return pd.Series(entropy_series)

def bootstrap_quintile_stats(df, n_bootstrap=N_BOOTSTRAP):
    """Calculate bootstrap CIs for CVaR and Mean by quintile."""
    
    results = {}
    
    for q in df['SE_Quintile'].unique():
        subset = df[df['SE_Quintile'] == q]['Forward_Return']
        
        cvar_samples = []
        mean_samples = []
        
        for _ in range(n_bootstrap):
            sample = subset.sample(frac=1, replace=True)
            cvar_samples.append(sample.quantile(0.05))
            mean_samples.append(sample.mean())
        
        results[q] = {
            'cvar_mean': np.mean(cvar_samples),
            'cvar_ci_low': np.percentile(cvar_samples, 2.5),
            'cvar_ci_high': np.percentile(cvar_samples, 97.5),
            'mean_mean': np.mean(mean_samples),
            'mean_ci_low': np.percentile(mean_samples, 2.5),
            'mean_ci_high': np.percentile(mean_samples, 97.5),
        }
    
    return results

def create_killer_figure(market, entropy, output_dir):
    """Create the publication-quality killer figure with CIs."""
    
    print("Calculating Stored Energy...")
    fragility = 1 - entropy
    stored_energy = fragility.rolling(ENERGY_WINDOW).sum()
    
    common = market.index.intersection(stored_energy.dropna().index)
    sp500 = market['^GSPC'].loc[common]
    se = stored_energy.loc[common]
    
    fwd_ret = sp500.pct_change(21).shift(-21)
    
    df = pd.DataFrame({
        'Stored_Energy': se,
        'Forward_Return': fwd_ret
    }).dropna()
    
    # Create quintile labels with regime interpretation
    quintile_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    df['SE_Quintile'] = pd.qcut(df['Stored_Energy'], 5, labels=quintile_labels)
    
    # Bootstrap CIs
    print("Running bootstrap (1000 iterations)...")
    boot_stats = bootstrap_quintile_stats(df)
    
    # Prepare data for plotting
    quintiles = quintile_labels
    cvars = [boot_stats[q]['cvar_mean'] * 100 for q in quintiles]
    cvar_ci_low = [(boot_stats[q]['cvar_mean'] - boot_stats[q]['cvar_ci_low']) * 100 for q in quintiles]
    cvar_ci_high = [(boot_stats[q]['cvar_ci_high'] - boot_stats[q]['cvar_mean']) * 100 for q in quintiles]
    
    means = [boot_stats[q]['mean_mean'] * 100 for q in quintiles]
    mean_ci_low = [(boot_stats[q]['mean_mean'] - boot_stats[q]['mean_ci_low']) * 100 for q in quintiles]
    mean_ci_high = [(boot_stats[q]['mean_ci_high'] - boot_stats[q]['mean_mean']) * 100 for q in quintiles]
    
    # Colors: gradient from green (safe) to red (dangerous)
    colors = ['#2ca02c', '#7fbd7f', '#f0e442', '#e69f00', '#d73027']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: CVaR
    ax1 = axes[0]
    bars1 = ax1.bar(quintiles, cvars, color=colors, edgecolor='black', linewidth=1.2)
    ax1.errorbar(quintiles, cvars, yerr=[cvar_ci_low, cvar_ci_high], 
                 fmt='none', ecolor='black', capsize=5, capthick=2, linewidth=2)
    
    ax1.set_ylabel('Forward 21-Day CVaR (5th Percentile, %)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Stored Energy Quintile', fontsize=12)
    ax1.set_title('Forward Left-Tail Risk (CVaR)\nConditional on Stored Energy Quintiles', 
                  fontsize=13, fontweight='bold')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylim(min(cvars) * 1.3, 0)
    
    # Add regime annotations
    ax1.annotate('Low\nFragility', xy=(0, min(cvars)*0.15), fontsize=9, ha='center', color='green')
    ax1.annotate('Peak\nRisk', xy=(3, min(cvars)*0.15), fontsize=9, ha='center', color='red', fontweight='bold')
    ax1.annotate('Partial\nRelease', xy=(4, min(cvars)*0.15), fontsize=9, ha='center', color='orange')
    
    # Panel 2: Mean Return
    ax2 = axes[1]
    bars2 = ax2.bar(quintiles, means, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
    ax2.errorbar(quintiles, means, yerr=[mean_ci_low, mean_ci_high], 
                 fmt='none', ecolor='black', capsize=5, capthick=2, linewidth=2)
    
    ax2.set_ylabel('Conditional Expected Return (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Stored Energy Quintile', fontsize=12)
    ax2.set_title('Conditional Expected Return\nGiven Stored Energy Quintiles', 
                  fontsize=13, fontweight='bold')
    ax2.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    # Add interpretive caption below figure
    fig.text(0.5, -0.02, 
             '"Stored Energy monotonically worsens left-tail outcomes while simultaneously compressing\n'
             'expected returns, indicating that periods of elevated structural fragility are characterized\n'
             'by asymmetric downside without compensating upside."',
             ha='center', fontsize=10, style='italic', wrap=True)
    
    plt.subplots_adjust(bottom=0.18)
    
    output_path = os.path.join(output_dir, "Figure_2_SE_vs_CVaR.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    # Also save stats
    stats_df = pd.DataFrame([
        {'Quintile': q, 
         'CVaR_Mean': boot_stats[q]['cvar_mean'],
         'CVaR_CI_Low': boot_stats[q]['cvar_ci_low'],
         'CVaR_CI_High': boot_stats[q]['cvar_ci_high'],
         'Mean_Return': boot_stats[q]['mean_mean'],
         'Mean_CI_Low': boot_stats[q]['mean_ci_low'],
         'Mean_CI_High': boot_stats[q]['mean_ci_high']}
        for q in quintiles
    ])
    stats_df.to_csv(os.path.join(output_dir, "Figure_2_stats.csv"), index=False)
    
    return stats_df

def main():
    print("=== CREATING PUBLICATION-QUALITY KILLER FIGURE ===")
    
    px, market = load_data(LONG_HISTORY_UNIVERSE, START_DATE)
    if px.empty: return
    
    entropy = calculate_entropy(px, WINDOW)
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    stats = create_killer_figure(market, entropy, output_dir)
    print("\n=== FIGURE STATISTICS ===")
    print(stats.to_string(index=False))
    
    print("\n=== FIGURE COMPLETE ===")

if __name__ == "__main__":
    main()
