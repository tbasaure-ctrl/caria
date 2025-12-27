"""
THE COMPRESSION MATRIX: A Unique Visual Proof
==============================================

Shows the actual correlation structure at:
- LOW Stored Energy (healthy, diverse market)
- HIGH Stored Energy (fragile, "one trade" market)

This is visually striking — you literally SEE the compression.

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import os
import warnings

warnings.filterwarnings("ignore")

# Use sector ETFs for clear interpretation
SECTOR_ETFS = {
    'XLB': 'Materials',
    'XLC': 'Comm',
    'XLE': 'Energy',
    'XLF': 'Financials',
    'XLI': 'Industrials',
    'XLK': 'Tech',
    'XLP': 'Staples',
    'XLRE': 'Real Est',
    'XLU': 'Utilities',
    'XLV': 'Health',
    'XLY': 'Discret',
}

START_DATE = "2010-01-01"
WINDOW = 63
ENERGY_WINDOW = 60

def load_data(tickers, start_date):
    print("Loading sector ETF data...")
    df = yf.download(list(tickers.keys()), start=start_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        px = df['Adj Close' if 'Adj Close' in df else 'Close']
    else:
        px = df
    return px.dropna(how='all')

def calculate_entropy(px, window):
    returns = px.pct_change()
    entropy_series = {}

    for i in range(window, len(returns)):
        idx = returns.index[i]
        window_ret = returns.iloc[i-window : i]
        valid_cols = window_ret.dropna(axis=1, thresh=int(window*0.8)).columns
        
        if len(valid_cols) < 8:
            continue
            
        corr_matrix = window_ret[valid_cols].corr()
        
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

def get_correlation_at_percentile(returns, se, percentile, window=63):
    """Get average correlation matrix during high/low SE periods."""
    
    se_threshold = se.quantile(percentile)
    
    if percentile > 0.5:
        # High SE periods
        target_dates = se[se > se_threshold].index
    else:
        # Low SE periods
        target_dates = se[se < se_threshold].index
    
    # Get correlation matrices at these dates and average them
    corr_sum = None
    count = 0
    
    for date in target_dates[:50]:  # Sample 50 dates
        try:
            idx = returns.index.get_indexer([date], method='nearest')[0]
            if idx >= window and idx < len(returns):
                window_ret = returns.iloc[idx-window:idx].dropna(axis=1)
                if len(window_ret.columns) >= 8:
                    corr = window_ret.corr()
                    if corr_sum is None:
                        corr_sum = corr.copy()
                    else:
                        corr_sum = corr_sum.add(corr, fill_value=0)
                    count += 1
        except Exception as e:
            pass
    
    if corr_sum is not None and count > 0:
        return corr_sum / count
    return None

def create_compression_matrix_figure(returns, se, output_dir):
    """Create the iconic 'Compression Matrix' visualization."""
    
    # Get correlation at low and high SE
    print("Extracting correlation at LOW SE (10th percentile)...")
    low_se_corr = get_correlation_at_percentile(returns, se, 0.10)
    
    print("Extracting correlation at HIGH SE (90th percentile)...")
    high_se_corr = get_correlation_at_percentile(returns, se, 0.90)
    
    if low_se_corr is None or high_se_corr is None:
        print("ERROR: Could not compute correlations")
        return
    
    # Rename columns for readability
    low_se_corr.columns = [SECTOR_ETFS.get(c, c) for c in low_se_corr.columns]
    low_se_corr.index = [SECTOR_ETFS.get(c, c) for c in low_se_corr.index]
    high_se_corr.columns = [SECTOR_ETFS.get(c, c) for c in high_se_corr.columns]
    high_se_corr.index = [SECTOR_ETFS.get(c, c) for c in high_se_corr.index]
    
    # Calculate average off-diagonal correlation
    def avg_offdiag(corr):
        mask = ~np.eye(len(corr), dtype=bool)
        return corr.values[mask].mean()
    
    low_avg = avg_offdiag(low_se_corr)
    high_avg = avg_offdiag(high_se_corr)
    
    print(f"\n  Low SE avg correlation: {low_avg:.3f}")
    print(f"  High SE avg correlation: {high_avg:.3f}")
    print(f"  Compression: {(high_avg - low_avg)/low_avg*100:.1f}% increase")
    
    # Create figure with more vertical space
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Custom colormap
    cmap = 'RdBu_r'
    
    # Left: LOW SE (healthy market)
    ax1 = axes[0]
    sns.heatmap(low_se_corr, ax=ax1, cmap=cmap, center=0, 
                vmin=-0.5, vmax=1, annot=True, fmt='.2f', 
                annot_kws={'size': 8}, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.7, 'label': 'Correlation'})
    ax1.set_title(f'LOW Stored Energy (Diversified)\nAvg ρ = {low_avg:.2f}', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=9)
    
    # Right: HIGH SE (fragile market)
    ax2 = axes[1]
    sns.heatmap(high_se_corr, ax=ax2, cmap=cmap, center=0, 
                vmin=-0.5, vmax=1, annot=True, fmt='.2f', 
                annot_kws={'size': 8}, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.7, 'label': 'Correlation'})
    ax2.set_title(f'HIGH Stored Energy ("One Trade")\nAvg ρ = {high_avg:.2f}', 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=9)
    
    # Add arrow between panels (positioned lower)
    fig.text(0.5, 0.45, '→', fontsize=40, ha='center', va='center', 
             color='#C0392B', fontweight='bold', transform=fig.transFigure)
    fig.text(0.5, 0.38, 'COMPRESSION', fontsize=10, ha='center', va='center', 
             color='#C0392B', fontweight='bold', transform=fig.transFigure)
    
    # Main title at top with more space
    fig.suptitle('THE COMPRESSION MATRIX', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Caption at bottom
    fig.text(0.5, 0.02, 
             f'Average correlation increases from {low_avg:.2f} to {high_avg:.2f} '
             f'({(high_avg-low_avg)/low_avg*100:.0f}% increase)',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    # Save
    output_path = os.path.join(output_dir, "Figure_Compression_Matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_path}")
    
    return output_path

def main():
    print("=" * 70)
    print("THE COMPRESSION MATRIX")
    print("=" * 70)
    
    px = load_data(SECTOR_ETFS, START_DATE)
    if px.empty: 
        print("ERROR: No data")
        return
    
    returns = px.pct_change()
    
    # Calculate SE
    print("Calculating Stored Energy...")
    entropy = calculate_entropy(px, WINDOW)
    fragility = 1 - entropy
    se = fragility.rolling(ENERGY_WINDOW).sum()
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    # Create the compression matrix figure
    create_compression_matrix_figure(returns, se, output_dir)
    
    print("\n=== COMPRESSION MATRIX COMPLETE ===")

if __name__ == "__main__":
    main()
