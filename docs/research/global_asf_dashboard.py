"""
Global ASF Dashboard: December 2023 to December 2024
=====================================================

Creates a publication-quality figure showing Accumulated Spectral Fragility
for the 5 major economies over the past year.

Economies tracked:
- United States (SPY + Sectors)
- Europe (VGK + Country ETFs)
- Japan (EWJ)
- China (FXI/MCHI)
- United Kingdom (EWU)

Author: Research Team
Date: December 2024
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")

# Output directory
OUTPUT_DIR = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"

# Define economy-specific universes
ECONOMIES = {
    'United States': {
        'tickers': ['SPY', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB'],
        'color': '#1f77b4',
        'benchmark': 'SPY'
    },
    'Europe': {
        'tickers': ['VGK', 'EWG', 'EWQ', 'EWI', 'EWP', 'EWN', 'EWK', 'EIRL'],
        'color': '#2ca02c',
        'benchmark': 'VGK'
    },
    'Japan': {
        'tickers': ['EWJ', 'DXJ', 'HEWJ', 'JPXN', 'BBJP', 'FLJP'],
        'color': '#d62728',
        'benchmark': 'EWJ'
    },
    'China': {
        'tickers': ['FXI', 'MCHI', 'KWEB', 'CQQQ', 'GXC', 'ASHR'],
        'color': '#ff7f0e',
        'benchmark': 'FXI'
    },
    'United Kingdom': {
        'tickers': ['EWU', 'FLGB', 'HEWU'],
        'color': '#9467bd',
        'benchmark': 'EWU'
    }
}

# Parameters
WINDOW = 63  # ~3 months for correlation estimation
DECAY_LAMBDA = 0.005  # Optimal from our tests (half-life = 139 days)


def load_economy_data(tickers, start_date, end_date):
    """Load price data for an economy's ETFs."""
    try:
        df = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            prices = df['Adj Close'] if 'Adj Close' in df.columns.get_level_values(0) else df['Close']
        else:
            prices = df
        return prices.dropna(axis=1, how='all')
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def calculate_entropy(corr_matrix):
    """Calculate normalized spectral entropy from correlation matrix."""
    try:
        eigvals = np.linalg.eigvalsh(corr_matrix)
        eigvals = eigvals[eigvals > 1e-10]
        probs = eigvals / np.sum(eigvals)
        S = -np.sum(probs * np.log(probs))
        N = len(probs)
        return S / np.log(N) if N > 1 else 1.0
    except:
        return np.nan


def calculate_asf_series(returns, window=63):
    """Calculate ASF time series with decay weighting."""
    fragility = {}
    
    for i in range(window, len(returns)):
        idx = returns.index[i]
        window_ret = returns.iloc[i-window:i].dropna(axis=1)
        
        if len(window_ret.columns) >= 3:  # Need at least 3 assets
            try:
                # Ledoit-Wolf shrinkage
                lw = LedoitWolf()
                lw.fit(window_ret.values)
                cov = lw.covariance_
                
                # Convert to correlation
                std = np.sqrt(np.diag(cov))
                corr = cov / np.outer(std, std)
                np.fill_diagonal(corr, 1.0)
                
                entropy = calculate_entropy(corr)
                fragility[idx] = 1 - entropy
            except:
                fragility[idx] = np.nan
    
    frag_series = pd.Series(fragility)
    
    # Apply exponential decay weighting
    asf = frag_series.ewm(halflife=139).mean()
    
    return asf


def main():
    print("=" * 70)
    print("GLOBAL ASF DASHBOARD: December 2024 - December 2025")
    print("=" * 70)
    
    # Date range: last 12 months + buffer for calculation
    end_date = datetime(2025, 12, 20)
    start_date = datetime(2024, 9, 1)  # Buffer for window calculation
    
    # Calculate ASF for each economy
    economy_asf = {}
    economy_current = {}
    
    for economy, config in ECONOMIES.items():
        print(f"\nProcessing {economy}...")
        
        prices = load_economy_data(config['tickers'], start_date, end_date)
        
        if prices.empty or len(prices.columns) < 3:
            print(f"  Insufficient data for {economy}")
            continue
        
        returns = prices.pct_change().dropna()
        print(f"  Loaded {len(prices.columns)} assets, {len(returns)} days")
        
        asf = calculate_asf_series(returns, window=WINDOW)
        
        # Filter to last 12 months only
        asf_12m = asf[asf.index >= '2024-12-01']
        
        if len(asf_12m) > 0:
            economy_asf[economy] = asf_12m
            economy_current[economy] = asf_12m.iloc[-1]
            print(f"  Current ASF: {asf_12m.iloc[-1]:.3f}")
    
    # Create publication figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top panel: ASF time series
    ax1 = axes[0]
    
    for economy, asf in economy_asf.items():
        color = ECONOMIES[economy]['color']
        ax1.plot(asf.index, asf.values, label=economy, color=color, linewidth=2)
    
    # Danger zone
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Danger Threshold (0.7)')
    ax1.fill_between(ax1.get_xlim(), 0.7, 1.0, alpha=0.1, color='red')
    
    ax1.set_ylabel('Accumulated Spectral Fragility', fontsize=12)
    ax1.set_title('Global Structural Fragility: December 2024 - December 2025', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add key events
    events = [
        ('2025-01-20', 'Trump Inauguration'),
        ('2025-03-19', 'Fed Decision'),
        ('2025-08-05', 'Market Correction'),
        ('2025-12-01', 'Year End'),
    ]
    
    for date_str, label in events:
        try:
            event_date = pd.Timestamp(date_str)
            if event_date >= ax1.get_xlim()[0]:
                ax1.axvline(x=event_date, color='gray', linestyle=':', alpha=0.5)
                ax1.text(event_date, 0.95, label, rotation=90, va='top', ha='right', fontsize=8, alpha=0.7)
        except:
            pass
    
    # Bottom panel: Current levels bar chart
    ax2 = axes[1]
    
    economies_sorted = sorted(economy_current.items(), key=lambda x: x[1], reverse=True)
    names = [e[0] for e in economies_sorted]
    values = [e[1] for e in economies_sorted]
    colors = [ECONOMIES[e]['color'] for e in names]
    
    bars = ax2.barh(names, values, color=colors, edgecolor='black', linewidth=1)
    
    # Add danger line
    ax2.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='Danger Threshold')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                va='center', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Current ASF Level (December 2025)', fontsize=12)
    ax2.set_title('Current Structural Fragility by Economy', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Subtitle with interpretation
    fig.text(0.5, 0.01, 
             'ASF > 0.7 = Danger Zone (High fragility, low volatility) | Lower ASF = More diversified/robust structure',
             ha='center', fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "Figure_Global_ASF_Dashboard.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = os.path.join(OUTPUT_DIR, "Figure_Global_ASF_Dashboard.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to: {pdf_path}")
    
    plt.show()
    
    # Print summary table
    print("\n" + "=" * 50)
    print("CURRENT ASF LEVELS (December 2025)")
    print("=" * 50)
    print(f"{'Economy':<20} {'ASF':>10} {'Status':>15}")
    print("-" * 50)
    
    for economy, asf_val in economies_sorted:
        if asf_val > 0.7:
            status = "DANGER ZONE"
        elif asf_val > 0.5:
            status = "Elevated"
        else:
            status = "Normal"
        print(f"{economy:<20} {asf_val:>10.3f} {status:>15}")
    
    print("\n" + "=" * 50)
    
    # Save summary to CSV
    summary_df = pd.DataFrame({
        'Economy': names,
        'Current_ASF': values,
        'Status': ['DANGER ZONE' if v > 0.7 else 'Elevated' if v > 0.5 else 'Normal' for v in values]
    })
    csv_path = os.path.join(OUTPUT_DIR, "Table_Global_ASF_Current.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary to: {csv_path}")


if __name__ == "__main__":
    main()
