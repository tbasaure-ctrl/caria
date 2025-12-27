"""
THE ICONIC FIGURE: 35 Years of Stored Energy
=============================================

A publication-quality, "mind-blowing" figure showing:
- S&P 500 price (log scale)
- Stored Energy as filled area below
- Vertical annotations for major crises
- Clear visual of SE building BEFORE crashes

This is Figure 1 — the figure reviewers remember.

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import yfinance as yf
import os
import warnings

warnings.filterwarnings("ignore")

# Long history universe (stocks with data back to 1990)
LONG_HISTORY_UNIVERSE = [
    "AAPL", "MSFT", "INTC", "IBM", "ORCL",
    "PG", "KO", "PEP", "JNJ", "WMT", "MCD",
    "JPM", "BAC", "WFC", "GS",
    "XOM", "CVX",
    "GE", "MMM", "CAT", "BA",
    "MRK", "PFE", "ABT",
]

START_DATE = "1990-01-01"
WINDOW = 63
ENERGY_WINDOW = 60

# Major historical events
EVENTS = [
    {'name': 'Black Monday\n1987', 'date': '1987-10-19', 'type': 'flash'},
    {'name': 'LTCM\n1998', 'date': '1998-08-31', 'type': 'slow'},
    {'name': 'Dot-Com\nPeak', 'date': '2000-03-24', 'type': 'slow'},
    {'name': 'Dot-Com\nBottom', 'date': '2002-10-09', 'type': 'recovery'},
    {'name': 'GFC\nStart', 'date': '2007-10-09', 'type': 'slow'},
    {'name': 'Lehman', 'date': '2008-09-15', 'type': 'slow'},
    {'name': 'GFC\nBottom', 'date': '2009-03-09', 'type': 'recovery'},
    {'name': 'Flash\nCrash', 'date': '2010-05-06', 'type': 'flash'},
    {'name': 'VIX\nSpike', 'date': '2015-08-24', 'type': 'flash'},
    {'name': 'Volmageddon', 'date': '2018-02-05', 'type': 'flash'},
    {'name': 'COVID\nCrash', 'date': '2020-03-23', 'type': 'exogenous'},
    {'name': '2022\nBear', 'date': '2022-10-12', 'type': 'slow'},
]

# Crisis periods (for shading)
CRISIS_PERIODS = [
    {'name': 'Dot-Com', 'start': '2000-03-01', 'end': '2002-10-01', 'color': '#FF6B6B'},
    {'name': 'GFC', 'start': '2007-10-01', 'end': '2009-03-01', 'color': '#FF6B6B'},
    {'name': 'COVID', 'start': '2020-02-01', 'end': '2020-04-01', 'color': '#FFE66D'},
    {'name': '2022', 'start': '2022-01-01', 'end': '2022-10-01', 'color': '#FFE66D'},
]

def load_data(tickers, start_date):
    print("Loading historical data...")
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

def create_iconic_figure(sp500, stored_energy, output_dir):
    """Create the mind-blowing historical figure."""
    
    # Align data
    common = sp500.index.intersection(stored_energy.dropna().index)
    sp = sp500.loc[common]
    se = stored_energy.loc[common]
    
    # Normalize SE for plotting (0-100 scale)
    se_norm = (se - se.min()) / (se.max() - se.min()) * 100
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Color scheme
    sp_color = '#2C3E50'  # Dark blue-gray
    se_color = '#E74C3C'  # Coral red
    
    # Plot S&P 500 (log scale)
    ax1.semilogy(sp.index, sp.values, color=sp_color, linewidth=1.5, label='S&P 500', alpha=0.9)
    ax1.set_ylabel('S&P 500 (Log Scale)', fontsize=12, color=sp_color, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=sp_color)
    ax1.set_ylim([sp.min() * 0.9, sp.max() * 1.1])
    
    # Add crisis period shading
    for crisis in CRISIS_PERIODS:
        try:
            start = pd.Timestamp(crisis['start'])
            end = pd.Timestamp(crisis['end'])
            ax1.axvspan(start, end, alpha=0.2, color=crisis['color'], zorder=0)
        except:
            pass
    
    # Second axis for SE
    ax2 = ax1.twinx()
    
    # Plot SE as filled area (inverted so high SE = danger at top)
    ax2.fill_between(se.index, 0, se_norm.values, alpha=0.4, color=se_color, label='Stored Energy')
    ax2.plot(se.index, se_norm.values, color=se_color, linewidth=1, alpha=0.7)
    ax2.set_ylabel('Stored Energy (Normalized 0-100)', fontsize=12, color=se_color, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=se_color)
    ax2.set_ylim([0, 120])
    
    # Add danger zone
    ax2.axhline(y=80, color=se_color, linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(se.index[10], 82, 'DANGER ZONE', fontsize=9, color=se_color, alpha=0.7)
    
    # Add event annotations
    for event in EVENTS:
        try:
            event_date = pd.Timestamp(event['date'])
            if event_date in se.index or (event_date >= se.index.min() and event_date <= se.index.max()):
                # Find closest date
                closest_idx = se.index.get_indexer([event_date], method='nearest')[0]
                actual_date = se.index[closest_idx]
                
                # Get SE value at that date
                se_val = se_norm.loc[actual_date]
                
                # Color based on event type
                if event['type'] == 'slow':
                    color = '#C0392B'  # Dark red
                elif event['type'] == 'flash':
                    color = '#F39C12'  # Orange
                elif event['type'] == 'exogenous':
                    color = '#9B59B6'  # Purple
                else:
                    color = '#27AE60'  # Green (recovery)
                
                ax2.annotate(event['name'], 
                            xy=(actual_date, min(se_val + 5, 100)),
                            xytext=(actual_date, 105),
                            fontsize=7,
                            ha='center',
                            color=color,
                            fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color=color, lw=0.5))
        except:
            pass
    
    # Title and styling
    ax1.set_title('35 Years of Structural Fragility: Stored Energy and Market Crises\n(1990-2024)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
    
    # Add interpretive note at bottom
    fig.text(0.5, 0.02, 
             '"Stored Energy rises during periods of compressed correlation (low entropy), often preceding major drawdowns.\n'
             'Red shading = slow-burn crises (Dot-Com, GFC); Yellow = rapid events (COVID, 2022)."',
             ha='center', fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save
    output_path = os.path.join(output_dir, "Figure_1_Historical_SE.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    return output_path

def main():
    print("=" * 70)
    print("CREATING THE ICONIC HISTORICAL FIGURE")
    print("=" * 70)
    
    px, market = load_data(LONG_HISTORY_UNIVERSE, START_DATE)
    if px.empty: 
        print("ERROR: No data")
        return
    
    # Calculate SE
    print("Calculating Stored Energy...")
    entropy = calculate_entropy(px, WINDOW)
    fragility = 1 - entropy
    stored_energy = fragility.rolling(ENERGY_WINDOW).sum()
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    sp500 = market['^GSPC']
    
    # Create the iconic figure
    output_path = create_iconic_figure(sp500, stored_energy, output_dir)
    
    print("\n=== ICONIC FIGURE COMPLETE ===")
    print(f"This is your Figure 1 for the paper.")

if __name__ == "__main__":
    main()
