"""
Minsky Signal Visualization: When Does Silent Fragility Appear?
===============================================================

Creates a chart showing:
1. S&P 500 Price (log scale)
2. Shaded regions when Minsky Signal is ACTIVE (Silent Fragility)
3. Entropy level as secondary axis (inverted: Low Entropy = High Fragility)

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

# Long-History Universe
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
ROLL_RANK = 252 * 2

def load_data(tickers, start_date):
    print(f"Loading data...")
    try:
        df = yf.download(tickers, start=start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            px = df['Adj Close' if 'Adj Close' in df else 'Close']
        else:
            px = df
    except Exception as e:
        print(f"Error: {e}")
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

    print("Calculating features...")
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
        'Vol': pd.Series(vol_series),
    })
    return df.dropna()

def create_minsky_visualization(market, features, output_dir):
    """Create the main visualization."""
    
    common = market.index.intersection(features.index)
    sp500 = market['^GSPC'].loc[common]
    feats = features.loc[common]
    
    # Calculate Minsky Signal
    ent_rank = feats['Entropy'].rolling(ROLL_RANK, min_periods=252).rank(pct=True)
    vol_rank = feats['Vol'].rolling(ROLL_RANK, min_periods=252).rank(pct=True)
    
    # Signal: Low Entropy (<20%) AND Low Vol (<50%)
    is_fragile = ent_rank < 0.20
    is_complacent = vol_rank < 0.50
    minsky_signal = (is_fragile & is_complacent).astype(int)
    
    # Fragility = 1 - Entropy (for visualization: High = Dangerous)
    fragility = 1 - feats['Entropy']
    
    # Create the figure
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # --- S&P 500 Price ---
    ax1.semilogy(sp500.index, sp500.values, color='black', linewidth=1.5, label='S&P 500')
    ax1.set_ylabel('S&P 500 (Log Scale)', fontsize=12)
    ax1.set_xlabel('')
    
    # --- Shade Minsky Signal Periods ---
    # Find contiguous Signal=1 periods
    signal_diff = minsky_signal.diff().fillna(0)
    starts = minsky_signal.index[signal_diff == 1]
    ends = minsky_signal.index[signal_diff == -1]
    
    # Handle edge cases
    if len(ends) == 0 or (len(starts) > 0 and (len(ends) == 0 or starts[0] < ends[0])):
        # Signal is on at start
        pass
    if len(starts) > 0 and len(ends) > 0:
        if starts[-1] > ends[-1]:
            ends = ends.append(pd.Index([minsky_signal.index[-1]]))
    
    # Pair starts and ends
    for i, start in enumerate(starts):
        if i < len(ends):
            end = ends[i]
            ax1.axvspan(start, end, alpha=0.3, color='red', label='_nolegend_')
    
    # Add one legend entry for shaded regions
    ax1.axvspan(pd.Timestamp('1900-01-01'), pd.Timestamp('1900-01-02'), 
                alpha=0.3, color='red', label='Minsky Warning (Silent Fragility)')
    
    # --- Mark Major Crises ---
    crises = {
        'Dot-Com Peak': ('2000-03-24', 'red'),
        'GFC Peak': ('2007-10-09', 'darkred'),
        'COVID': ('2020-02-19', 'purple'),
    }
    for name, (date, color) in crises.items():
        try:
            ax1.axvline(pd.Timestamp(date), color=color, linestyle='--', alpha=0.7, linewidth=1)
            # Annotate
            y_pos = sp500.loc[:date].iloc[-1] if date in sp500.index else sp500.max()
            ax1.annotate(name, xy=(pd.Timestamp(date), y_pos), 
                        xytext=(10, 20), textcoords='offset points',
                        fontsize=9, color=color, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=color, alpha=0.5))
        except:
            pass
    
    # --- Fragility on Secondary Axis ---
    ax2 = ax1.twinx()
    ax2.plot(fragility.index, fragility.values, color='green', alpha=0.5, linewidth=0.8, label='Fragility (1-Entropy)')
    ax2.set_ylabel('Fragility (1 - Entropy)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 0.5)  # Normalize view
    
    # Add threshold line for "High Fragility"
    ax2.axhline(fragility.quantile(0.80), color='green', linestyle=':', alpha=0.5, label='Fragility 80th Percentile')
    
    # --- Legend & Title ---
    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.title("When Does Silent Fragility Appear?\nS&P 500 with Minsky Warnings (1990-2024)", fontsize=14, fontweight='bold')
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "Minsky_Signal_Timeline.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path

def main():
    print("=== MINSKY SIGNAL VISUALIZATION ===")
    
    px, market = load_data(LONG_HISTORY_UNIVERSE, START_DATE)
    
    if px.empty or market.empty: 
        print("Data load failed.")
        return
    
    feats = calculate_features(px, WINDOW)
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    create_minsky_visualization(market, feats, output_dir)
    
    print("=== VISUALIZATION COMPLETE ===")

if __name__ == "__main__":
    main()
