import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

# --- Configuration ---
ASF_FILE = 'Table_Theory_Data.csv'
SPX_FILE = 'coodination_data/S&P_500.csv'
TLT_FILE = 'coodination_data/Treasuries_20Y.csv'
OUTPUT_FILE = 'Figure_Stock_Bond_Correlation.png'

def load_data():
    print("Loading data...")
    # Load ASF
    asf = pd.read_csv(ASF_FILE)
    asf.rename(columns={asf.columns[0]: 'Date'}, inplace=True)
    asf['Date'] = pd.to_datetime(asf['Date'])
    asf = asf.set_index('Date').sort_index()

    # Load Assets
    assets = {
        'SPX': (SPX_FILE, 'SPX'),
        'TLT': (TLT_FILE, 'TLT'),
    }
    
    df = asf[['ASF']].copy()
    
    for key, (path, col_name) in assets.items():
        if os.path.exists(path):
            temp = pd.read_csv(path)
            d_col = 'date' if 'date' in temp.columns else 'Date'
            temp[d_col] = pd.to_datetime(temp[d_col])
            temp = temp.set_index(d_col).sort_index()
            c_col = 'adjClose' if 'adjClose' in temp.columns else 'close'
            temp[col_name] = temp[c_col]
            
            # Join
            how_join = 'inner' if key == 'SPX' else 'left'
            df = df.join(temp[[col_name]], how=how_join)
        else:
            print(f"Warning: {path} not found")

    return df

def generate_impact_plot():
    df = load_data()
    
    # Calculate Correlation
    rets = df[['SPX', 'TLT']].pct_change()
    rolling_corr = rets['SPX'].rolling(window=63).corr(rets['TLT'])
    df['StockBond_Corr'] = rolling_corr
    
    # Forward Average Correlation (21 days)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    df['Fwd_Corr'] = df['StockBond_Corr'].rolling(window=indexer).mean()
    
    # Clean
    plot_data = df.dropna().copy()
    
    # --- IMPACT VISUALIZATION: Binning ---
    # Create 20 quantiles (bins) of ASF
    plot_data['ASF_Bin'] = pd.qcut(plot_data['ASF'], q=20, labels=False)
    
    # Calculate Mean and CI for each bin
    bin_stats = plot_data.groupby('ASF_Bin').agg({
        'ASF': 'mean',
        'Fwd_Corr': ['mean', 'sem', 'count']
    })
    bin_stats.columns = ['ASF_Mean', 'Corr_Mean', 'Corr_SEM', 'Count']
    bin_stats['CI95'] = 1.96 * bin_stats['Corr_SEM']
    
    # Setup Plot
    plt.figure(figsize=(10, 6))
    
    # 1. Background Shading for Regimes
    # Threshold approx 0.14
    plt.axvspan(plot_data['ASF'].min(), 0.14, color='tab:red', alpha=0.1, label='Fragility Regime (Contagion)')
    plt.axvspan(0.14, plot_data['ASF'].max(), color='tab:blue', alpha=0.1, label='Coordination Regime (Stable)')
    
    # 2. Main Visual: Binned Expected Correlation with Error Bars
    # This removes the "scatter cloud" noise.
    plt.errorbar(bin_stats['ASF_Mean'], bin_stats['Corr_Mean'], 
                 yerr=bin_stats['CI95'], fmt='-o', 
                 color='black', ecolor='gray', capsize=3, linewidth=2, markersize=6,
                 label='Conditional Expectation (Binned)')
                 
    # 3. Add a smoothed LOWESS trend for elegance
    lowess = sm.nonparametric.lowess(plot_data['Fwd_Corr'], plot_data['ASF'], frac=0.3)
    plt.plot(lowess[:, 0], lowess[:, 1], color='crimson', linewidth=3, alpha=0.8, linestyle='--', label='Structural Trend')

    # Formatting
    plt.axvline(x=0.14, color='black', linestyle=':', linewidth=1.5, label='Critical Threshold ($\\tau \\approx 0.14$)')
    plt.axhline(y=0, color='black', linewidth=0.5)
    
    plt.title('The Breakdown of Diversification\n(Structural Phase Transition)', fontsize=14, fontweight='bold')
    plt.xlabel('Accumulated Spectral Fragility (ASF)', fontsize=12)
    plt.ylabel('Forward Stock-Bond Correlation', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Annotations to make it "Impactful"
    # Arrow for Breakdown
    plt.annotate('BREAKDOWN\n(Diversification Fails)', 
                 xy=(0.08, 0.4), xytext=(0.05, 0.7),
                 arrowprops=dict(facecolor='crimson', shrink=0.05),
                 fontsize=10, color='crimson', fontweight='bold')

    # Arrow for Normal State
    plt.annotate('NORMAL STATE\n(Hedge Works)', 
                 xy=(0.25, -0.4), xytext=(0.20, -0.1),
                 arrowprops=dict(facecolor='tab:blue', shrink=0.05),
                 fontsize=10, color='tab:blue', fontweight='bold')
                 
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Plot saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_impact_plot()
