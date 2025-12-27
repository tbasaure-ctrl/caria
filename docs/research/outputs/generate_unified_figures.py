"""
Generate all figures and tables for the unified manuscript.

This script produces:
1. Figure I: Phase diagram showing risk surface
2. Figure II: Which indicators detect the transition
3. Figure III: Stock-bond correlation breakdown
4. Table export for LaTeX
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os

# Configuration
ASF_FILE = 'Table_Theory_Data.csv'
VIX_FILE = 'coodination_data/CBOE_Volatility_Index.csv'
SPX_FILE = 'coodination_data/S&P_500.csv'
TLT_FILE = 'coodination_data/Treasuries_20Y.csv'
GLD_FILE = 'coodination_data/Gold.csv'

THRESHOLD = 0.14
plt.style.use('seaborn-v0_8-whitegrid')

def load_data():
    """Load and merge all data sources."""
    asf = pd.read_csv(ASF_FILE)
    asf.rename(columns={asf.columns[0]: 'Date'}, inplace=True)
    asf['Date'] = pd.to_datetime(asf['Date'])
    asf = asf.set_index('Date').sort_index()
    
    df = asf[['ASF']].copy()
    
    files = {
        'VIX': (VIX_FILE, True),
        'SPX': (SPX_FILE, False),
        'TLT': (TLT_FILE, False),
        'GLD': (GLD_FILE, False),
    }
    
    for name, (path, scale) in files.items():
        if os.path.exists(path):
            temp = pd.read_csv(path)
            d_col = 'date' if 'date' in temp.columns else 'Date'
            temp[d_col] = pd.to_datetime(temp[d_col])
            temp = temp.set_index(d_col).sort_index()
            c_col = 'adjClose' if 'adjClose' in temp.columns else 'close'
            if scale:
                temp[name] = temp[c_col] / 100.0
            else:
                temp[name] = temp[c_col]
            how = 'inner' if name in ['SPX', 'VIX'] else 'left'
            df = df.join(temp[[name]], how=how)
    
    return df

def calculate_forward_drawdown(series, window=21):
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    rolling_min = series.rolling(window=indexer).min()
    fwd_worst_return = (rolling_min / series) - 1
    return -fwd_worst_return

def compute_absorption_ratio(df, assets, window=63):
    """Compute rolling absorption ratio."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    rets = df[assets].pct_change()
    ar_values = []
    
    for i in range(len(rets)):
        if i < window:
            ar_values.append(np.nan)
        else:
            window_data = rets.iloc[i-window:i][assets].dropna()
            if len(window_data) < window * 0.8:
                ar_values.append(np.nan)
                continue
            try:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(window_data)
                pca = PCA(n_components=1)
                pca.fit(scaled)
                ar_values.append(pca.explained_variance_ratio_[0])
            except:
                ar_values.append(np.nan)
    
    return pd.Series(ar_values, index=df.index)

def compute_mean_correlation(df, assets, window=63):
    """Compute rolling mean pairwise correlation."""
    rets = df[assets].pct_change()
    rolling_corrs = []
    
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            col1, col2 = assets[i], assets[j]
            rc = rets[col1].rolling(window).corr(rets[col2])
            rolling_corrs.append(rc)
    
    return pd.concat(rolling_corrs, axis=1).mean(axis=1)

def generate_figure_1_phase_diagram():
    """Generate phase diagram showing risk surface with regime boundary."""
    print("Generating Figure I: Phase Diagram...")
    
    df = load_data()
    df['Fwd_MaxDD'] = calculate_forward_drawdown(df['SPX'], window=21)
    
    # Compute mean correlation as connectivity proxy
    assets = [c for c in ['SPX', 'TLT', 'GLD'] if c in df.columns]
    if len(assets) >= 2:
        df['Connectivity'] = compute_mean_correlation(df, assets)
    else:
        df['Connectivity'] = df['ASF'] * 0.5  # Fallback
    
    plot_df = df.dropna(subset=['ASF', 'Connectivity', 'Fwd_MaxDD'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create 2D histogram / contour
    from scipy.interpolate import griddata
    
    x = plot_df['ASF'].values
    y = plot_df['Connectivity'].values
    z = plot_df['Fwd_MaxDD'].values
    
    # Create grid
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    
    # Plot contour
    cmap = LinearSegmentedColormap.from_list('risk', ['#2166AC', '#FFFFFF', '#B2182B'])
    contour = ax.contourf(Xi, Yi, Zi, levels=20, cmap=cmap, alpha=0.8)
    plt.colorbar(contour, ax=ax, label='Forward Maximum Drawdown')
    
    # Add threshold line
    ax.axhline(y=THRESHOLD, color='black', linestyle='--', linewidth=2, 
               label=f'Threshold ($\\tau = {THRESHOLD}$)')
    
    # Add regime labels
    ax.text(0.75, 0.05, 'Low Connectivity\n(Contagion)', transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.75, 0.85, 'High Connectivity\n(Coordination)', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Accumulated Spectral Fragility (ASF)', fontsize=12)
    ax.set_ylabel('Mean Correlation (Connectivity)', fontsize=12)
    ax.set_title('Regime-Dependent Risk Surface', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('Figure_I_Phase_Diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure_I_Phase_Diagram.pdf', bbox_inches='tight')
    print("  Saved: Figure_I_Phase_Diagram.png/pdf")
    plt.close()

def generate_figure_2_indicator_comparison():
    """Generate comparison of which indicators detect the transition."""
    print("Generating Figure II: Indicator Comparison...")
    
    df = load_data()
    df['Fwd_MaxDD'] = calculate_forward_drawdown(df['SPX'], window=21)
    
    # Compute additional indicators
    assets = [c for c in ['SPX', 'TLT', 'GLD'] if c in df.columns]
    if len(assets) >= 2:
        df['Mean_Corr'] = compute_mean_correlation(df, assets)
        df['Absorption_Ratio'] = compute_absorption_ratio(df, assets)
    
    df = df.dropna(subset=['ASF', 'Fwd_MaxDD'])
    df['Regime'] = np.where(df['ASF'] > THRESHOLD, 'High', 'Low')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    indicators = {
        'ASF': ('Eigenstructure', 'tab:blue'),
        'Absorption_Ratio': ('Eigenstructure', 'tab:cyan'),
        'VIX': ('Volatility', 'tab:orange'),
        'Mean_Corr': ('Network', 'tab:green')
    }
    
    for idx, (indicator, (ind_type, color)) in enumerate(indicators.items()):
        ax = axes[idx // 2, idx % 2]
        
        if indicator not in df.columns:
            ax.text(0.5, 0.5, f'{indicator}\nData not available', 
                    transform=ax.transAxes, ha='center', va='center')
            continue
        
        sub_df = df.dropna(subset=[indicator, 'Fwd_MaxDD'])
        
        # Bin the indicator and compute mean drawdown by regime
        for regime, regime_color, offset in [('Low', 'red', -0.02), ('High', 'blue', 0.02)]:
            regime_df = sub_df[sub_df['Regime'] == regime]
            if len(regime_df) > 50:
                regime_df = regime_df.copy()
                regime_df['Bin'] = pd.qcut(regime_df[indicator], q=10, labels=False, duplicates='drop')
                bin_means = regime_df.groupby('Bin').agg({
                    indicator: 'mean',
                    'Fwd_MaxDD': ['mean', 'sem']
                })
                bin_means.columns = ['x', 'y', 'se']
                
                ax.errorbar(bin_means['x'], bin_means['y'], yerr=1.96*bin_means['se'],
                           fmt='-o', color=regime_color, capsize=3, alpha=0.7,
                           label=f'{regime} Connectivity')
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel(indicator, fontsize=11)
        ax.set_ylabel('Forward Max Drawdown', fontsize=11)
        ax.set_title(f'{indicator} ({ind_type})', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
    
    plt.suptitle('Which Indicators Detect the Regime Transition?', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('Figure_II_Indicator_Comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure_II_Indicator_Comparison.pdf', bbox_inches='tight')
    print("  Saved: Figure_II_Indicator_Comparison.png/pdf")
    plt.close()

def generate_figure_3_stock_bond():
    """Generate stock-bond correlation breakdown figure."""
    print("Generating Figure III: Stock-Bond Breakdown...")
    
    df = load_data()
    
    # Calculate rolling correlation
    if 'TLT' in df.columns:
        rets = df[['SPX', 'TLT']].pct_change()
        df['StockBond_Corr'] = rets['SPX'].rolling(window=63).corr(rets['TLT'])
        
        # Forward correlation
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
        df['Fwd_Corr'] = df['StockBond_Corr'].rolling(window=indexer).mean()
    
    plot_df = df.dropna(subset=['ASF', 'Fwd_Corr'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Binned relationship
    ax1 = axes[0]
    plot_df['ASF_Bin'] = pd.qcut(plot_df['ASF'], q=20, labels=False, duplicates='drop')
    bin_stats = plot_df.groupby('ASF_Bin').agg({
        'ASF': 'mean',
        'Fwd_Corr': ['mean', 'sem']
    })
    bin_stats.columns = ['x', 'y', 'se']
    
    ax1.fill_between(bin_stats['x'], 
                     bin_stats['y'] - 1.96*bin_stats['se'],
                     bin_stats['y'] + 1.96*bin_stats['se'],
                     alpha=0.3, color='crimson')
    ax1.plot(bin_stats['x'], bin_stats['y'], 'o-', color='crimson', linewidth=2)
    ax1.axvline(x=THRESHOLD, color='black', linestyle='--', label=f'Threshold ({THRESHOLD})')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    ax1.set_xlabel('Accumulated Spectral Fragility (ASF)', fontsize=12)
    ax1.set_ylabel('Forward Stock-Bond Correlation', fontsize=12)
    ax1.set_title('The Breakdown of Diversification', fontsize=13, fontweight='bold')
    ax1.legend()
    
    # Right: Distribution by regime
    ax2 = axes[1]
    low_asf = plot_df[plot_df['ASF'] < THRESHOLD]['Fwd_Corr']
    high_asf = plot_df[plot_df['ASF'] >= THRESHOLD]['Fwd_Corr']
    
    ax2.hist(low_asf, bins=30, alpha=0.6, color='red', label=f'Low ASF (N={len(low_asf)})', density=True)
    ax2.hist(high_asf, bins=30, alpha=0.6, color='blue', label=f'High ASF (N={len(high_asf)})', density=True)
    ax2.axvline(x=low_asf.mean(), color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=high_asf.mean(), color='blue', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Stock-Bond Correlation', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Correlation Distribution by Structural State', fontsize=13, fontweight='bold')
    ax2.legend()
    
    # Add annotation
    ax2.annotate(f'Mean (Low ASF): {low_asf.mean():.2f}', 
                 xy=(low_asf.mean(), 0.5), xytext=(low_asf.mean() + 0.2, 1.5),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')
    ax2.annotate(f'Mean (High ASF): {high_asf.mean():.2f}', 
                 xy=(high_asf.mean(), 0.5), xytext=(high_asf.mean() - 0.4, 2),
                 arrowprops=dict(arrowstyle='->', color='blue'),
                 fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.savefig('Figure_III_Stock_Bond.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure_III_Stock_Bond.pdf', bbox_inches='tight')
    print("  Saved: Figure_III_Stock_Bond.png/pdf")
    plt.close()

def generate_figure_4_timeline():
    """Generate timeline showing ASF, VIX, and regime transitions."""
    print("Generating Figure IV: Historical Timeline...")
    
    df = load_data()
    df['Fwd_MaxDD'] = calculate_forward_drawdown(df['SPX'], window=21)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Panel A: ASF
    ax1 = axes[0]
    ax1.plot(df.index, df['ASF'], color='tab:blue', linewidth=1)
    ax1.axhline(y=THRESHOLD, color='red', linestyle='--', alpha=0.7)
    ax1.fill_between(df.index, 0, 1, where=df['ASF'] > THRESHOLD, 
                     alpha=0.2, color='blue', label='High ASF')
    ax1.set_ylabel('ASF', fontsize=11)
    ax1.set_title('A. Accumulated Spectral Fragility', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Panel B: VIX
    ax2 = axes[1]
    ax2.plot(df.index, df['VIX']*100, color='tab:orange', linewidth=1)
    ax2.set_ylabel('VIX (%)', fontsize=11)
    ax2.set_title('B. Volatility Index', fontsize=12, fontweight='bold')
    
    # Panel C: Drawdowns
    ax3 = axes[2]
    ax3.fill_between(df.index, 0, df['Fwd_MaxDD']*100, color='tab:red', alpha=0.5)
    ax3.set_ylabel('Drawdown (%)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('C. Forward Maximum Drawdown', fontsize=12, fontweight='bold')
    
    # Add crisis annotations
    crises = {
        '2008-09': 'GFC',
        '2011-08': 'Euro Crisis',
        '2015-08': 'China',
        '2020-03': 'COVID',
        '2022-06': 'Inflation'
    }
    
    for date_str, label in crises.items():
        try:
            date = pd.to_datetime(date_str)
            for ax in axes:
                ax.axvline(x=date, color='gray', linestyle=':', alpha=0.5)
            ax3.annotate(label, xy=(date, ax3.get_ylim()[1]*0.8), fontsize=8, rotation=90)
        except:
            pass
    
    plt.tight_layout()
    plt.savefig('Figure_IV_Timeline.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure_IV_Timeline.pdf', bbox_inches='tight')
    print("  Saved: Figure_IV_Timeline.png/pdf")
    plt.close()

def main():
    print("\n" + "="*60)
    print("Generating Figures for Unified Manuscript")
    print("="*60 + "\n")
    
    generate_figure_1_phase_diagram()
    generate_figure_2_indicator_comparison()
    generate_figure_3_stock_bond()
    generate_figure_4_timeline()
    
    print("\n" + "="*60)
    print("All figures generated successfully")
    print("="*60)

if __name__ == "__main__":
    main()
