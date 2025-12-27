"""
PATH 2 EMPIRICAL TEST: The Infrastructure of Correlation

The core claim: Intermediary leverage, network density, and factor concentration
are not separate phenomena—they are three projections of the same underlying object:
market dimensionality.

We test:
1. Cross-correlation between proxies (ASF, VIX, Absorption Ratio, Mean Correlation)
2. They all produce the SAME regime transition
3. Principal component analysis shows they load on one factor
4. The phase transition is universal across measures
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Configuration
ASF_FILE = 'Table_Theory_Data.csv'
VIX_FILE = 'coodination_data/CBOE_Volatility_Index.csv'
SPX_FILE = 'coodination_data/S&P_500.csv'
TLT_FILE = 'coodination_data/Treasuries_20Y.csv'
GLD_FILE = 'coodination_data/Gold.csv'
EUR_FILE = 'coodination_data/Euro_USD.csv'
HYG_FILE = 'coodination_data/HYG.csv'
LQD_FILE = 'coodination_data/LQD.csv'

THRESHOLD = 0.14

def load_data():
    """Load and merge all data sources."""
    asf = pd.read_csv(ASF_FILE)
    asf.rename(columns={asf.columns[0]: 'Date'}, inplace=True)
    asf['Date'] = pd.to_datetime(asf['Date'])
    asf = asf.set_index('Date').sort_index()
    
    df = asf[['ASF']].copy()
    
    # Load additional columns from ASF file if available
    for col in ['Connectivity', 'Entropy', 'Fragility']:
        if col in asf.columns:
            df[col] = asf[col]
    
    files = {
        'VIX': (VIX_FILE, True),
        'SPX': (SPX_FILE, False),
        'TLT': (TLT_FILE, False),
        'GLD': (GLD_FILE, False),
        'EUR': (EUR_FILE, False),
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
    
    # Load credit spreads if available
    if os.path.exists(HYG_FILE) and os.path.exists(LQD_FILE):
        try:
            hyg = pd.read_csv(HYG_FILE, parse_dates=['date']).set_index('date').sort_index()
            lqd = pd.read_csv(LQD_FILE, parse_dates=['date']).set_index('date').sort_index()
            df = df.join(hyg[['HYG']], how='left')
            df = df.join(lqd[['LQD']], how='left')
        except:
            pass
    
    return df

def construct_alternative_proxies(df):
    """
    Construct alternative proxies for the structural state variable:
    1. Mean Correlation (network density proxy)
    2. Absorption Ratio (factor concentration proxy)
    3. Credit Spread (intermediary stress proxy)
    4. VIX (market risk proxy)
    """
    print("\n" + "-"*60)
    print("Constructing Alternative Structural Proxies")
    print("-"*60)
    
    # Get available asset columns
    assets = [col for col in ['SPX', 'TLT', 'GLD', 'EUR'] if col in df.columns]
    
    if len(assets) >= 2:
        # Calculate returns
        rets = df[assets].pct_change()
        
        # 1. Mean Pairwise Correlation (Rolling 63-day)
        rolling_corrs = []
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                col1, col2 = assets[i], assets[j]
                rc = rets[col1].rolling(63).corr(rets[col2])
                rolling_corrs.append(rc)
        df['Mean_Corr'] = pd.concat(rolling_corrs, axis=1).mean(axis=1)
        print(f"  ✓ Mean Correlation (Network Density proxy)")
        
        # 2. Absorption Ratio (variance explained by first PC)
        def calc_absorption_ratio(window_data, n_components=1):
            if len(window_data.dropna()) < 30:
                return np.nan
            clean = window_data.dropna()
            if len(clean) < 30:
                return np.nan
            try:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(clean)
                pca = PCA(n_components=min(n_components, len(assets)))
                pca.fit(scaled)
                return pca.explained_variance_ratio_[0]
            except:
                return np.nan
        
        # Rolling Absorption Ratio (slower but accurate)
        ar_values = []
        for i in range(len(rets)):
            if i < 63:
                ar_values.append(np.nan)
            else:
                window = rets.iloc[i-63:i][assets]
                ar_values.append(calc_absorption_ratio(window))
        df['Absorption_Ratio'] = ar_values
        print(f"  ✓ Absorption Ratio (Factor Concentration proxy)")
    
    # 3. Credit Spread (if available)
    if 'HYG' in df.columns and 'LQD' in df.columns:
        df['Credit_Spread'] = (df['LQD'] / df['HYG']) - 1  # Relative spread
        print(f"  ✓ Credit Spread (Intermediary Stress proxy)")
    
    # 4. VIX is already loaded as volatility proxy
    if 'VIX' in df.columns:
        print(f"  ✓ VIX (Volatility/Fear proxy)")
    
    return df

def calculate_forward_drawdown(series, window=21):
    """Calculate forward maximum drawdown."""
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    rolling_min = series.rolling(window=indexer).min()
    fwd_worst_return = (rolling_min / series) - 1
    return -fwd_worst_return

def test_infrastructure_of_correlation():
    """
    PATH 2 CORE TEST: Show that different proxies are measuring the same 
    underlying structural state, and they all produce the same regime transition.
    """
    print("\n" + "="*80)
    print("PATH 2: THE INFRASTRUCTURE OF CORRELATION")
    print("="*80)
    
    df = load_data()
    df = construct_alternative_proxies(df)
    df['Fwd_MaxDD'] = calculate_forward_drawdown(df['SPX'], window=21)
    
    print(f"\nSample: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total observations: {len(df)}")
    
    results = []
    
    # ============================================
    # TEST 1: Cross-Correlations Between Proxies
    # ============================================
    print("\n" + "-"*60)
    print("TEST 1: Cross-Correlations Between Structural Proxies")
    print("-"*60)
    print("\nIf these are all measuring the same thing, correlations should be HIGH.\n")
    
    proxy_cols = ['ASF', 'Mean_Corr', 'Absorption_Ratio', 'VIX']
    proxy_cols = [c for c in proxy_cols if c in df.columns and df[c].notna().sum() > 100]
    
    if len(proxy_cols) >= 2:
        corr_matrix = df[proxy_cols].corr()
        print("Correlation Matrix:")
        print(corr_matrix.round(3).to_string())
        
        # Extract pairwise correlations
        print("\nKey Findings:")
        for i in range(len(proxy_cols)):
            for j in range(i+1, len(proxy_cols)):
                corr = corr_matrix.iloc[i, j]
                result = {
                    'Test': 'Proxy Correlation',
                    'Pair': f"{proxy_cols[i]} vs {proxy_cols[j]}",
                    'Correlation': corr
                }
                results.append(result)
                strength = "STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
                print(f"  {proxy_cols[i]} ↔ {proxy_cols[j]}: {corr:.3f} ({strength})")
    
    # ============================================
    # TEST 2: PCA - Do They Load on One Factor?
    # ============================================
    print("\n" + "-"*60)
    print("TEST 2: Principal Component Analysis - One Latent Dimension?")
    print("-"*60)
    
    pca_cols = [c for c in proxy_cols if c in df.columns]
    pca_data = df[pca_cols].dropna()
    
    if len(pca_data) > 100 and len(pca_cols) >= 3:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(pca_data)
        
        pca = PCA()
        pca.fit(scaled)
        
        print(f"\nVariance Explained by Principal Components:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            cum_var = sum(pca.explained_variance_ratio_[:i+1])
            print(f"  PC{i+1}: {100*var:.1f}% (Cumulative: {100*cum_var:.1f}%)")
        
        pc1_var = pca.explained_variance_ratio_[0]
        results.append({
            'Test': 'PCA',
            'Variance_PC1': pc1_var,
            'Interpretation': 'ONE FACTOR' if pc1_var > 0.5 else 'MULTIPLE FACTORS'
        })
        
        print(f"\n  PC1 Loadings:")
        for col, loading in zip(pca_cols, pca.components_[0]):
            direction = "↑" if loading > 0 else "↓"
            print(f"    {col}: {loading:.3f} {direction}")
        
        if pc1_var > 0.5:
            print(f"\n  ✓ CONFIRMED: {100*pc1_var:.1f}% of variance is ONE LATENT FACTOR")
        else:
            print(f"\n  ✗ Multiple factors present - proxies may capture different dimensions")
    
    # ============================================
    # TEST 3: Regime Transition Universality
    # ============================================
    print("\n" + "-"*60)
    print("TEST 3: Do ALL Proxies Show the Same Regime Transition?")
    print("-"*60)
    
    # For each proxy, run threshold regression and check for sign inversion
    test_df = df.dropna(subset=['ASF', 'SPX', 'Fwd_MaxDD'])
    test_df['Regime'] = np.where(test_df['ASF'] > THRESHOLD, 'Coordination', 'Contagion')
    
    print(f"\nRegime Definition: ASF > {THRESHOLD} (Coordination) vs ASF <= {THRESHOLD} (Contagion)")
    print(f"\n{'Proxy':<20} {'Contagion β':>15} {'Coordination β':>18} {'Sign Inversion?':>18}")
    print("-"*75)
    
    test_proxies = ['ASF', 'Mean_Corr', 'Absorption_Ratio', 'VIX']
    test_proxies = [p for p in test_proxies if p in test_df.columns and test_df[p].notna().sum() > 50]
    
    for proxy in test_proxies:
        sub_df = test_df.dropna(subset=[proxy, 'Fwd_MaxDD'])
        
        # Contagion regime
        contagion = sub_df[sub_df['Regime'] == 'Contagion']
        if len(contagion) > 30:
            X = sm.add_constant(contagion[proxy])
            y = contagion['Fwd_MaxDD']
            try:
                model_c = sm.OLS(y, X).fit()
                beta_c = model_c.params[proxy]
            except:
                beta_c = np.nan
        else:
            beta_c = np.nan
        
        # Coordination regime
        coord = sub_df[sub_df['Regime'] == 'Coordination']
        if len(coord) > 30:
            X = sm.add_constant(coord[proxy])
            y = coord['Fwd_MaxDD']
            try:
                model_h = sm.OLS(y, X).fit()
                beta_h = model_h.params[proxy]
            except:
                beta_h = np.nan
        else:
            beta_h = np.nan
        
        # Check for sign inversion
        if not np.isnan(beta_c) and not np.isnan(beta_h):
            inverts = (beta_c > 0 and beta_h < 0) or (beta_c < 0 and beta_h > 0)
            invert_str = "✓ YES" if inverts else "✗ NO"
        else:
            invert_str = "N/A"
        
        print(f"{proxy:<20} {beta_c:>15.4f} {beta_h:>18.4f} {invert_str:>18}")
        
        results.append({
            'Test': 'Regime Universality',
            'Proxy': proxy,
            'Beta_Contagion': beta_c,
            'Beta_Coordination': beta_h,
            'Sign_Inverts': invert_str
        })
    
    # ============================================
    # TEST 4: Granger Causality - What Leads What?
    # ============================================
    print("\n" + "-"*60)
    print("TEST 4: Temporal Relationships (Lead-Lag)")
    print("-"*60)
    
    from statsmodels.tsa.stattools import grangercausalitytests
    
    test_pairs = [('ASF', 'VIX'), ('ASF', 'Mean_Corr'), ('VIX', 'Mean_Corr')]
    test_pairs = [(a, b) for a, b in test_pairs if a in test_df.columns and b in test_df.columns]
    
    print("\nGranger Causality Tests (max lag = 5):")
    for x, y in test_pairs:
        pair_df = test_df[[x, y]].dropna()
        if len(pair_df) > 100:
            try:
                # Test if X Granger-causes Y
                test_result = grangercausalitytests(pair_df[[y, x]], maxlag=5, verbose=False)
                # Get minimum p-value across lags
                p_vals = [test_result[i+1][0]['ssr_ftest'][1] for i in range(5)]
                min_p = min(p_vals)
                best_lag = p_vals.index(min_p) + 1
                
                if min_p < 0.05:
                    print(f"  {x} → {y}: p={min_p:.4f} at lag {best_lag} (SIGNIFICANT)")
                else:
                    print(f"  {x} → {y}: p={min_p:.4f} (not significant)")
            except:
                pass
    
    # ============================================
    # SYNTHESIS
    # ============================================
    print("\n" + "="*60)
    print("SYNTHESIS: THE INFRASTRUCTURE OF CORRELATION")
    print("="*60)
    
    print("""
    KEY FINDINGS:
    
    1. CROSS-CORRELATIONS: If high (>0.5), these proxies measure the same thing.
    
    2. PCA: If PC1 explains >50% variance, there is ONE latent structural dimension.
    
    3. REGIME UNIVERSALITY: If all proxies show sign inversion at the same threshold,
       the phase transition is REAL, not an artifact of measurement.
    
    4. TEMPORAL STRUCTURE: Granger causality reveals which is the "deepest" measure.
    
    INTERPRETATION:
    ─────────────────
    • Intermediary Leverage → shows up in Credit Spreads
    • Network Density → shows up in Mean Correlation
    • Factor Concentration → shows up in Absorption Ratio
    • Structural State → shows up in ASF (Spectral Entropy)
    
    If these all move together, they are DIFFERENT VIEWS OF ONE OBJECT.
    The market has ONE structural state, and we've been measuring it many ways.
    """)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('Path2_Infrastructure_Results.csv', index=False)
    
    # ============================================
    # CREATE VISUALIZATION
    # ============================================
    print("\n" + "-"*60)
    print("Generating Visualization...")
    print("-"*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Correlation Heatmap
    ax1 = axes[0, 0]
    if len(proxy_cols) >= 2:
        corr_matrix = df[proxy_cols].corr()
        im = ax1.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_xticks(range(len(proxy_cols)))
        ax1.set_yticks(range(len(proxy_cols)))
        ax1.set_xticklabels(proxy_cols, rotation=45, ha='right')
        ax1.set_yticklabels(proxy_cols)
        for i in range(len(proxy_cols)):
            for j in range(len(proxy_cols)):
                ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center')
        ax1.set_title('Correlation Between Structural Proxies')
        plt.colorbar(im, ax=ax1)
    
    # Plot 2: PCA Variance Explained
    ax2 = axes[0, 1]
    if len(pca_data) > 100:
        var_explained = pca.explained_variance_ratio_
        ax2.bar(range(1, len(var_explained)+1), var_explained, color='steelblue')
        ax2.axhline(y=0.5, color='red', linestyle='--', label='50% threshold')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Variance Explained')
        ax2.set_title('PCA: One Latent Dimension?')
        ax2.legend()
    
    # Plot 3: Proxies Over Time
    ax3 = axes[1, 0]
    plot_cols = [c for c in ['ASF', 'Mean_Corr', 'Absorption_Ratio'] if c in df.columns]
    plot_df = df[plot_cols].dropna()
    if len(plot_df) > 100:
        # Normalize for comparison
        normalized = (plot_df - plot_df.mean()) / plot_df.std()
        for col in plot_cols:
            ax3.plot(normalized.index, normalized[col], label=col, alpha=0.7)
        ax3.legend()
        ax3.set_title('Structural Proxies (Normalized)')
        ax3.set_ylabel('Z-Score')
    
    # Plot 4: Regime Coefficients Comparison
    ax4 = axes[1, 1]
    regime_results = [r for r in results if r.get('Test') == 'Regime Universality']
    if regime_results:
        proxies = [r['Proxy'] for r in regime_results]
        betas_c = [r['Beta_Contagion'] for r in regime_results]
        betas_h = [r['Beta_Coordination'] for r in regime_results]
        x = range(len(proxies))
        width = 0.35
        ax4.bar([i - width/2 for i in x], betas_c, width, label='Contagion', color='red', alpha=0.7)
        ax4.bar([i + width/2 for i in x], betas_h, width, label='Coordination', color='blue', alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels(proxies, rotation=45, ha='right')
        ax4.axhline(y=0, color='black', linewidth=0.5)
        ax4.set_ylabel('Beta (Predicting Drawdowns)')
        ax4.set_title('Sign Inversion Across Proxies')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('Path2_Infrastructure.png', dpi=150)
    print("  Saved: Path2_Infrastructure.png")
    
    return df_results

if __name__ == "__main__":
    results = test_infrastructure_of_correlation()
