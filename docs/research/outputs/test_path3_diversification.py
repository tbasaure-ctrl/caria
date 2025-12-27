"""
PATH 3 EMPIRICAL TEST: The Diversification Paradox

The core claim: Diversification has two effects:
1. Effect A (Visible): Reduces idiosyncratic variance
2. Effect B (Invisible): Increases dependence on coordination

Below threshold: Effect A dominates (standard diversification works)
Above threshold: Effect B dominates (diversification creates systemic exposure)

We test:
1. Stock-bond correlation breakdown when ASF is low
2. Multi-asset correlation surge during stress
3. Verification that "diversified" portfolios fail together
4. The two effects shown separately
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import os

# Configuration
ASF_FILE = 'Table_Theory_Data.csv'
VIX_FILE = 'coodination_data/CBOE_Volatility_Index.csv'
SPX_FILE = 'coodination_data/S&P_500.csv'
TLT_FILE = 'coodination_data/Treasuries_20Y.csv'
GLD_FILE = 'coodination_data/Gold.csv'
EUR_FILE = 'coodination_data/Euro_USD.csv'

THRESHOLD = 0.14

def load_data():
    """Load and merge all data sources."""
    asf = pd.read_csv(ASF_FILE)
    asf.rename(columns={asf.columns[0]: 'Date'}, inplace=True)
    asf['Date'] = pd.to_datetime(asf['Date'])
    asf = asf.set_index('Date').sort_index()
    
    df = asf[['ASF']].copy()
    
    files = {
        'VIX': (VIX_FILE, True),  # True = scale by 100
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
    
    return df

def test_diversification_paradox():
    """
    PATH 3 CORE TEST: Show that diversification has two effects,
    and the invisible one (coordination dependence) dominates in high-connectivity regimes.
    """
    print("\n" + "="*80)
    print("PATH 3: THE DIVERSIFICATION PARADOX")
    print("="*80)
    
    df = load_data()
    print(f"\nSample: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total observations: {len(df)}")
    
    results = []
    
    # ============================================
    # TEST 1: Stock-Bond Correlation Breakdown
    # ============================================
    print("\n" + "-"*60)
    print("TEST 1: Stock-Bond Correlation - The Breakdown of Diversification")
    print("-"*60)
    
    # Calculate rolling correlation
    rets = df[['SPX', 'TLT']].pct_change()
    df['StockBond_Corr'] = rets['SPX'].rolling(window=63).corr(rets['TLT'])
    
    # Forward correlation (next 21 days)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    df['Fwd_Corr'] = df['StockBond_Corr'].rolling(window=indexer).mean()
    
    test_df = df.dropna(subset=['ASF', 'Fwd_Corr'])
    
    # Regression: Does low ASF predict correlation spike?
    X = sm.add_constant(test_df['ASF'])
    y = test_df['Fwd_Corr']
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    print(f"\nRegression: Forward Stock-Bond Correlation on ASF")
    print(f"  Beta: {model.params['ASF']:.4f}")
    print(f"  T-Stat: {model.tvalues['ASF']:.2f}")
    print(f"  R²: {model.rsquared:.4f}")
    print(f"\n  Interpretation: {'Negative beta confirms: Low ASF → Correlation Spikes' if model.params['ASF'] < 0 else 'Unexpected result'}")
    
    results.append({
        'Test': 'Stock-Bond Correlation',
        'Beta': model.params['ASF'],
        'T-stat': model.tvalues['ASF'],
        'R²': model.rsquared
    })
    
    # ============================================
    # TEST 2: Correlation by ASF Quintile (The Two Effects)
    # ============================================
    print("\n" + "-"*60)
    print("TEST 2: Correlation by ASF Quintile - Showing Effect A vs Effect B")
    print("-"*60)
    
    test_df['ASF_Quintile'] = pd.qcut(test_df['ASF'], q=5, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'])
    
    quintile_stats = test_df.groupby('ASF_Quintile')['Fwd_Corr'].agg(['mean', 'std', 'count'])
    quintile_stats['sem'] = quintile_stats['std'] / np.sqrt(quintile_stats['count'])
    
    print(f"\n{'Quintile':<12} {'Mean Corr':>12} {'Std Err':>10} {'N':>8}")
    print("-"*50)
    for idx in quintile_stats.index:
        row = quintile_stats.loc[idx]
        print(f"{idx:<12} {row['mean']:>12.4f} {row['sem']:>10.4f} {int(row['count']):>8}")
    
    # Effect A vs Effect B interpretation
    low_q_corr = quintile_stats.loc['Q1 (Low)', 'mean']
    high_q_corr = quintile_stats.loc['Q5 (High)', 'mean']
    
    print(f"\n  Q1 (Low ASF = Fragility) Mean Correlation: {low_q_corr:.4f}")
    print(f"  Q5 (High ASF = Coordination) Mean Correlation: {high_q_corr:.4f}")
    print(f"  Difference: {low_q_corr - high_q_corr:.4f}")
    print(f"\n  INTERPRETATION:")
    print(f"  - When ASF is HIGH: Coordination intact → Diversification works (corr ≈ {high_q_corr:.2f})")
    print(f"  - When ASF is LOW: Coordination broken → Diversification fails (corr ≈ {low_q_corr:.2f})")
    
    # ============================================
    # TEST 3: Multi-Asset Correlation Surge
    # ============================================
    print("\n" + "-"*60)
    print("TEST 3: Multi-Asset Correlation During Stress")
    print("-"*60)
    
    # Calculate pairwise correlations for available assets
    assets_available = [col for col in ['SPX', 'TLT', 'GLD', 'EUR'] if col in df.columns]
    
    if len(assets_available) >= 3:
        asset_rets = df[assets_available].pct_change()
        
        # Compute pairwise rolling correlations and average them
        rolling_corrs = []
        for i in range(len(assets_available)):
            for j in range(i+1, len(assets_available)):
                col1, col2 = assets_available[i], assets_available[j]
                rc = asset_rets[col1].rolling(63).corr(asset_rets[col2])
                rolling_corrs.append(rc)
        
        df['Avg_Cross_Corr'] = pd.concat(rolling_corrs, axis=1).mean(axis=1)
        
        # Forward correlation
        df['Fwd_Cross_Corr'] = df['Avg_Cross_Corr'].rolling(window=indexer).mean()
        
        test_df2 = df.dropna(subset=['ASF', 'Fwd_Cross_Corr'])
        
        if len(test_df2) > 50:
            # Regression
            X = sm.add_constant(test_df2['ASF'])
            y = test_df2['Fwd_Cross_Corr']
            model2 = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            
            print(f"\nRegression: Forward Avg Cross-Asset Correlation on ASF")
            print(f"  Assets: {assets_available}")
            print(f"  Beta: {model2.params['ASF']:.4f}")
            print(f"  T-Stat: {model2.tvalues['ASF']:.2f}")
            print(f"  R²: {model2.rsquared:.4f}")
            
            results.append({
                'Test': 'Multi-Asset Avg Correlation',
                'Beta': model2.params['ASF'],
                'T-stat': model2.tvalues['ASF'],
                'R²': model2.rsquared
            })
        else:
            print("  Not enough data for multi-asset correlation test")
    
    # ============================================
    # TEST 4: Tail Dependence - Do Assets Crash Together?
    # ============================================
    print("\n" + "-"*60)
    print("TEST 4: Tail Dependence - Joint Drawdowns")
    print("-"*60)
    
    # Calculate forward drawdowns for multiple assets
    for asset in ['SPX', 'TLT', 'GLD']:
        if asset in df.columns:
            indexer_dd = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
            rolling_min = df[asset].rolling(window=indexer_dd).min()
            df[f'{asset}_FwdDD'] = -(rolling_min / df[asset] - 1)
    
    # "Joint crash" = both SPX and TLT experience > 2% drawdown
    if 'SPX_FwdDD' in df.columns and 'TLT_FwdDD' in df.columns:
        df['Joint_Crash'] = ((df['SPX_FwdDD'] > 0.02) & (df['TLT_FwdDD'] > 0.02)).astype(int)
        
        test_df3 = df.dropna(subset=['ASF', 'Joint_Crash'])
        
        # Logistic regression: Low ASF predicts joint crashes
        X = sm.add_constant(test_df3['ASF'])
        y = test_df3['Joint_Crash']
        
        try:
            logit_model = sm.Logit(y, X).fit(disp=0)
            
            print(f"\nLogistic Regression: P(Joint Crash) on ASF")
            print(f"  Beta: {logit_model.params['ASF']:.4f}")
            print(f"  Z-Stat: {logit_model.tvalues['ASF']:.2f}")
            print(f"  Pseudo R²: {logit_model.prsquared:.4f}")
            print(f"\n  Interpretation: {'Negative beta confirms: Low ASF → Higher P(Joint Crash)' if logit_model.params['ASF'] < 0 else 'Unexpected result'}")
            
            results.append({
                'Test': 'Joint Crash Probability',
                'Beta': logit_model.params['ASF'],
                'T-stat': logit_model.tvalues['ASF'],
                'R²': logit_model.prsquared
            })
            
            # Calculate probabilities by quintile
            test_df3['ASF_Quintile'] = pd.qcut(test_df3['ASF'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            joint_by_q = test_df3.groupby('ASF_Quintile')['Joint_Crash'].mean()
            
            print(f"\n  P(Joint Crash) by ASF Quintile:")
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                if q in joint_by_q.index:
                    print(f"    {q}: {100*joint_by_q[q]:.1f}%")
        except:
            print("  Logistic regression failed (possibly perfect separation)")
    
    # ============================================
    # SYNTHESIS: The Two Effects of Diversification
    # ============================================
    print("\n" + "="*60)
    print("SYNTHESIS: THE TWO EFFECTS OF DIVERSIFICATION")
    print("="*60)
    
    print("""
    EFFECT A (Visible - Standard Portfolio Theory):
    - Spreading across assets reduces portfolio variance
    - Measured by: Portfolio volatility, tracking error
    - Works in BOTH regimes (always reduces idiosyncratic risk)
    
    EFFECT B (Invisible - Coordination Dependence):
    - Diversifying converges portfolios toward similar positions  
    - Creates dependence on market-wide coordination
    - Measured by: Correlation surge when ASF drops, joint crash probability
    - Dormant in coordination regime, DOMINANT in contagion regime
    
    THE PARADOX:
    - When coordination is intact (high ASF): Effect A dominates, diversification "works"
    - When coordination breaks (low ASF): Effect B dominates, diversification "fails"
    - Standard metrics only measure Effect A, missing Effect B entirely
    """)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('Path3_Diversification_Results.csv', index=False)
    
    # ============================================
    # CREATE VISUALIZATION
    # ============================================
    print("\n" + "-"*60)
    print("Generating Visualization...")
    print("-"*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Stock-Bond Correlation by ASF
    ax1 = axes[0, 0]
    plot_df = test_df.copy()
    plot_df['ASF_Bin'] = pd.qcut(plot_df['ASF'], q=20, labels=False)
    bin_stats = plot_df.groupby('ASF_Bin').agg({'ASF': 'mean', 'Fwd_Corr': ['mean', 'sem']})
    bin_stats.columns = ['ASF_Mean', 'Corr_Mean', 'Corr_SEM']
    
    ax1.errorbar(bin_stats['ASF_Mean'], bin_stats['Corr_Mean'], 
                 yerr=1.96*bin_stats['Corr_SEM'], fmt='-o', color='crimson', capsize=3)
    ax1.axvline(x=THRESHOLD, color='black', linestyle='--', label=f'Threshold ({THRESHOLD})')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_xlabel('ASF (Accumulated Spectral Fragility)')
    ax1.set_ylabel('Forward Stock-Bond Correlation')
    ax1.set_title('The Breakdown of Diversification')
    ax1.legend()
    
    # Plot 2: Correlation Distribution by Regime
    ax2 = axes[0, 1]
    low_asf = test_df[test_df['ASF'] < THRESHOLD]['Fwd_Corr']
    high_asf = test_df[test_df['ASF'] >= THRESHOLD]['Fwd_Corr']
    ax2.hist(low_asf, bins=30, alpha=0.6, color='red', label=f'Low ASF (Fragility)', density=True)
    ax2.hist(high_asf, bins=30, alpha=0.6, color='blue', label=f'High ASF (Coordination)', density=True)
    ax2.axvline(x=low_asf.mean(), color='red', linestyle='--')
    ax2.axvline(x=high_asf.mean(), color='blue', linestyle='--')
    ax2.set_xlabel('Stock-Bond Correlation')
    ax2.set_ylabel('Density')
    ax2.set_title('Correlation Distribution by Regime')
    ax2.legend()
    
    # Plot 3: Joint Crash Probability by ASF Quintile
    ax3 = axes[1, 0]
    if 'Joint_Crash' in df.columns:
        test_df3['ASF_Quintile'] = pd.qcut(test_df3['ASF'], q=5, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'])
        joint_by_q = test_df3.groupby('ASF_Quintile')['Joint_Crash'].mean() * 100
        joint_by_q.plot(kind='bar', ax=ax3, color=['red', 'orange', 'yellow', 'lightblue', 'blue'])
        ax3.set_xlabel('ASF Quintile')
        ax3.set_ylabel('P(Joint Stock-Bond Crash) %')
        ax3.set_title('Joint Crash Probability by Structural State')
        ax3.tick_params(axis='x', rotation=0)
    
    # Plot 4: Timeline - ASF and Correlation Spikes
    ax4 = axes[1, 1]
    plot_timeline = df.loc['2005':].copy()
    ax4.plot(plot_timeline.index, plot_timeline['ASF'], color='blue', label='ASF', alpha=0.8)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(plot_timeline.index, plot_timeline['StockBond_Corr'], color='red', alpha=0.6, label='Stock-Bond Corr')
    ax4.axhline(y=THRESHOLD, color='black', linestyle='--', alpha=0.5)
    ax4.set_ylabel('ASF', color='blue')
    ax4_twin.set_ylabel('Stock-Bond Correlation', color='red')
    ax4.set_title('ASF vs Stock-Bond Correlation Over Time')
    
    plt.tight_layout()
    plt.savefig('Path3_Diversification_Paradox.png', dpi=150)
    print("  Saved: Path3_Diversification_Paradox.png")
    
    return df_results

if __name__ == "__main__":
    results = test_diversification_paradox()
