"""
CARIA-SR Sensitivity Analysis
==============================

Comprehensive parameter sensitivity analysis for publication.

This module performs a systematic grid search over:
1. E4 Weight distributions
2. Rolling window sizes
3. Crash quantile thresholds
4. Alert thresholds

Outputs:
- Heatmaps showing AUC sensitivity
- Tables with exact values
- Optimal parameter configurations

Author: Tomás Basaure
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    """Load SPY and HYG data for sensitivity analysis."""
    print("Loading data...")
    
    # Credit anchor
    hyg = yf.download("HYG", start="2005-01-01", progress=False)
    if isinstance(hyg.columns, pd.MultiIndex):
        hyg_close = hyg["Close"].iloc[:, 0]
    else:
        hyg_close = hyg["Close"]
    ret_hyg = hyg_close.pct_change().dropna()
    
    # SPY
    spy = yf.download("SPY", start="2000-01-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy["Close"].iloc[:, 0]
    else:
        spy_close = spy["Close"]
    ret_spy = spy_close.pct_change().dropna()
    
    print(f"  HYG: {len(ret_hyg)} obs | SPY: {len(ret_spy)} obs")
    
    return ret_spy, ret_hyg


def compute_sr_with_params(ret_asset, ret_credit, params):
    """
    Compute CARIA-SR with specified parameters.
    
    Parameters:
    -----------
    ret_asset : pd.Series
        Asset returns
    ret_credit : pd.Series
        Credit returns (HYG)
    params : dict
        Parameters including:
        - w_fast, w_med, w_slow, w_credit: E4 weights
        - win_fast, win_med, win_slow, win_credit, win_rank: Windows
        - fwd_window: Forward looking window
        - crash_quantile: Quantile for crash definition
    
    Returns:
    --------
    pd.DataFrame with SR and Target
    """
    # Align indices
    common_idx = ret_asset.index.intersection(ret_credit.index)
    if len(common_idx) < 500:
        return None
    
    r = ret_asset.loc[common_idx]
    r_cred = ret_credit.loc[common_idx]
    
    # Credit volatility
    vol_credit = r_cred.rolling(params['win_credit']).std() * np.sqrt(252)
    
    # Asset volatility at multiple scales
    v_fast = r.rolling(params['win_fast']).std() * np.sqrt(252)
    v_med = r.rolling(params['win_med']).std() * np.sqrt(252)
    v_slow = r.rolling(params['win_slow']).std() * np.sqrt(252)
    
    # E4: Multi-scale energy
    E4_raw = (params['w_fast'] * v_fast + 
              params['w_med'] * v_med + 
              params['w_slow'] * v_slow + 
              params['w_credit'] * vol_credit)
    E4 = E4_raw.rolling(params['win_rank']).rank(pct=True)
    
    # Sync: Momentum correlation
    m_fast = r.rolling(params['win_fast']).sum()
    m_slow = r.rolling(params['win_slow']).sum()
    sync_raw = m_fast.rolling(params['win_med']).corr(m_slow)
    sync = ((sync_raw + 1) / 2).rolling(params['win_rank']).rank(pct=True)
    
    # CARIA-SR
    SR_raw = E4 * (1 + sync)
    SR = SR_raw.rolling(params['win_rank']).rank(pct=True)
    
    # Target: Real crashes
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=params['fwd_window'])
    fwd_ret = r.rolling(window=indexer).sum()
    crash_threshold = fwd_ret.quantile(params['crash_quantile'])
    is_crash = (fwd_ret < crash_threshold).astype(int)
    
    df = pd.DataFrame({
        'SR': SR,
        'Target': is_crash
    }).dropna()
    
    return df


def grid_search_weights(ret_asset, ret_credit, base_params):
    """
    Grid search over E4 weight combinations.
    
    Tests different distributions of weights across scales.
    """
    print("\n[1] Grid Search: E4 Weights")
    print("-" * 50)
    
    results = []
    
    # Weight combinations (must sum to 1)
    weight_grid = [
        # (fast, medium, slow, credit)
        (0.10, 0.30, 0.30, 0.30),  # Credit-heavy
        (0.15, 0.30, 0.30, 0.25),  # Balanced with credit
        (0.20, 0.30, 0.25, 0.25),  # DEFAULT
        (0.25, 0.25, 0.25, 0.25),  # Equal
        (0.30, 0.40, 0.30, 0.00),  # HAR-RV classic (no credit)
        (0.20, 0.40, 0.20, 0.20),  # Medium-heavy
        (0.30, 0.30, 0.20, 0.20),  # Fast-heavy
        (0.15, 0.25, 0.35, 0.25),  # Slow-heavy
    ]
    
    for w_fast, w_med, w_slow, w_credit in weight_grid:
        params = base_params.copy()
        params['w_fast'] = w_fast
        params['w_med'] = w_med
        params['w_slow'] = w_slow
        params['w_credit'] = w_credit
        
        df = compute_sr_with_params(ret_asset, ret_credit, params)
        
        if df is not None and len(df) > 500:
            auc = roc_auc_score(df['Target'], df['SR'])
            
            label = f"({w_fast:.2f},{w_med:.2f},{w_slow:.2f},{w_credit:.2f})"
            is_default = (w_fast == 0.20 and w_med == 0.30 and 
                         w_slow == 0.25 and w_credit == 0.25)
            
            results.append({
                'w_fast': w_fast,
                'w_med': w_med,
                'w_slow': w_slow,
                'w_credit': w_credit,
                'label': label,
                'auc': auc,
                'is_default': is_default
            })
            
            marker = " ← DEFAULT" if is_default else ""
            print(f"  {label}: AUC = {auc:.4f}{marker}")
    
    return pd.DataFrame(results)


def grid_search_windows(ret_asset, ret_credit, base_params):
    """
    Grid search over rolling window sizes.
    """
    print("\n[2] Grid Search: Rolling Windows")
    print("-" * 50)
    
    results = []
    
    # Window combinations
    # (fast, medium, slow, credit, rank)
    window_grid = [
        (3, 10, 42, 21, 126),    # Very fast
        (5, 15, 42, 30, 180),    # Fast
        (5, 21, 63, 42, 252),    # DEFAULT
        (5, 21, 63, 63, 252),    # Longer credit
        (10, 30, 90, 60, 252),   # Slower
        (10, 42, 126, 63, 504),  # Much slower
    ]
    
    for win_fast, win_med, win_slow, win_credit, win_rank in window_grid:
        params = base_params.copy()
        params['win_fast'] = win_fast
        params['win_med'] = win_med
        params['win_slow'] = win_slow
        params['win_credit'] = win_credit
        params['win_rank'] = win_rank
        
        df = compute_sr_with_params(ret_asset, ret_credit, params)
        
        if df is not None and len(df) > 500:
            auc = roc_auc_score(df['Target'], df['SR'])
            
            label = f"({win_fast},{win_med},{win_slow},{win_credit},{win_rank})"
            is_default = (win_fast == 5 and win_med == 21 and 
                         win_slow == 63 and win_credit == 42)
            
            results.append({
                'win_fast': win_fast,
                'win_med': win_med,
                'win_slow': win_slow,
                'win_credit': win_credit,
                'win_rank': win_rank,
                'label': label,
                'auc': auc,
                'is_default': is_default
            })
            
            marker = " ← DEFAULT" if is_default else ""
            print(f"  {label}: AUC = {auc:.4f}{marker}")
    
    return pd.DataFrame(results)


def grid_search_crash_quantile(ret_asset, ret_credit, base_params):
    """
    Grid search over crash quantile definitions.
    """
    print("\n[3] Grid Search: Crash Quantile")
    print("-" * 50)
    
    results = []
    
    quantiles = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
    
    for q in quantiles:
        params = base_params.copy()
        params['crash_quantile'] = q
        
        df = compute_sr_with_params(ret_asset, ret_credit, params)
        
        if df is not None and len(df) > 500:
            auc = roc_auc_score(df['Target'], df['SR'])
            n_crashes = df['Target'].sum()
            
            is_default = (q == 0.05)
            
            results.append({
                'quantile': q,
                'auc': auc,
                'n_crashes': n_crashes,
                'is_default': is_default
            })
            
            marker = " ← DEFAULT" if is_default else ""
            print(f"  q={q:.2f}: AUC = {auc:.4f}, N_crashes = {n_crashes}{marker}")
    
    return pd.DataFrame(results)


def grid_search_fwd_window(ret_asset, ret_credit, base_params):
    """
    Grid search over forward window sizes.
    """
    print("\n[4] Grid Search: Forward Window")
    print("-" * 50)
    
    results = []
    
    fwd_windows = [5, 10, 15, 21, 42, 63]
    
    for fwd in fwd_windows:
        params = base_params.copy()
        params['fwd_window'] = fwd
        
        df = compute_sr_with_params(ret_asset, ret_credit, params)
        
        if df is not None and len(df) > 500:
            auc = roc_auc_score(df['Target'], df['SR'])
            
            is_default = (fwd == 21)
            
            results.append({
                'fwd_window': fwd,
                'auc': auc,
                'is_default': is_default
            })
            
            marker = " ← DEFAULT" if is_default else ""
            print(f"  fwd={fwd}d: AUC = {auc:.4f}{marker}")
    
    return pd.DataFrame(results)


def create_heatmap_weights(results_df, output_path):
    """
    Create heatmap for weight sensitivity.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart ordered by AUC
    results_sorted = results_df.sort_values('auc', ascending=True)
    
    colors = ['#10b981' if row['is_default'] else '#6b7280' 
              for _, row in results_sorted.iterrows()]
    
    bars = ax.barh(range(len(results_sorted)), results_sorted['auc'], color=colors)
    
    ax.set_yticks(range(len(results_sorted)))
    ax.set_yticklabels(results_sorted['label'], fontsize=9)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title('E4 Weight Sensitivity: (fast, med, slow, credit)', fontsize=14)
    
    # Add value labels
    for bar, val in zip(bars, results_sorted['auc']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    # Highlight default
    ax.legend(['Random baseline', 'Default config'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_line_plot_quantile(results_df, output_path):
    """
    Create line plot for crash quantile sensitivity.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # AUC line
    ax1.plot(results_df['quantile'], results_df['auc'], 
             'b-o', linewidth=2, markersize=8, label='AUC')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax1.axvline(x=0.05, color='green', linestyle='--', alpha=0.5, label='Default (5%)')
    ax1.set_xlabel('Crash Quantile', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # N crashes on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(results_df['quantile'], results_df['n_crashes'], 
            alpha=0.3, color='gray', width=0.008, label='N Crashes')
    ax2.set_ylabel('Number of Crashes', fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    ax1.set_title('Sensitivity to Crash Definition Threshold', fontsize=14)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_summary_table(all_results, output_path):
    """
    Create summary table of all sensitivity analyses.
    """
    summary = []
    
    for name, df in all_results.items():
        if len(df) > 0:
            best = df.loc[df['auc'].idxmax()]
            worst = df.loc[df['auc'].idxmin()]
            default = df[df['is_default']] if 'is_default' in df.columns else df.iloc[[0]]
            
            summary.append({
                'Parameter': name,
                'Best_Config': best.get('label', str(best.iloc[0])),
                'Best_AUC': best['auc'],
                'Default_AUC': default['auc'].values[0] if len(default) > 0 else np.nan,
                'Worst_AUC': worst['auc'],
                'Range': best['auc'] - worst['auc']
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved summary: {output_path}")
    
    return summary_df


def run_full_sensitivity_analysis():
    """
    Run complete sensitivity analysis.
    """
    print("=" * 70)
    print("CARIA-SR SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Load data
    ret_spy, ret_hyg = load_data()
    
    # Default parameters
    base_params = {
        'w_fast': 0.20,
        'w_med': 0.30,
        'w_slow': 0.25,
        'w_credit': 0.25,
        'win_fast': 5,
        'win_med': 21,
        'win_slow': 63,
        'win_credit': 42,
        'win_rank': 252,
        'fwd_window': 21,
        'crash_quantile': 0.05
    }
    
    # Run all grid searches
    all_results = {}
    
    all_results['E4_Weights'] = grid_search_weights(ret_spy, ret_hyg, base_params)
    all_results['Windows'] = grid_search_windows(ret_spy, ret_hyg, base_params)
    all_results['Crash_Quantile'] = grid_search_crash_quantile(ret_spy, ret_hyg, base_params)
    all_results['Fwd_Window'] = grid_search_fwd_window(ret_spy, ret_hyg, base_params)
    
    # Generate visualizations
    print("\n[5] Generating Visualizations")
    print("-" * 50)
    
    create_heatmap_weights(
        all_results['E4_Weights'], 
        os.path.join(OUTPUT_DIR, 'Sensitivity_E4_Weights.png')
    )
    
    create_line_plot_quantile(
        all_results['Crash_Quantile'],
        os.path.join(OUTPUT_DIR, 'Sensitivity_Crash_Quantile.png')
    )
    
    # Save all results
    for name, df in all_results.items():
        df.to_csv(os.path.join(OUTPUT_DIR, f'Sensitivity_{name}.csv'), index=False)
        print(f"  ✓ Saved: Sensitivity_{name}.csv")
    
    # Create summary
    summary = create_summary_table(
        all_results,
        os.path.join(OUTPUT_DIR, 'Sensitivity_Summary.csv')
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SENSITIVITY SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    
    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Most sensitive parameter
    most_sensitive = summary.loc[summary['Range'].idxmax()]
    print(f"1. Most Sensitive Parameter: {most_sensitive['Parameter']}")
    print(f"   AUC Range: {most_sensitive['Range']:.4f}")
    
    # Robustness
    avg_range = summary['Range'].mean()
    print(f"\n2. Average AUC Range: {avg_range:.4f}")
    if avg_range < 0.05:
        print("   ✓ Model is ROBUST to parameter changes")
    else:
        print("   ⚠ Model shows sensitivity to parameters")
    
    # Default vs optimal
    default_auc = summary['Default_AUC'].mean()
    best_auc = summary['Best_AUC'].mean()
    print(f"\n3. Default vs Optimal:")
    print(f"   Default AUC (avg): {default_auc:.4f}")
    print(f"   Best AUC (avg):    {best_auc:.4f}")
    print(f"   Room for improvement: {(best_auc - default_auc):.4f}")
    
    return all_results


if __name__ == "__main__":
    results = run_full_sensitivity_analysis()
    print("\n✓ Sensitivity analysis complete!")

