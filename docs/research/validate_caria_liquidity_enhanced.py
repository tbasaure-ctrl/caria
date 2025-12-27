"""
CARIA-SR v8.0: Liquidity Enhanced Validation
=============================================

Enhancement: Amihud Illiquidity Ratio as additional filter.

Hypothesis (Empty Rally): 
High synchronization + High credit stress + High illiquidity = Dangerous melt-up
The market is rising on thin volume - a sign of fragility.

Components:
1. Original CARIA-SR (Sync × Energy with Credit)
2. Amihud Illiquidity Ratio: |Return| / (Price × Volume)
3. Combined Signal: Sync × Energy × Illiquidity

Reference:
- Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and 
  time-series effects." Journal of Financial Markets.

Author: Tomás Basaure
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# CONFIGURATION
# ==============================================================================

ASSETS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "EFA", "EEM", "GLD"]
START_DATE = "2005-01-01"
FWD_WINDOW = 21
CRASH_QUANTILE = 0.05

# Thresholds for signal generation
SYNC_THRESHOLD = 0.80
ENERGY_THRESHOLD = 0.80
ILLIQUIDITY_THRESHOLD = 0.70


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_credit_anchor():
    """Load HYG as credit anchor."""
    print("[1] Loading Credit Anchor (HYG)...")
    
    hyg = yf.download("HYG", start=START_DATE, progress=False)["Close"]
    if isinstance(hyg, pd.DataFrame):
        hyg = hyg.iloc[:, 0]
    
    ret_hyg = hyg.pct_change().dropna()
    vol_credit = ret_hyg.rolling(42).std() * np.sqrt(252)
    
    print(f"    ✓ Credit loaded: {len(vol_credit)} samples")
    return vol_credit


# ==============================================================================
# AMIHUD ILLIQUIDITY
# ==============================================================================

def compute_amihud(returns, price, volume, window=21, rank_window=252):
    """
    Compute Amihud Illiquidity Ratio.
    
    Formula: ILLIQ = |r| / (P × V)
    
    Interpretation:
    - Higher ILLIQ = Higher price impact per dollar traded = Less liquid
    - Rising ILLIQ in a rally = "Empty rally" - prices moving on thin volume
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    price : pd.Series
        Daily prices
    volume : pd.Series
        Daily volume
    window : int
        Smoothing window for Amihud
    rank_window : int
        Window for percentile ranking
        
    Returns:
    --------
    pd.Series : Percentile-ranked illiquidity
    """
    # Dollar volume
    dollar_vol = price * volume
    
    # Raw Amihud: |return| / dollar_volume
    # Add small constant to avoid division by zero
    amihud_raw = np.abs(returns) / (dollar_vol + 1e-9)
    
    # Smooth over window (reduce noise)
    amihud_smooth = amihud_raw.rolling(window).mean()
    
    # Rank over past year (0 = most liquid, 1 = most illiquid)
    amihud_rank = amihud_smooth.rolling(rank_window).rank(pct=True)
    
    return amihud_rank


# ==============================================================================
# VIX DATA
# ==============================================================================

def load_vix():
    """Load VIX data."""
    vix = yf.download("^VIX", start=START_DATE, progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    return vix


# ==============================================================================
# CARIA-SR COMPUTATION (WITH LIQUIDITY)
# ==============================================================================

def compute_caria_with_liquidity(ticker, vol_credit_series, vix_series=None):
    """
    Compute CARIA-SR with Amihud liquidity enhancement.
    
    Returns DataFrame with:
    - Original CARIA-SR components
    - Amihud illiquidity
    - Amihud × VIX combination
    - Combined signals
    - Forward returns and crash target
    """
    # Download with volume
    data = yf.download(ticker, start=START_DATE, progress=False)
    
    if len(data) < 500:
        return None
    
    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        price = data["Close"].iloc[:, 0]
        volume = data["Volume"].iloc[:, 0]
    else:
        price = data["Close"]
        volume = data["Volume"]
    
    ret = price.pct_change().dropna()
    
    # Align all data
    common_idx = (ret.index
                  .intersection(vol_credit_series.index)
                  .intersection(volume.index)
                  .intersection(price.index))
    
    # Also align with VIX if provided
    if vix_series is not None:
        common_idx = common_idx.intersection(vix_series.index)
    
    if len(common_idx) < 500:
        return None
    
    r = ret.loc[common_idx]
    p = price.loc[common_idx]
    v = volume.loc[common_idx]
    v_cred = vol_credit_series.loc[common_idx]
    
    # VIX percentile ranking
    if vix_series is not None:
        vix = vix_series.loc[common_idx]
        vix_pct = vix.rolling(252).rank(pct=True)
    else:
        vix_pct = None
    
    # --- ORIGINAL CARIA-SR COMPONENTS ---
    
    # Multi-scale volatility
    v5 = r.rolling(5).std() * np.sqrt(252)
    v21 = r.rolling(21).std() * np.sqrt(252)
    v63 = r.rolling(63).std() * np.sqrt(252)
    
    # E4: Energy with credit
    E4_raw = 0.20 * v5 + 0.30 * v21 + 0.25 * v63 + 0.25 * v_cred
    E4 = E4_raw.rolling(252).rank(pct=True)
    
    # Sync: Momentum correlation
    m_fast = r.rolling(5).sum()
    m_slow = r.rolling(63).sum()
    sync_raw = m_fast.rolling(21).corr(m_slow)
    sync = ((sync_raw + 1) / 2).rolling(252).rank(pct=True)
    
    # Original CARIA-SR
    SR_raw = E4 * (1 + sync)
    SR_original = SR_raw.rolling(252).rank(pct=True)
    
    # --- NEW: AMIHUD ILLIQUIDITY ---
    illiquidity = compute_amihud(r, p, v, window=21, rank_window=252)
    
    # --- NEW: AMIHUD × VIX ---
    # Theory: High illiquidity + High VIX = Liquidity crisis
    #         High illiquidity + Low VIX = Empty rally (complacent but fragile)
    if vix_pct is not None:
        amihud_vix = illiquidity * vix_pct
        amihud_vix = amihud_vix.rolling(252).rank(pct=True)
    else:
        amihud_vix = illiquidity  # Fallback
    
    # --- ENHANCED CARIA-SR ---
    # Combine: SR × Illiquidity (higher = more fragile)
    SR_enhanced = SR_original * illiquidity
    SR_enhanced = SR_enhanced.rolling(252).rank(pct=True)
    
    # Alternative: Additive combination
    SR_additive = (0.5 * SR_original + 0.5 * illiquidity)
    SR_additive = SR_additive.rolling(252).rank(pct=True)
    
    # NEW: SR × (Amihud × VIX) - Double interaction
    SR_amihud_vix = SR_original * amihud_vix
    SR_amihud_vix = SR_amihud_vix.rolling(252).rank(pct=True)
    
    # NEW: Direct Amihud × VIX (without SR)
    # This tests if liquidity × fear alone predicts crashes
    
    # --- TARGET: Real Crashes ---
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FWD_WINDOW)
    fwd_ret = r.rolling(window=indexer).sum()
    crash_threshold = fwd_ret.quantile(CRASH_QUANTILE)
    is_crash = (fwd_ret < crash_threshold).astype(int)
    
    # --- BUILD DATAFRAME ---
    df_dict = {
        'SR_Original': SR_original,
        'SR_Enhanced': SR_enhanced,
        'SR_Additive': SR_additive,
        'SR_AmihudVix': SR_amihud_vix,
        'E4': E4,
        'Sync': sync,
        'Illiquidity': illiquidity,
        'Amihud_x_VIX': amihud_vix,
        'Fwd_Ret': fwd_ret,
        'Target_Crash': is_crash,
        'Returns': r
    }
    
    # Add VIX if available
    if vix_pct is not None:
        df_dict['VIX_pct'] = vix_pct
    
    df = pd.DataFrame(df_dict).dropna()
    
    return df


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def bootstrap_auc_ci(y_true, y_score, n_bootstrap=1000):
    """Bootstrap CI for AUC."""
    np.random.seed(RANDOM_SEED)
    
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    n = len(y_true)
    
    point_auc = roc_auc_score(y_true, y_score)
    
    bootstrap_aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        y_boot = y_true[idx]
        s_boot = y_score[idx]
        
        if len(np.unique(y_boot)) < 2:
            continue
        
        try:
            bootstrap_aucs.append(roc_auc_score(y_boot, s_boot))
        except:
            pass
    
    bootstrap_aucs = np.array(bootstrap_aucs)
    
    return {
        'point': point_auc,
        'ci_lower': np.percentile(bootstrap_aucs, 2.5),
        'ci_upper': np.percentile(bootstrap_aucs, 97.5),
        'se': np.std(bootstrap_aucs)
    }


def compute_signal_precision(df, signal_col, threshold=0.8):
    """
    Compute precision for a signal at threshold.
    
    Precision = P(Crash | Signal > threshold)
    """
    signal_high = df[signal_col] > threshold
    
    if signal_high.sum() == 0:
        return np.nan, 0
    
    precision = df.loc[signal_high, 'Target_Crash'].mean()
    n_signals = signal_high.sum()
    
    return precision, n_signals


def validate_liquidity_enhancement(ticker, vol_credit, vix_series=None, n_bootstrap=1000):
    """
    Full validation comparing original vs liquidity-enhanced CARIA-SR.
    """
    df = compute_caria_with_liquidity(ticker, vol_credit, vix_series)
    
    if df is None or len(df) < 500:
        return None
    
    results = {'ticker': ticker, 'n_obs': len(df)}
    
    # --- AUC Comparison ---
    
    # Original CARIA-SR
    auc_orig = bootstrap_auc_ci(df['Target_Crash'], df['SR_Original'], n_bootstrap)
    results['auc_original'] = auc_orig['point']
    results['auc_original_ci'] = (auc_orig['ci_lower'], auc_orig['ci_upper'])
    
    # Enhanced (multiplicative)
    auc_enh = bootstrap_auc_ci(df['Target_Crash'], df['SR_Enhanced'], n_bootstrap)
    results['auc_enhanced'] = auc_enh['point']
    results['auc_enhanced_ci'] = (auc_enh['ci_lower'], auc_enh['ci_upper'])
    
    # Additive
    auc_add = bootstrap_auc_ci(df['Target_Crash'], df['SR_Additive'], n_bootstrap)
    results['auc_additive'] = auc_add['point']
    results['auc_additive_ci'] = (auc_add['ci_lower'], auc_add['ci_upper'])
    
    # Illiquidity alone
    auc_illiq = bootstrap_auc_ci(df['Target_Crash'], df['Illiquidity'], n_bootstrap)
    results['auc_illiquidity'] = auc_illiq['point']
    
    # Amihud × VIX
    auc_amihud_vix = bootstrap_auc_ci(df['Target_Crash'], df['Amihud_x_VIX'], n_bootstrap)
    results['auc_amihud_vix'] = auc_amihud_vix['point']
    results['auc_amihud_vix_ci'] = (auc_amihud_vix['ci_lower'], auc_amihud_vix['ci_upper'])
    
    # SR × (Amihud × VIX)
    auc_sr_amihud_vix = bootstrap_auc_ci(df['Target_Crash'], df['SR_AmihudVix'], n_bootstrap)
    results['auc_sr_amihud_vix'] = auc_sr_amihud_vix['point']
    results['auc_sr_amihud_vix_ci'] = (auc_sr_amihud_vix['ci_lower'], auc_sr_amihud_vix['ci_upper'])
    
    # Delta for Amihud × VIX version
    results['delta_auc_amihud_vix'] = auc_sr_amihud_vix['point'] - auc_orig['point']
    
    # --- Precision at High Threshold ---
    
    prec_orig, n_orig = compute_signal_precision(df, 'SR_Original', 0.8)
    prec_enh, n_enh = compute_signal_precision(df, 'SR_Enhanced', 0.8)
    prec_add, n_add = compute_signal_precision(df, 'SR_Additive', 0.8)
    
    results['precision_original'] = prec_orig
    results['precision_enhanced'] = prec_enh
    results['precision_additive'] = prec_add
    results['n_signals_original'] = n_orig
    results['n_signals_enhanced'] = n_enh
    
    # --- Delta AUC ---
    results['delta_auc_enhanced'] = auc_enh['point'] - auc_orig['point']
    results['delta_auc_additive'] = auc_add['point'] - auc_orig['point']
    
    # --- Minsky Premium (Forward returns when signal high) ---
    alert_mask_orig = df['SR_Original'] > 0.8
    alert_mask_enh = df['SR_Enhanced'] > 0.8
    
    if alert_mask_orig.sum() > 10:
        results['minsky_original'] = df.loc[alert_mask_orig, 'Fwd_Ret'].mean()
    else:
        results['minsky_original'] = np.nan
    
    if alert_mask_enh.sum() > 10:
        results['minsky_enhanced'] = df.loc[alert_mask_enh, 'Fwd_Ret'].mean()
    else:
        results['minsky_enhanced'] = np.nan
    
    # Store df for visualization
    results['_df'] = df
    
    return results


# ==============================================================================
# MAIN VALIDATION
# ==============================================================================

def run_liquidity_validation(assets=None, n_bootstrap=1000):
    """
    Run complete liquidity enhancement validation.
    """
    if assets is None:
        assets = ASSETS
    
    print("=" * 80)
    print("CARIA-SR v8.0: LIQUIDITY ENHANCED VALIDATION")
    print("=" * 80)
    print(f"Hypothesis: High Sync + High Energy + High Illiquidity = Fragile Melt-up")
    print(f"NEW: Testing Amihud × VIX interaction")
    print(f"Assets: {len(assets)} | Bootstrap: {n_bootstrap}")
    print("=" * 80)
    
    # Load credit
    vol_credit = load_credit_anchor()
    
    # Load VIX
    print("[2] Loading VIX...")
    vix_series = load_vix()
    print(f"    ✓ VIX loaded: {len(vix_series)} samples")
    
    # Validate each asset
    print("\n[3] Asset Validation")
    print("-" * 110)
    print(f"{'Asset':<6} | {'AUC Orig':>9} | {'AUC Illiq':>9} | {'AUC Amh×VIX':>11} | "
          f"{'AUC SR×Amh×VIX':>14} | {'ΔAUC':>7} | {'Best'}")
    print("-" * 110)
    
    all_results = []
    asset_dfs = {}
    
    for ticker in assets:
        result = validate_liquidity_enhancement(ticker, vol_credit, vix_series, n_bootstrap)
        
        if result is None:
            print(f"{ticker:<6} | {'SKIP':^9}")
            continue
        
        # Store df
        asset_dfs[ticker] = result.pop('_df')
        all_results.append(result)
        
        # Format output
        delta_amihud_vix = result.get('delta_auc_amihud_vix', 0)
        
        # Find best model
        aucs = {
            'Original': result['auc_original'],
            'Illiq': result['auc_illiquidity'],
            'Amh×VIX': result['auc_amihud_vix'],
            'SR×Amh×VIX': result['auc_sr_amihud_vix']
        }
        best_model = max(aucs, key=aucs.get)
        
        print(f"{ticker:<6} | {result['auc_original']:>9.4f} | {result['auc_illiquidity']:>9.4f} | "
              f"{result['auc_amihud_vix']:>11.4f} | {result['auc_sr_amihud_vix']:>14.4f} | "
              f"{delta_amihud_vix:>+7.4f} | {best_model}")
    
    print("-" * 110)
    
    # --- Aggregate Results ---
    results_df = pd.DataFrame(all_results)
    
    print("\n[4] Aggregate Statistics")
    print("-" * 60)
    print(f"Mean AUC Original:       {results_df['auc_original'].mean():.4f}")
    print(f"Mean AUC Illiquidity:    {results_df['auc_illiquidity'].mean():.4f}")
    print(f"Mean AUC Amihud×VIX:     {results_df['auc_amihud_vix'].mean():.4f}")
    print(f"Mean AUC SR×Amihud×VIX:  {results_df['auc_sr_amihud_vix'].mean():.4f}")
    print(f"Mean ΔAUC (Amh×VIX):     {results_df['delta_auc_amihud_vix'].mean():+.4f}")
    
    # Count improvements for Amihud×VIX
    n_improved = (results_df['delta_auc_amihud_vix'] > 0.01).sum()
    n_same = ((results_df['delta_auc_amihud_vix'] >= -0.01) & 
              (results_df['delta_auc_amihud_vix'] <= 0.01)).sum()
    n_worse = (results_df['delta_auc_amihud_vix'] < -0.01).sum()
    
    print(f"\nAmihud×VIX Improvement Summary:")
    print(f"  ✓ Better: {n_improved}/{len(results_df)}")
    print(f"  ≈ Same:   {n_same}/{len(results_df)}")
    print(f"  ✗ Worse:  {n_worse}/{len(results_df)}")
    
    # Best model count
    print(f"\nBest Model per Asset:")
    for model in ['Original', 'Illiq', 'Amh×VIX', 'SR×Amh×VIX']:
        count = 0
        for _, row in results_df.iterrows():
            aucs = {
                'Original': row['auc_original'],
                'Illiq': row['auc_illiquidity'],
                'Amh×VIX': row['auc_amihud_vix'],
                'SR×Amh×VIX': row['auc_sr_amihud_vix']
            }
            if max(aucs, key=aucs.get) == model:
                count += 1
        print(f"  {model}: {count}/{len(results_df)}")
    
    # --- Precision Analysis ---
    print("\n[4] Precision Analysis (Signal Quality)")
    print("-" * 50)
    
    prec_orig_mean = results_df['precision_original'].dropna().mean()
    prec_enh_mean = results_df['precision_enhanced'].dropna().mean()
    
    print(f"Mean Precision Original: {prec_orig_mean:.1%}")
    print(f"Mean Precision Enhanced: {prec_enh_mean:.1%}")
    print(f"Precision Improvement:   {prec_enh_mean - prec_orig_mean:+.1%}")
    
    # --- Statistical Test ---
    print("\n[6] Statistical Significance")
    print("-" * 60)
    
    # Paired t-test on Amihud×VIX improvement
    delta_aucs = results_df['delta_auc_amihud_vix'].values
    t_stat, p_value = stats.ttest_1samp(delta_aucs, 0)
    
    print(f"Paired t-test (H0: ΔAUC SR×Amihud×VIX = 0):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_value:.4f}")
    
    if p_value < 0.05 and t_stat > 0:
        print(f"  ✓ Amihud×VIX SIGNIFICANTLY improves AUC (p < 0.05)")
    elif p_value < 0.05 and t_stat < 0:
        print(f"  ✗ Amihud×VIX SIGNIFICANTLY worsens AUC (p < 0.05)")
    else:
        print(f"  ≈ No significant difference (p >= 0.05)")
    
    # Compare Amihud×VIX vs Illiquidity alone
    print(f"\nComparing Amihud×VIX vs Illiquidity alone:")
    delta_vs_illiq = results_df['auc_amihud_vix'].values - results_df['auc_illiquidity'].values
    t_stat2, p_value2 = stats.ttest_1samp(delta_vs_illiq, 0)
    print(f"  Mean ΔAUC (Amh×VIX - Illiq): {delta_vs_illiq.mean():+.4f}")
    print(f"  t-statistic: {t_stat2:.3f}, p-value: {p_value2:.4f}")
    
    return results_df, asset_dfs


def generate_liquidity_outputs(results_df, asset_dfs, output_dir=None):
    """
    Generate tables and figures for liquidity enhancement.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # --- Table: Liquidity Enhancement Comparison ---
    table_cols = ['ticker', 'auc_original', 'auc_illiquidity', 'auc_amihud_vix', 
                  'auc_sr_amihud_vix', 'delta_auc_amihud_vix']
    table = results_df[[c for c in table_cols if c in results_df.columns]].copy()
    table.columns = ['Asset', 'AUC_Original', 'AUC_Illiquidity', 'AUC_Amihud_x_VIX',
                     'AUC_SR_x_Amihud_x_VIX', 'Delta_AUC']
    table.to_csv(os.path.join(output_dir, 'Table_Liquidity_Enhancement.csv'), index=False)
    print(f"\n✓ Saved: Table_Liquidity_Enhancement.csv")
    
    # --- Figure: AUC Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart - all models
    ax1 = axes[0]
    x = np.arange(len(results_df))
    width = 0.2
    
    bars1 = ax1.bar(x - 1.5*width, results_df['auc_original'], width, 
                    label='Original CARIA-SR', color='#6b7280')
    bars2 = ax1.bar(x - 0.5*width, results_df['auc_illiquidity'], width, 
                    label='Illiquidity Only', color='#f97316')
    bars3 = ax1.bar(x + 0.5*width, results_df['auc_amihud_vix'], width, 
                    label='Amihud × VIX', color='#8b5cf6')
    bars4 = ax1.bar(x + 1.5*width, results_df['auc_sr_amihud_vix'], width, 
                    label='SR × Amihud × VIX', color='#3b82f6')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['ticker'], rotation=45)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax1.set_ylabel('AUC')
    ax1.set_title('AUC Comparison: All Liquidity Features')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_ylim(0.45, 0.85)
    
    # Delta AUC for Amihud×VIX
    ax2 = axes[1]
    colors = ['#10b981' if d > 0 else '#ef4444' for d in results_df['delta_auc_amihud_vix']]
    bars = ax2.bar(results_df['ticker'], results_df['delta_auc_amihud_vix'], color=colors)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax2.set_ylabel('ΔAUC (SR×Amihud×VIX - Original)')
    ax2.set_title('AUC Improvement from Amihud × VIX Enhancement')
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_Liquidity_AUC_Comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: Figure_Liquidity_AUC_Comparison.png")
    
    # --- Figure: ROC Curves (SPY) ---
    if 'SPY' in asset_dfs:
        df_spy = asset_dfs['SPY']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for col, label, color, lw in [
            ('SR_Original', 'CARIA-SR Original', '#6b7280', 1.5),
            ('Illiquidity', 'Illiquidity Only', '#f97316', 1.5),
            ('Amihud_x_VIX', 'Amihud × VIX', '#8b5cf6', 2.0),
            ('SR_AmihudVix', 'SR × Amihud × VIX', '#3b82f6', 2.5),
        ]:
            if col in df_spy.columns:
                fpr, tpr, _ = roc_curve(df_spy['Target_Crash'], df_spy[col])
                auc_val = roc_auc_score(df_spy['Target_Crash'], df_spy[col])
                ax.plot(fpr, tpr, color=color, linewidth=lw, 
                        label=f'{label} (AUC={auc_val:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves: Amihud × VIX Enhancement (SPY)', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure_Liquidity_ROC_SPY.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: Figure_Liquidity_ROC_SPY.png")
    
    # --- Figure: Illiquidity Timeline (SPY) ---
    if 'SPY' in asset_dfs:
        df_spy = asset_dfs['SPY']
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        dates = df_spy.index
        
        # Panel 1: Original SR vs SR×Amihud×VIX
        ax1 = axes[0]
        ax1.plot(dates, df_spy['SR_Original'], 'b-', linewidth=1, alpha=0.7, label='Original')
        ax1.plot(dates, df_spy['SR_AmihudVix'], 'purple', linewidth=1, alpha=0.7, label='SR×Amihud×VIX')
        ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('CARIA-SR')
        ax1.legend(loc='upper left')
        ax1.set_title('CARIA-SR: Original vs Amihud×VIX Enhanced (SPY)')
        
        # Panel 2: Illiquidity
        ax2 = axes[1]
        ax2.fill_between(dates, 0, df_spy['Illiquidity'], alpha=0.5, color='orange', label='Amihud')
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Illiquidity')
        ax2.legend(loc='upper left')
        
        # Panel 3: Amihud × VIX
        ax3 = axes[2]
        ax3.fill_between(dates, 0, df_spy['Amihud_x_VIX'], alpha=0.5, color='purple', label='Amihud×VIX')
        ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
        ax3.set_ylabel('Amihud × VIX')
        ax3.legend(loc='upper left')
        
        # Panel 4: Forward returns
        ax4 = axes[3]
        ax4.fill_between(dates, 0, df_spy['Fwd_Ret'], 
                         where=df_spy['Fwd_Ret'] > 0, alpha=0.5, color='green')
        ax4.fill_between(dates, 0, df_spy['Fwd_Ret'], 
                         where=df_spy['Fwd_Ret'] <= 0, alpha=0.5, color='red')
        ax4.set_ylabel('Forward Return (21d)')
        ax4.set_xlabel('Date')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure_Liquidity_Timeline_SPY.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: Figure_Liquidity_Timeline_SPY.png")
    
    print(f"\n✓ All liquidity enhancement outputs saved")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Run validation
    results_df, asset_dfs = run_liquidity_validation(n_bootstrap=1000)
    
    # Generate outputs
    generate_liquidity_outputs(results_df, asset_dfs)
    
    # Summary
    print("\n" + "=" * 80)
    print("LIQUIDITY ENHANCEMENT SUMMARY")
    print("=" * 80)
    
    mean_delta_amihud_vix = results_df['delta_auc_amihud_vix'].mean()
    mean_auc_amihud_vix = results_df['auc_amihud_vix'].mean()
    mean_auc_illiq = results_df['auc_illiquidity'].mean()
    
    print(f"\nKey Results:")
    print(f"  Mean AUC Illiquidity alone:  {mean_auc_illiq:.4f}")
    print(f"  Mean AUC Amihud × VIX:       {mean_auc_amihud_vix:.4f}")
    print(f"  Mean ΔAUC (SR×Amh×VIX):      {mean_delta_amihud_vix:+.4f}")
    
    if mean_delta_amihud_vix > 0.01:
        print(f"\n✓ CONCLUSION: Amihud × VIX IMPROVES CARIA-SR")
        print(f"  Interpretation: Combining illiquidity with VIX (fear)")
        print(f"  helps identify stress conditions that precede crashes.")
    elif mean_delta_amihud_vix < -0.01:
        print(f"\n✗ CONCLUSION: Amihud × VIX DOES NOT improve CARIA-SR")
        print(f"  Interpretation: The interaction may add noise or")
        print(f"  the original CARIA-SR already captures these dynamics.")
    else:
        print(f"\n≈ CONCLUSION: Amihud × VIX has MARGINAL effect on CARIA-SR")
        print(f"  However, note that Amihud × VIX alone achieves AUC {mean_auc_amihud_vix:.4f}")
        print(f"  which is competitive with original CARIA-SR.")
    
    # Check if Amihud×VIX is better than Illiquidity alone
    if mean_auc_amihud_vix > mean_auc_illiq + 0.01:
        print(f"\n  ✓ VIX adds value: Amihud×VIX ({mean_auc_amihud_vix:.4f}) > Illiq ({mean_auc_illiq:.4f})")
    elif mean_auc_amihud_vix < mean_auc_illiq - 0.01:
        print(f"\n  ✗ VIX reduces value: Amihud×VIX ({mean_auc_amihud_vix:.4f}) < Illiq ({mean_auc_illiq:.4f})")
    else:
        print(f"\n  ≈ VIX neutral: Amihud×VIX ({mean_auc_amihud_vix:.4f}) ≈ Illiq ({mean_auc_illiq:.4f})")
    
    print("\n" + "=" * 80)

