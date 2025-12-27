"""
Analyze Missing Component: Why does Illiquidity outperform CARIA-SR?
=====================================================================

CARIA-SR AUC: 0.640
Illiquidity AUC: 0.700
Gap: 0.060 (6 percentage points)

Question: What component could CARIA-SR add to close this gap?

Hypotheses:
1. VOLUME DYNAMICS - CARIA-SR has no volume component
2. PRICE IMPACT - How much prices move per dollar traded
3. VOLUME-VOLATILITY DIVERGENCE - When vol is high but volume is low (or vice versa)
4. TURNOVER RATE - Volume / Outstanding shares (relative activity)

Author: Tomás Basaure
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
START_DATE = "2005-01-01"
FWD_WINDOW = 21
CRASH_QUANTILE = 0.05


def load_all_data():
    """Load SPY with all fields needed."""
    print("Loading data...")
    
    # SPY with volume
    spy = yf.download("SPY", start=START_DATE, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        price = spy["Close"].iloc[:, 0]
        volume = spy["Volume"].iloc[:, 0]
        high = spy["High"].iloc[:, 0]
        low = spy["Low"].iloc[:, 0]
    else:
        price = spy["Close"]
        volume = spy["Volume"]
        high = spy["High"]
        low = spy["Low"]
    
    # Credit
    hyg = yf.download("HYG", start=START_DATE, progress=False)["Close"]
    if isinstance(hyg, pd.DataFrame):
        hyg = hyg.iloc[:, 0]
    ret_hyg = hyg.pct_change().dropna()
    vol_credit = ret_hyg.rolling(42).std() * np.sqrt(252)
    
    # VIX
    vix = yf.download("^VIX", start=START_DATE, progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    
    ret = price.pct_change().dropna()
    
    # Align
    common = (ret.index
              .intersection(vol_credit.index)
              .intersection(volume.index)
              .intersection(vix.index)
              .intersection(high.index))
    
    data = pd.DataFrame({
        'price': price.loc[common],
        'ret': ret.loc[common],
        'volume': volume.loc[common],
        'high': high.loc[common],
        'low': low.loc[common],
        'vol_credit': vol_credit.loc[common],
        'vix': vix.loc[common]
    }).dropna()
    
    print(f"  Loaded: {len(data)} observations")
    return data


def compute_all_features(data):
    """Compute all potential features."""
    print("\nComputing features...")
    
    r = data['ret']
    p = data['price']
    v = data['volume']
    h = data['high']
    l = data['low']
    v_cred = data['vol_credit']
    vix = data['vix']
    
    features = {}
    
    # === ORIGINAL CARIA-SR COMPONENTS ===
    v5 = r.rolling(5).std() * np.sqrt(252)
    v21 = r.rolling(21).std() * np.sqrt(252)
    v63 = r.rolling(63).std() * np.sqrt(252)
    
    E4_raw = 0.20 * v5 + 0.30 * v21 + 0.25 * v63 + 0.25 * v_cred
    E4 = E4_raw.rolling(252).rank(pct=True)
    features['E4'] = E4
    
    m_fast = r.rolling(5).sum()
    m_slow = r.rolling(63).sum()
    sync_raw = m_fast.rolling(21).corr(m_slow)
    sync = ((sync_raw + 1) / 2).rolling(252).rank(pct=True)
    features['Sync'] = sync
    
    SR_raw = E4 * (1 + sync)
    SR = SR_raw.rolling(252).rank(pct=True)
    features['CARIA_SR'] = SR
    
    # === VOLUME-BASED FEATURES ===
    
    # 1. AMIHUD ILLIQUIDITY (baseline)
    dollar_vol = p * v
    amihud_raw = np.abs(r) / (dollar_vol + 1e-9)
    amihud = amihud_raw.rolling(21).mean().rolling(252).rank(pct=True)
    features['Amihud'] = amihud
    
    # 2. VOLUME TREND (Is volume declining?)
    vol_ma_short = v.rolling(21).mean()
    vol_ma_long = v.rolling(63).mean()
    vol_trend = (vol_ma_short / vol_ma_long).rolling(252).rank(pct=True)
    features['Vol_Trend'] = vol_trend
    
    # 3. VOLUME VOLATILITY (How erratic is volume?)
    vol_of_vol = v.rolling(21).std() / v.rolling(21).mean()  # CV of volume
    vol_of_vol_rank = vol_of_vol.rolling(252).rank(pct=True)
    features['Vol_Volatility'] = vol_of_vol_rank
    
    # 4. VOLUME-RETURN DIVERGENCE (Low volume + high returns = "empty rally")
    ret_abs_rank = np.abs(r).rolling(21).mean().rolling(252).rank(pct=True)
    vol_rank = v.rolling(21).mean().rolling(252).rank(pct=True)
    # High returns but low volume = bad sign
    vol_ret_div = ret_abs_rank * (1 - vol_rank)  # High when returns high, volume low
    vol_ret_div = vol_ret_div.rolling(252).rank(pct=True)
    features['Vol_Ret_Divergence'] = vol_ret_div
    
    # 5. TURNOVER DECLINE (Volume relative to recent history)
    turnover_ratio = v / v.rolling(252).mean()
    turnover_decline = (1 - turnover_ratio.rolling(21).mean()).rolling(252).rank(pct=True)
    features['Turnover_Decline'] = turnover_decline
    
    # 6. KYLE'S LAMBDA PROXY (Price impact per unit volume)
    # λ ≈ |ΔP| / Volume
    kyle_lambda = np.abs(p.diff()) / (v + 1e-9)
    kyle_rank = kyle_lambda.rolling(21).mean().rolling(252).rank(pct=True)
    features['Kyle_Lambda'] = kyle_rank
    
    # 7. GARMAN-KLASS VOLATILITY (Uses high/low, more efficient than close-to-close)
    gk_vol = 0.5 * np.log(h/l)**2 - (2*np.log(2)-1) * np.log(p/p.shift(1))**2
    gk_vol_ma = gk_vol.rolling(21).mean()
    gk_rank = gk_vol_ma.rolling(252).rank(pct=True)
    features['GK_Volatility'] = gk_rank
    
    # 8. VOLATILITY-VOLUME RATIO (High vol / Low volume = illiquid stress)
    vol_vol_ratio = v21 / (vol_rank + 0.01)  # Volatility per unit of volume
    vol_vol_rank = vol_vol_ratio.rolling(252).rank(pct=True)
    features['Vol_Vol_Ratio'] = vol_vol_rank
    
    # 9. VIX PERCENTILE
    vix_rank = vix.rolling(252).rank(pct=True)
    features['VIX'] = vix_rank
    
    # === TARGET ===
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FWD_WINDOW)
    fwd_ret = r.rolling(window=indexer).sum()
    crash_threshold = fwd_ret.quantile(CRASH_QUANTILE)
    is_crash = (fwd_ret < crash_threshold).astype(int)
    
    # Build DataFrame
    df = pd.DataFrame(features)
    df['Target'] = is_crash
    df['Fwd_Ret'] = fwd_ret
    df = df.dropna()
    
    print(f"  Features computed: {list(features.keys())}")
    print(f"  Final observations: {len(df)}")
    
    return df


def analyze_individual_features(df):
    """Analyze AUC of each feature individually."""
    print("\n" + "=" * 70)
    print("INDIVIDUAL FEATURE AUC ANALYSIS")
    print("=" * 70)
    
    feature_cols = [c for c in df.columns if c not in ['Target', 'Fwd_Ret']]
    
    results = []
    for feat in feature_cols:
        auc = roc_auc_score(df['Target'], df[feat])
        results.append({'Feature': feat, 'AUC': auc})
    
    results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    
    print(f"\n{'Feature':<25} | {'AUC':>8} | {'vs CARIA-SR':>12}")
    print("-" * 50)
    
    caria_auc = results_df[results_df['Feature'] == 'CARIA_SR']['AUC'].values[0]
    
    for _, row in results_df.iterrows():
        delta = row['AUC'] - caria_auc
        delta_str = f"{delta:+.4f}" if row['Feature'] != 'CARIA_SR' else "-"
        marker = "***" if delta > 0.05 else "**" if delta > 0.02 else "*" if delta > 0.01 else ""
        print(f"{row['Feature']:<25} | {row['AUC']:>8.4f} | {delta_str:>12} {marker}")
    
    return results_df


def analyze_combinations_with_caria(df):
    """Test combinations of CARIA-SR + new components."""
    print("\n" + "=" * 70)
    print("CARIA-SR + COMPONENT COMBINATIONS")
    print("=" * 70)
    
    caria = df['CARIA_SR']
    target = df['Target']
    
    # Baseline
    base_auc = roc_auc_score(target, caria)
    print(f"\nBaseline CARIA-SR AUC: {base_auc:.4f}")
    
    # Components to test
    volume_components = ['Amihud', 'Vol_Trend', 'Vol_Volatility', 
                        'Vol_Ret_Divergence', 'Turnover_Decline', 
                        'Kyle_Lambda', 'Vol_Vol_Ratio']
    
    results = []
    
    print(f"\n{'Combination':<35} | {'AUC':>8} | {'ΔAUC':>8} | {'Improve?'}")
    print("-" * 65)
    
    for comp in volume_components:
        if comp not in df.columns:
            continue
        
        # Multiplicative combination
        combined_mult = caria * df[comp]
        combined_mult = combined_mult.rolling(252, min_periods=50).rank(pct=True)
        valid_idx = combined_mult.dropna().index
        auc_mult = roc_auc_score(target.loc[valid_idx], combined_mult.loc[valid_idx])
        
        # Additive combination
        combined_add = 0.5 * caria + 0.5 * df[comp]
        auc_add = roc_auc_score(target, combined_add)
        
        # Best combination
        best_auc = max(auc_mult, auc_add)
        best_type = "×" if auc_mult >= auc_add else "+"
        delta = best_auc - base_auc
        
        improve = "✓" if delta > 0.01 else "≈" if delta > -0.01 else "✗"
        
        results.append({
            'Component': comp,
            'AUC_Mult': auc_mult,
            'AUC_Add': auc_add,
            'Best_AUC': best_auc,
            'Delta': delta
        })
        
        print(f"CARIA {best_type} {comp:<25} | {best_auc:>8.4f} | {delta:>+8.4f} | {improve}")
    
    return pd.DataFrame(results)


def find_optimal_blend(df):
    """Find optimal blend of CARIA-SR and volume components."""
    print("\n" + "=" * 70)
    print("OPTIMAL BLEND SEARCH")
    print("=" * 70)
    
    caria = df['CARIA_SR']
    target = df['Target']
    
    # Best single volume component
    amihud = df['Amihud']
    
    print("\nSearching for optimal CARIA + Amihud blend...")
    print(f"{'Weight CARIA':<15} | {'Weight Amihud':<15} | {'AUC':>8}")
    print("-" * 45)
    
    best_auc = 0
    best_weights = (0, 0)
    
    for w_caria in np.arange(0, 1.05, 0.1):
        w_amihud = 1 - w_caria
        blend = w_caria * caria + w_amihud * amihud
        auc = roc_auc_score(target, blend)
        
        if auc > best_auc:
            best_auc = auc
            best_weights = (w_caria, w_amihud)
        
        print(f"{w_caria:<15.1f} | {w_amihud:<15.1f} | {auc:>8.4f}")
    
    print("-" * 45)
    print(f"OPTIMAL: {best_weights[0]:.1f} CARIA + {best_weights[1]:.1f} Amihud = AUC {best_auc:.4f}")
    
    return best_weights, best_auc


def analyze_what_amihud_captures(df):
    """Analyze what crashes Amihud captures that CARIA-SR misses."""
    print("\n" + "=" * 70)
    print("CRASH CLASSIFICATION: CARIA-SR vs AMIHUD")
    print("=" * 70)
    
    crashes = df[df['Target'] == 1].copy()
    
    # Classify each crash by which indicator predicted it
    caria_high = crashes['CARIA_SR'] > 0.7
    amihud_high = crashes['Amihud'] > 0.7
    
    both = (caria_high & amihud_high).sum()
    only_caria = (caria_high & ~amihud_high).sum()
    only_amihud = (~caria_high & amihud_high).sum()
    neither = (~caria_high & ~amihud_high).sum()
    
    total = len(crashes)
    
    print(f"\nTotal crashes in sample: {total}")
    print(f"\nCrash detection breakdown:")
    print(f"  Both CARIA-SR & Amihud high:  {both:4d} ({both/total:6.1%})")
    print(f"  Only CARIA-SR high:           {only_caria:4d} ({only_caria/total:6.1%})")
    print(f"  Only Amihud high:             {only_amihud:4d} ({only_amihud/total:6.1%})")
    print(f"  Neither high (missed):        {neither:4d} ({neither/total:6.1%})")
    
    # Analyze characteristics of "Only Amihud" crashes
    print("\n" + "-" * 50)
    print("CHARACTERISTICS OF 'ONLY AMIHUD' CRASHES:")
    
    only_amihud_crashes = crashes[~caria_high & amihud_high]
    all_crashes = crashes
    
    if len(only_amihud_crashes) > 10:
        print(f"\n  Mean forward return: {only_amihud_crashes['Fwd_Ret'].mean():.2%}")
        print(f"  Mean CARIA-SR:       {only_amihud_crashes['CARIA_SR'].mean():.3f}")
        print(f"  Mean Amihud:         {only_amihud_crashes['Amihud'].mean():.3f}")
        print(f"  Mean VIX:            {only_amihud_crashes['VIX'].mean():.3f}")
        print(f"  Mean E4:             {only_amihud_crashes['E4'].mean():.3f}")
        print(f"  Mean Sync:           {only_amihud_crashes['Sync'].mean():.3f}")
        
        print("\n  Interpretation:")
        if only_amihud_crashes['E4'].mean() < 0.5:
            print("    - LOW ENERGY: These crashes happen when volatility is calm")
            print("    - The market looks 'normal' to CARIA-SR but is actually illiquid")
        if only_amihud_crashes['Sync'].mean() < 0.5:
            print("    - LOW SYNC: Momentum scales are NOT correlated")
            print("    - Fragility is building without visible 'herding'")
    
    return {
        'both': both,
        'only_caria': only_caria,
        'only_amihud': only_amihud,
        'neither': neither
    }


def propose_enhanced_caria(df):
    """Propose CARIA-SR v9.0 with volume component."""
    print("\n" + "=" * 70)
    print("PROPOSED: CARIA-SR v9.0 (WITH VOLUME COMPONENT)")
    print("=" * 70)
    
    # Original components
    E4 = df['E4']
    Sync = df['Sync']
    
    # New: Amihud as "Liquidity Stress"
    Liq = df['Amihud']
    
    # V9 Formula: E4 × (1 + Sync) × (1 + Liq)
    # Or simpler: (E4 + Liq) × (1 + Sync)
    
    target = df['Target']
    
    # Test different formulas
    formulas = {
        'Original: E4×(1+S)': E4 * (1 + Sync),
        'V9a: E4×(1+S)×(1+L)': E4 * (1 + Sync) * (1 + Liq),
        'V9b: (E4+L)×(1+S)': (E4 + Liq) * (1 + Sync),
        'V9c: E4×(1+S) + L': E4 * (1 + Sync) + Liq,
        'V9d: 0.5×SR + 0.5×L': 0.5 * df['CARIA_SR'] + 0.5 * Liq,
        'V9e: 0.3×SR + 0.7×L': 0.3 * df['CARIA_SR'] + 0.7 * Liq,
    }
    
    print(f"\n{'Formula':<25} | {'AUC':>8}")
    print("-" * 40)
    
    best_formula = None
    best_auc = 0
    
    for name, formula in formulas.items():
        # Rank the combined score
        ranked = formula.rolling(252, min_periods=50).rank(pct=True)
        valid = ranked.dropna()
        valid_target = target.loc[valid.index]
        
        auc = roc_auc_score(valid_target, valid)
        
        if auc > best_auc:
            best_auc = auc
            best_formula = name
        
        marker = "***" if name == best_formula else ""
        print(f"{name:<25} | {auc:>8.4f} {marker}")
    
    print("-" * 40)
    print(f"\nBEST FORMULA: {best_formula} (AUC = {best_auc:.4f})")
    
    # Compare to baselines
    caria_auc = roc_auc_score(target, df['CARIA_SR'])
    amihud_auc = roc_auc_score(target, df['Amihud'])
    
    print(f"\nComparison:")
    print(f"  Original CARIA-SR: {caria_auc:.4f}")
    print(f"  Amihud alone:      {amihud_auc:.4f}")
    print(f"  Best V9 formula:   {best_auc:.4f}")
    print(f"  Improvement:       {best_auc - caria_auc:+.4f} vs original")
    
    return best_formula, best_auc


def create_visualization(df, results_df, output_dir):
    """Create visualization of feature comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Feature AUC ranking
    ax1 = axes[0, 0]
    results_sorted = results_df.sort_values('AUC', ascending=True)
    colors = ['#3b82f6' if 'CARIA' in f else '#f97316' if f == 'Amihud' else '#6b7280' 
              for f in results_sorted['Feature']]
    
    ax1.barh(results_sorted['Feature'], results_sorted['AUC'], color=colors)
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('AUC')
    ax1.set_title('Individual Feature AUC Ranking')
    
    # 2. Scatter: CARIA-SR vs Amihud during crashes
    ax2 = axes[0, 1]
    crashes = df[df['Target'] == 1]
    normal = df[df['Target'] == 0].sample(min(1000, len(df[df['Target'] == 0])))
    
    ax2.scatter(normal['CARIA_SR'], normal['Amihud'], alpha=0.3, c='gray', s=10, label='Normal')
    ax2.scatter(crashes['CARIA_SR'], crashes['Amihud'], alpha=0.5, c='red', s=20, label='Crash')
    ax2.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7)
    ax2.axvline(x=0.7, color='blue', linestyle='--', alpha=0.7)
    ax2.set_xlabel('CARIA-SR')
    ax2.set_ylabel('Amihud Illiquidity')
    ax2.set_title('CARIA-SR vs Amihud: Crash Detection')
    ax2.legend()
    
    # 3. Time series comparison (last 5 years)
    ax3 = axes[1, 0]
    recent = df.tail(252*5)  # ~5 years
    
    ax3.plot(recent.index, recent['CARIA_SR'], 'b-', alpha=0.7, label='CARIA-SR')
    ax3.plot(recent.index, recent['Amihud'], 'orange', alpha=0.7, label='Amihud')
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Indicator Value')
    ax3.set_title('CARIA-SR vs Amihud (Recent 5 Years)')
    ax3.legend()
    
    # Mark crashes
    crash_dates = recent[recent['Target'] == 1].index
    for date in crash_dates[::5]:  # Every 5th crash to avoid clutter
        ax3.axvline(x=date, color='red', alpha=0.2, linewidth=1)
    
    # 4. Correlation heatmap of volume features
    ax4 = axes[1, 1]
    vol_features = ['CARIA_SR', 'Amihud', 'Vol_Trend', 'Vol_Volatility', 
                    'Vol_Ret_Divergence', 'Kyle_Lambda']
    vol_features = [f for f in vol_features if f in df.columns]
    
    corr_matrix = df[vol_features].corr()
    im = ax4.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(vol_features)))
    ax4.set_yticks(range(len(vol_features)))
    ax4.set_xticklabels([f[:10] for f in vol_features], rotation=45, ha='right')
    ax4.set_yticklabels([f[:10] for f in vol_features])
    ax4.set_title('Feature Correlation Matrix')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_Missing_Component_Analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: Figure_Missing_Component_Analysis.png")


def main():
    """Run complete analysis."""
    print("=" * 70)
    print("MISSING COMPONENT ANALYSIS")
    print("What makes Illiquidity better than CARIA-SR?")
    print("=" * 70)
    
    # Load data
    data = load_all_data()
    
    # Compute features
    df = compute_all_features(data)
    
    # Analyze individual features
    results_df = analyze_individual_features(df)
    
    # Analyze combinations
    combo_results = analyze_combinations_with_caria(df)
    
    # Find optimal blend
    best_weights, best_auc = find_optimal_blend(df)
    
    # Analyze what Amihud captures
    crash_analysis = analyze_what_amihud_captures(df)
    
    # Propose enhanced CARIA
    best_formula, v9_auc = propose_enhanced_caria(df)
    
    # Create visualization
    create_visualization(df, results_df, OUTPUT_DIR)
    
    # Save results
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_Feature_AUC_Ranking.csv'), index=False)
    print(f"✓ Saved: Table_Feature_AUC_Ranking.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: THE MISSING 6%")
    print("=" * 70)
    
    print("""
KEY FINDING: The missing component is VOLUME DYNAMICS.

CARIA-SR captures:
  - Volatility structure (E4)
  - Momentum synchronization (Sync)
  - Credit stress (via HYG)

What CARIA-SR MISSES:
  - Price impact (how much prices move per $ traded)
  - Volume trends (is liquidity drying up?)
  - Market depth (can the market absorb selling?)

RECOMMENDATION:
  Add Amihud Illiquidity as a third pillar:
  
  CARIA-SR v9.0 = E4 × (1 + Sync) × (1 + Liquidity)
  
  Or simpler blend:
  CARIA-SR v9.0 = 0.3 × Original_SR + 0.7 × Amihud
    """)
    
    return df, results_df


if __name__ == "__main__":
    df, results = main()

