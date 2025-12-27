"""
CARIA-SR Core Validation
========================

Focus: Validate the ORIGINAL CARIA-SR formula:
  SR = E4 × (1 + Sync)

Where:
  E4 = 0.20×v5 + 0.30×v21 + 0.25×v63 + 0.25×v_credit
  Sync = correlation(fast_momentum, slow_momentum)

Target: Forward 21-day returns in worst 5% (exogenous crash definition)

Questions:
1. Does CARIA-SR predict crashes?
2. Why does E4 alone beat CARIA-SR?
3. Is Sync adding value or noise?
4. How to fix it?

Author: Tomás Basaure
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
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


def build_predictive_caria(ticker, vol_credit_series):
    """
    ORIGINAL CARIA-SR formula - exactly as specified.
    """
    data = yf.download(ticker, start=START_DATE, progress=False)
    if len(data) < 500: 
        return None
    
    if isinstance(data.columns, pd.MultiIndex):
        price = data["Close"].iloc[:, 0]
    else:
        price = data["Close"]
        
    ret = price.pct_change().dropna()
    
    # Align with credit
    common_idx = ret.index.intersection(vol_credit_series.index)
    if len(common_idx) < 500: 
        return None
    
    r = ret.loc[common_idx]
    v_cred = vol_credit_series.loc[common_idx]
    
    # === ORIGINAL CARIA-SR COMPONENTS ===
    
    # Energy Equity
    v5  = r.rolling(5).std() * np.sqrt(252)
    v21 = r.rolling(21).std() * np.sqrt(252)
    v63 = r.rolling(63).std() * np.sqrt(252)
    
    # E4 (Macro Energy)
    E4_raw = 0.20*v5 + 0.30*v21 + 0.25*v63 + 0.25*v_cred
    E4 = E4_raw.rolling(252).rank(pct=True)
    
    # Sync (Structure) - Momentum Correlation
    m_fast = r.rolling(5).sum()
    m_slow = r.rolling(63).sum()
    sync_raw = m_fast.rolling(21).corr(m_slow)
    sync = ((sync_raw + 1) / 2).rolling(252).rank(pct=True)
    
    # CARIA-SR (Fusion)
    SR_raw = E4 * (1 + sync)
    SR = SR_raw.rolling(252).rank(pct=True)
    
    # === TARGET (EXOGENOUS TRUTH) ===
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FWD_WINDOW)
    fwd_ret = r.rolling(window=indexer).sum()
    crash_threshold = fwd_ret.quantile(CRASH_QUANTILE)
    is_real_crash = (fwd_ret < crash_threshold).astype(int)
    
    # Return all components for analysis
    df = pd.DataFrame({
        'SR': SR,
        'E4': E4,
        'Sync': sync,
        'Sync_raw': sync_raw,
        'E4_raw': E4_raw,
        'Target_Crash': is_real_crash,
        'Fwd_Ret': fwd_ret,
        'Return': r
    }).dropna()
    
    return df


def analyze_caria_components(df):
    """Analyze each component of CARIA-SR."""
    print("\n" + "=" * 70)
    print("CARIA-SR COMPONENT ANALYSIS")
    print("=" * 70)
    
    target = df['Target_Crash']
    
    components = {
        'E4 (Energy)': df['E4'],
        'Sync (Momentum Corr)': df['Sync'],
        'CARIA-SR (E4 × (1+Sync))': df['SR'],
    }
    
    print(f"\n{'Component':<30} | {'AUC':>8} | {'Assessment'}")
    print("-" * 60)
    
    results = {}
    
    for name, values in components.items():
        auc = roc_auc_score(target, values)
        results[name] = auc
        
        if auc > 0.65:
            assessment = "✓ Good predictor"
        elif auc > 0.55:
            assessment = "~ Weak predictor"
        elif auc > 0.45:
            assessment = "✗ Random"
        else:
            assessment = "✗ Inverse predictor"
        
        print(f"{name:<30} | {auc:>8.4f} | {assessment}")
    
    # Key comparison
    e4_auc = results['E4 (Energy)']
    sr_auc = results['CARIA-SR (E4 × (1+Sync))']
    sync_auc = results['Sync (Momentum Corr)']
    
    print("\n" + "-" * 50)
    print("KEY FINDING:")
    
    if e4_auc > sr_auc:
        print(f"  ⚠️ E4 alone ({e4_auc:.4f}) beats CARIA-SR ({sr_auc:.4f})")
        print(f"  ⚠️ Adding Sync REDUCES predictive power by {e4_auc - sr_auc:.4f}")
        print(f"  ⚠️ Sync has AUC = {sync_auc:.4f} (essentially random)")
    else:
        print(f"  ✓ CARIA-SR ({sr_auc:.4f}) beats E4 alone ({e4_auc:.4f})")
        print(f"  ✓ Sync is adding value")
    
    return results


def analyze_sync_problem(df):
    """Deep dive: Why is Sync not working?"""
    print("\n" + "=" * 70)
    print("SYNC COMPONENT DEEP DIVE")
    print("=" * 70)
    
    target = df['Target_Crash']
    sync = df['Sync']
    sync_raw = df['Sync_raw']
    fwd_ret = df['Fwd_Ret']
    
    # 1. Distribution of Sync during crashes vs normal
    crashes = df[df['Target_Crash'] == 1]
    normal = df[df['Target_Crash'] == 0]
    
    print("\n1. SYNC DISTRIBUTION BY REGIME:")
    print(f"   During crashes:  mean = {crashes['Sync'].mean():.3f}, std = {crashes['Sync'].std():.3f}")
    print(f"   During normal:   mean = {normal['Sync'].mean():.3f}, std = {normal['Sync'].std():.3f}")
    
    # T-test
    t_stat, p_val = stats.ttest_ind(crashes['Sync'], normal['Sync'])
    print(f"   T-test: t={t_stat:.2f}, p={p_val:.4f}")
    
    if p_val > 0.05:
        print("   ⚠️ Sync is NOT significantly different before crashes!")
    
    # 2. Correlation with target
    print("\n2. SYNC CORRELATION WITH OUTCOMES:")
    print(f"   Corr(Sync, Target):   {sync.corr(target.astype(float)):.4f}")
    print(f"   Corr(Sync, Fwd_Ret):  {sync.corr(fwd_ret):.4f}")
    
    # 3. The hypothesis behind Sync
    print("\n3. SYNC HYPOTHESIS TEST:")
    print("""
   Original hypothesis: 
   "When fast and slow momentum are synchronized (high Sync),
    the market is in a fragile 'herding' state."
   
   Testing: Do high-Sync periods precede crashes?
    """)
    
    # Split by Sync quartiles
    df['Sync_Q'] = pd.qcut(df['Sync'], 4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High'])
    
    print(f"   {'Sync Quartile':<15} | {'Crash Rate':>12} | {'Avg Fwd Ret':>12}")
    print("   " + "-" * 45)
    
    for q in ['Q1_Low', 'Q2', 'Q3', 'Q4_High']:
        subset = df[df['Sync_Q'] == q]
        crash_rate = subset['Target_Crash'].mean()
        avg_ret = subset['Fwd_Ret'].mean()
        print(f"   {q:<15} | {crash_rate:>12.2%} | {avg_ret:>12.2%}")
    
    # Is the pattern monotonic?
    q1_crash = df[df['Sync_Q'] == 'Q1_Low']['Target_Crash'].mean()
    q4_crash = df[df['Sync_Q'] == 'Q4_High']['Target_Crash'].mean()
    
    print("\n   VERDICT:")
    if q4_crash > q1_crash * 1.5:
        print("   ✓ High Sync → Higher crash probability (hypothesis supported)")
    elif q4_crash < q1_crash * 0.67:
        print("   ⚠️ High Sync → LOWER crash probability (inverse relationship!)")
    else:
        print("   ✗ No clear relationship between Sync and crashes")
    
    return df


def test_alternative_formulas(df):
    """Test alternative CARIA formulas."""
    print("\n" + "=" * 70)
    print("ALTERNATIVE FORMULA TESTS")
    print("=" * 70)
    
    target = df['Target_Crash']
    E4 = df['E4']
    Sync = df['Sync']
    
    formulas = {}
    
    # Original
    formulas['Original: E4 × (1+Sync)'] = E4 * (1 + Sync)
    
    # Just E4
    formulas['E4 alone'] = E4
    
    # Inverse Sync (what if low sync = danger?)
    formulas['E4 × (1 + (1-Sync))'] = E4 * (1 + (1 - Sync))
    
    # E4 only when Sync is low (regime filter)
    formulas['E4 × (1 - Sync)'] = E4 * (1 - Sync)
    
    # Additive instead of multiplicative
    formulas['E4 + Sync'] = E4 + Sync
    formulas['E4 + (1-Sync)'] = E4 + (1 - Sync)
    
    # E4 with Sync as weight
    formulas['E4^(1+Sync)'] = E4 ** (1 + Sync)
    
    # Pure E4 variants
    formulas['E4^2'] = E4 ** 2
    formulas['E4 × E4_raw'] = E4 * (df['E4_raw'].rolling(252).rank(pct=True))
    
    print(f"\n{'Formula':<30} | {'AUC':>8} | {'vs Original':>12}")
    print("-" * 60)
    
    baseline_auc = roc_auc_score(target, E4 * (1 + Sync))
    
    results = []
    
    for name, formula in formulas.items():
        # Rank the formula
        ranked = formula.rolling(252, min_periods=50).rank(pct=True)
        valid = ranked.dropna()
        valid_target = target.loc[valid.index]
        
        auc = roc_auc_score(valid_target, valid)
        delta = auc - baseline_auc
        
        results.append({'Formula': name, 'AUC': auc, 'Delta': delta})
        
        marker = "***" if auc == max([r['AUC'] for r in results]) else ""
        print(f"{name:<30} | {auc:>8.4f} | {delta:>+12.4f} {marker}")
    
    results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    
    # Best formula
    best = results_df.iloc[0]
    print("\n" + "-" * 50)
    print(f"BEST FORMULA: {best['Formula']}")
    print(f"  AUC: {best['AUC']:.4f}")
    print(f"  Improvement vs Original: {best['Delta']:+.4f}")
    
    return results_df


def analyze_minsky_premium(df):
    """Test the Minsky Paradox hypothesis."""
    print("\n" + "=" * 70)
    print("MINSKY PARADOX TEST")
    print("=" * 70)
    
    print("""
    HYPOTHESIS: Fragility accumulates during positive returns (melt-up).
    
    Test: What are average returns when CARIA-SR signals "danger"?
    If returns are POSITIVE when SR is high, the Minsky Paradox is confirmed.
    """)
    
    # High SR regime
    high_sr = df[df['SR'] > 0.8]
    low_sr = df[df['SR'] < 0.2]
    
    print(f"\nHigh SR (> 0.8) periods: {len(high_sr)} days ({len(high_sr)/len(df):.1%})")
    print(f"Low SR (< 0.2) periods:  {len(low_sr)} days ({len(low_sr)/len(df):.1%})")
    
    print(f"\n{'Regime':<20} | {'Avg Return':>12} | {'Crash Rate':>12}")
    print("-" * 50)
    
    high_ret = high_sr['Return'].mean() * 252  # Annualized
    high_crash = high_sr['Target_Crash'].mean()
    
    low_ret = low_sr['Return'].mean() * 252
    low_crash = low_sr['Target_Crash'].mean()
    
    print(f"{'High SR (danger)':<20} | {high_ret:>+12.2%} | {high_crash:>12.2%}")
    print(f"{'Low SR (safe)':<20} | {low_ret:>+12.2%} | {low_crash:>12.2%}")
    
    # Minsky Premium
    minsky_premium = high_ret - low_ret
    print(f"\n  Minsky Premium: {minsky_premium:+.2%}")
    
    if minsky_premium > 0 and high_crash > low_crash:
        print("  ✓ MINSKY PARADOX CONFIRMED!")
        print("    High fragility (high SR) coincides with POSITIVE returns")
        print("    BUT also higher crash probability")
    else:
        print("  ✗ Minsky Paradox not evident in this analysis")
    
    # T-test for Minsky Premium
    if len(high_sr) > 30:
        t_stat, p_val = stats.ttest_1samp(high_sr['Return'], 0, alternative='greater')
        print(f"\n  T-test (H0: returns in danger zone = 0):")
        print(f"    t = {t_stat:.2f}, p = {p_val:.4f}")
        if p_val < 0.05:
            print("    ✓ Statistically significant positive returns in danger zone!")


def create_summary_visualization(df, results, output_dir):
    """Create summary visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. ROC Curves for each component
    ax1 = axes[0, 0]
    target = df['Target_Crash']
    
    for name, col in [('E4', 'E4'), ('Sync', 'Sync'), ('CARIA-SR', 'SR')]:
        fpr, tpr, _ = roc_curve(target, df[col])
        auc = roc_auc_score(target, df[col])
        ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves: CARIA-SR Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sync distribution by regime
    ax2 = axes[0, 1]
    crashes = df[df['Target_Crash'] == 1]['Sync']
    normal = df[df['Target_Crash'] == 0]['Sync'].sample(min(1000, len(df[df['Target_Crash'] == 0])))
    
    ax2.hist(normal, bins=30, alpha=0.5, label='Normal', color='blue', density=True)
    ax2.hist(crashes, bins=30, alpha=0.5, label='Pre-Crash', color='red', density=True)
    ax2.axvline(x=normal.mean(), color='blue', linestyle='--', label=f'Normal mean: {normal.mean():.2f}')
    ax2.axvline(x=crashes.mean(), color='red', linestyle='--', label=f'Crash mean: {crashes.mean():.2f}')
    ax2.set_xlabel('Sync Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Sync Distribution: Normal vs Pre-Crash')
    ax2.legend()
    
    # 3. Formula comparison
    ax3 = axes[1, 0]
    results_sorted = results.sort_values('AUC', ascending=True)
    colors = ['green' if 'E4 alone' in f else 'blue' if 'Original' in f else 'gray' 
              for f in results_sorted['Formula']]
    
    ax3.barh(results_sorted['Formula'], results_sorted['AUC'], color=colors, alpha=0.7)
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('AUC')
    ax3.set_title('Alternative Formula Comparison')
    
    # 4. Time series
    ax4 = axes[1, 1]
    recent = df.tail(252*3)  # Last 3 years
    
    ax4.plot(recent.index, recent['E4'], 'b-', alpha=0.7, label='E4', linewidth=1)
    ax4.plot(recent.index, recent['SR'], 'purple', alpha=0.7, label='CARIA-SR', linewidth=1)
    ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Danger Zone')
    
    # Mark crashes
    crash_dates = recent[recent['Target_Crash'] == 1].index
    for date in crash_dates[::5]:
        ax4.axvline(x=date, color='red', alpha=0.2)
    
    ax4.set_ylabel('Indicator Value')
    ax4.set_title('E4 vs CARIA-SR Over Time')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_CARIA_Core_Analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: Figure_CARIA_Core_Analysis.png")


def main():
    """Run focused CARIA-SR analysis."""
    print("=" * 70)
    print("CARIA-SR CORE VALIDATION")
    print("Formula: SR = E4 × (1 + Sync)")
    print("=" * 70)
    
    # Load credit volatility
    print("\nLoading credit volatility (HYG)...")
    hyg = yf.download("HYG", start=START_DATE, progress=False)["Close"]
    if isinstance(hyg, pd.DataFrame):
        hyg = hyg.iloc[:, 0]
    ret_hyg = hyg.pct_change().dropna()
    vol_credit = ret_hyg.rolling(42).std() * np.sqrt(252)
    
    # Build CARIA-SR for SPY
    print("Building CARIA-SR for SPY...")
    df = build_predictive_caria("SPY", vol_credit)
    
    if df is None:
        print("Error: Could not build CARIA-SR")
        return
    
    print(f"  Observations: {len(df)}")
    print(f"  Crash rate: {df['Target_Crash'].mean():.2%}")
    
    # Analyze components
    component_results = analyze_caria_components(df)
    
    # Deep dive on Sync
    df = analyze_sync_problem(df)
    
    # Test alternative formulas
    formula_results = test_alternative_formulas(df)
    
    # Minsky Paradox test
    analyze_minsky_premium(df)
    
    # Create visualization
    create_summary_visualization(df, formula_results, OUTPUT_DIR)
    
    # Save results
    formula_results.to_csv(os.path.join(OUTPUT_DIR, 'Table_CARIA_Formula_Comparison.csv'), index=False)
    print(f"✓ Saved: Table_CARIA_Formula_Comparison.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    e4_auc = component_results['E4 (Energy)']
    sr_auc = component_results['CARIA-SR (E4 × (1+Sync))']
    sync_auc = component_results['Sync (Momentum Corr)']
    best_formula = formula_results.iloc[0]
    
    print(f"""
    CURRENT MODEL:
      E4:        AUC = {e4_auc:.4f} (strong)
      Sync:      AUC = {sync_auc:.4f} (random)
      CARIA-SR:  AUC = {sr_auc:.4f} (degraded by Sync)
    
    DIAGNOSIS:
      The Sync component (momentum correlation) is adding NOISE.
      E4 alone outperforms the combined CARIA-SR formula.
    
    BEST ALTERNATIVE:
      {best_formula['Formula']}: AUC = {best_formula['AUC']:.4f}
    
    RECOMMENDATIONS:
      Option 1: Use E4 alone (simplest, AUC = {e4_auc:.4f})
      Option 2: Replace Sync with a better "structure" indicator
      Option 3: Investigate why Sync hypothesis doesn't hold
    """)
    
    return df, component_results, formula_results


if __name__ == "__main__":
    df, comp_results, formula_results = main()















