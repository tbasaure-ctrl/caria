"""
Test Amihud as Veto + Correlation Analysis
==========================================

Part 1: Amihud as VETO Signal
-----------------------------
Logic:
  - If Amihud > THRESHOLD → VETO (stay out, regime = fragile)
  - If Amihud < THRESHOLD → Trust CARIA-SR signal

Part 2: Correlation Analysis
----------------------------
Question: Are E4, Amihud, and Kyle's Lambda redundant?
  - If correlation > 0.85 → Redundant, pick one
  - If correlation < 0.60 → Complementary, can combine

Author: Tomás Basaure
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
START_DATE = "2005-01-01"
FWD_WINDOW = 21
CRASH_QUANTILE = 0.05


def load_data():
    """Load SPY and credit data."""
    print("Loading data...")
    
    spy = yf.download("SPY", start=START_DATE, progress=False)
    hyg = yf.download("HYG", start=START_DATE, progress=False)["Close"]
    
    if isinstance(spy.columns, pd.MultiIndex):
        p = spy["Close"].iloc[:, 0]
        v = spy["Volume"].iloc[:, 0]
    else:
        p = spy["Close"]
        v = spy["Volume"]
    
    if isinstance(hyg, pd.DataFrame):
        hyg = hyg.iloc[:, 0]
    
    r = p.pct_change()
    r_hyg = hyg.pct_change()
    
    common = r.index.intersection(r_hyg.index).intersection(v.index)
    
    return r.loc[common], p.loc[common], v.loc[common], r_hyg.loc[common]


def compute_features(r, p, v, r_hyg):
    """Compute all features."""
    print("Computing features...")
    
    features = {}
    
    # === E4 (Volatility Structure) ===
    v5 = r.rolling(5).std() * np.sqrt(252)
    v21 = r.rolling(21).std() * np.sqrt(252)
    v63 = r.rolling(63).std() * np.sqrt(252)
    v_cred = r_hyg.rolling(42).std() * np.sqrt(252)
    
    E4_raw = 0.20 * v5 + 0.30 * v21 + 0.25 * v63 + 0.25 * v_cred
    E4 = E4_raw.rolling(252).rank(pct=True)
    features['E4'] = E4
    
    # === Sync ===
    m_fast = r.rolling(5).sum()
    m_slow = r.rolling(63).sum()
    sync_raw = m_fast.rolling(21).corr(m_slow)
    sync = ((sync_raw + 1) / 2).rolling(252).rank(pct=True)
    features['Sync'] = sync
    
    # === CARIA-SR ===
    SR_raw = E4 * (1 + sync)
    SR = SR_raw.rolling(252).rank(pct=True)
    features['CARIA_SR'] = SR
    
    # === Amihud Illiquidity ===
    dollar_vol = p * v
    amihud_raw = np.abs(r) / (dollar_vol + 1e-9)
    amihud = amihud_raw.rolling(21).mean().rolling(252).rank(pct=True)
    features['Amihud'] = amihud
    
    # === Kyle's Lambda (Price Impact) ===
    kyle_raw = np.abs(p.diff()) / (v + 1e-9)
    kyle = kyle_raw.rolling(21).mean().rolling(252).rank(pct=True)
    features['Kyle_Lambda'] = kyle
    
    # === Raw versions for correlation ===
    features['E4_raw'] = E4_raw
    features['Amihud_raw'] = amihud_raw.rolling(21).mean()
    features['Kyle_raw'] = kyle_raw.rolling(21).mean()
    
    # === Target ===
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FWD_WINDOW)
    fwd_ret = r.rolling(window=indexer).sum()
    crash_threshold = fwd_ret.quantile(CRASH_QUANTILE)
    is_crash = (fwd_ret < crash_threshold).astype(int)
    
    df = pd.DataFrame(features)
    df['Target'] = is_crash
    df['Fwd_Ret'] = fwd_ret
    df['Return'] = r
    df = df.dropna()
    
    print(f"  Observations: {len(df)}")
    return df


# =============================================================================
# PART 1: AMIHUD AS VETO
# =============================================================================

def test_veto_system(df):
    """Test Amihud as a veto signal for CARIA-SR."""
    print("\n" + "=" * 70)
    print("PART 1: AMIHUD AS VETO SYSTEM")
    print("=" * 70)
    
    target = df['Target']
    caria = df['CARIA_SR']
    amihud = df['Amihud']
    fwd_ret = df['Fwd_Ret']
    
    # Test different veto thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\n{'Veto Threshold':<15} | {'Crashes Avoided':<18} | {'False Vetoes':<15} | {'Net Benefit'}")
    print("-" * 75)
    
    results = []
    
    for thresh in thresholds:
        # Veto regime: Amihud > threshold
        veto_regime = amihud > thresh
        normal_regime = ~veto_regime
        
        # In veto regime, we stay out (avoid all crashes AND all gains)
        # In normal regime, we follow CARIA-SR
        
        # Crashes in each regime
        crashes_in_veto = (target == 1) & veto_regime
        crashes_in_normal = (target == 1) & normal_regime
        
        # Returns in each regime
        veto_crash_avoided = fwd_ret[crashes_in_veto].sum()  # Losses avoided
        veto_gain_missed = fwd_ret[veto_regime & (target == 0)].sum()  # Gains missed
        
        normal_crash_hit = fwd_ret[crashes_in_normal].sum()  # Still hit these
        normal_gain_captured = fwd_ret[normal_regime & (target == 0)].sum()
        
        # Metrics
        pct_vetoed = veto_regime.mean()
        crashes_avoided = crashes_in_veto.sum()
        total_crashes = target.sum()
        crashes_still_hit = crashes_in_normal.sum()
        
        # Net benefit: Losses avoided - Gains missed
        net_benefit = -veto_crash_avoided - veto_gain_missed  # Negative because crashes have negative returns
        
        results.append({
            'Threshold': thresh,
            'Pct_Vetoed': pct_vetoed,
            'Crashes_Avoided': crashes_avoided,
            'Crashes_Still_Hit': crashes_still_hit,
            'Crash_Avoid_Rate': crashes_avoided / total_crashes if total_crashes > 0 else 0,
            'Veto_Crash_Return': veto_crash_avoided,
            'Veto_Gain_Missed': veto_gain_missed,
            'Net_Benefit': net_benefit
        })
        
        print(f"Amihud > {thresh:<8.1f} | {crashes_avoided:3d}/{total_crashes} ({crashes_avoided/total_crashes:.1%}) | "
              f"{pct_vetoed:.1%} time out     | {net_benefit:+.1%}")
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold
    print("\n" + "-" * 50)
    print("DETAILED ANALYSIS AT EACH THRESHOLD:")
    
    for _, row in results_df.iterrows():
        thresh = row['Threshold']
        print(f"\n  Threshold = {thresh}:")
        print(f"    Time in 'Veto' regime: {row['Pct_Vetoed']:.1%}")
        print(f"    Crashes avoided: {row['Crashes_Avoided']:.0f} ({row['Crash_Avoid_Rate']:.1%})")
        print(f"    Crashes still hit: {row['Crashes_Still_Hit']:.0f}")
        print(f"    Return from avoided crashes: {row['Veto_Crash_Return']:.1%} (would have lost)")
        print(f"    Return missed in veto: {row['Veto_Gain_Missed']:.1%} (opportunity cost)")
    
    return results_df


def test_veto_auc(df):
    """Test AUC with veto system."""
    print("\n" + "-" * 50)
    print("AUC COMPARISON: VETO vs BLEND")
    
    target = df['Target']
    caria = df['CARIA_SR']
    amihud = df['Amihud']
    
    # Baseline AUCs
    auc_caria = roc_auc_score(target, caria)
    auc_amihud = roc_auc_score(target, amihud)
    
    print(f"\n  CARIA-SR alone:  {auc_caria:.4f}")
    print(f"  Amihud alone:    {auc_amihud:.4f}")
    
    # Veto-based score: Use Amihud when high, CARIA otherwise
    for thresh in [0.7, 0.8, 0.9]:
        # Veto score: max(CARIA, Amihud) when Amihud > thresh, else CARIA
        veto_score = np.where(amihud > thresh, 
                              np.maximum(caria, amihud),  # In danger zone, be conservative
                              caria)  # In safe zone, use CARIA
        auc_veto = roc_auc_score(target, veto_score)
        print(f"  Veto @ {thresh}:       {auc_veto:.4f}")
    
    # Simple blend for comparison
    blend = 0.5 * caria + 0.5 * amihud
    auc_blend = roc_auc_score(target, blend)
    print(f"  50/50 Blend:     {auc_blend:.4f}")
    
    # Max of both
    max_both = np.maximum(caria, amihud)
    auc_max = roc_auc_score(target, max_both)
    print(f"  Max(CARIA,Amh):  {auc_max:.4f}")


def analyze_veto_regimes(df):
    """Analyze characteristics of veto vs normal regimes."""
    print("\n" + "-" * 50)
    print("REGIME CHARACTERISTICS (Amihud > 0.8)")
    
    veto_regime = df['Amihud'] > 0.8
    
    print(f"\n  {'Metric':<25} | {'Veto Regime':>12} | {'Normal Regime':>12}")
    print("  " + "-" * 55)
    
    metrics = {
        'Count': (veto_regime.sum(), (~veto_regime).sum()),
        'Crash Rate': (df.loc[veto_regime, 'Target'].mean(), df.loc[~veto_regime, 'Target'].mean()),
        'Avg Fwd Return': (df.loc[veto_regime, 'Fwd_Ret'].mean(), df.loc[~veto_regime, 'Fwd_Ret'].mean()),
        'Std Fwd Return': (df.loc[veto_regime, 'Fwd_Ret'].std(), df.loc[~veto_regime, 'Fwd_Ret'].std()),
        'Avg E4': (df.loc[veto_regime, 'E4'].mean(), df.loc[~veto_regime, 'E4'].mean()),
        'Avg CARIA-SR': (df.loc[veto_regime, 'CARIA_SR'].mean(), df.loc[~veto_regime, 'CARIA_SR'].mean()),
    }
    
    for name, (veto_val, normal_val) in metrics.items():
        if 'Rate' in name or 'Return' in name:
            print(f"  {name:<25} | {veto_val:>12.2%} | {normal_val:>12.2%}")
        else:
            print(f"  {name:<25} | {veto_val:>12.2f} | {normal_val:>12.2f}")


# =============================================================================
# PART 2: CORRELATION ANALYSIS
# =============================================================================

def correlation_analysis(df):
    """Analyze correlation between E4, Amihud, and Kyle's Lambda."""
    print("\n" + "=" * 70)
    print("PART 2: CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Ranked versions (what we use)
    ranked_features = ['E4', 'Amihud', 'Kyle_Lambda', 'CARIA_SR', 'Sync']
    
    print("\n1. RANKED FEATURES (Percentile)")
    print("-" * 50)
    
    corr_ranked = df[ranked_features].corr()
    
    print("\nCorrelation Matrix:")
    print(corr_ranked.round(3).to_string())
    
    # Key correlations
    print("\n  Key Correlations:")
    print(f"    E4 vs Amihud:      {corr_ranked.loc['E4', 'Amihud']:.3f}")
    print(f"    E4 vs Kyle:        {corr_ranked.loc['E4', 'Kyle_Lambda']:.3f}")
    print(f"    Amihud vs Kyle:    {corr_ranked.loc['Amihud', 'Kyle_Lambda']:.3f}")
    print(f"    E4 vs CARIA-SR:    {corr_ranked.loc['E4', 'CARIA_SR']:.3f}")
    print(f"    Amihud vs CARIA:   {corr_ranked.loc['Amihud', 'CARIA_SR']:.3f}")
    
    # Raw versions
    raw_features = ['E4_raw', 'Amihud_raw', 'Kyle_raw']
    
    print("\n2. RAW FEATURES (Before ranking)")
    print("-" * 50)
    
    corr_raw = df[raw_features].corr()
    
    print("\nCorrelation Matrix:")
    print(corr_raw.round(3).to_string())
    
    # Interpretation
    print("\n" + "-" * 50)
    print("INTERPRETATION:")
    
    e4_amihud = corr_ranked.loc['E4', 'Amihud']
    e4_kyle = corr_ranked.loc['E4', 'Kyle_Lambda']
    amihud_kyle = corr_ranked.loc['Amihud', 'Kyle_Lambda']
    
    if e4_amihud > 0.85:
        print(f"\n  ⚠️  E4 and Amihud are HIGHLY correlated ({e4_amihud:.2f})")
        print("     → They measure the SAME risk. Pick ONE.")
        print("     → Recommendation: Use Amihud (simpler)")
    elif e4_amihud > 0.60:
        print(f"\n  ⚡ E4 and Amihud are MODERATELY correlated ({e4_amihud:.2f})")
        print("     → They share some signal but capture different aspects")
        print("     → Can combine, but diminishing returns")
    else:
        print(f"\n  ✓ E4 and Amihud are WEAKLY correlated ({e4_amihud:.2f})")
        print("     → They capture DIFFERENT risks!")
        print("     → Combining them creates 'Super-Fragility' detector")
    
    if amihud_kyle > 0.85:
        print(f"\n  ⚠️  Amihud and Kyle's Lambda are HIGHLY correlated ({amihud_kyle:.2f})")
        print("     → They're essentially the same measure")
        print("     → Use Amihud (computationally cheaper)")
    
    return corr_ranked, corr_raw


def create_visualizations(df, veto_results, corr_ranked, output_dir):
    """Create visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Veto Performance by Threshold
    ax1 = axes[0, 0]
    thresholds = veto_results['Threshold']
    crash_avoid = veto_results['Crash_Avoid_Rate'] * 100
    pct_vetoed = veto_results['Pct_Vetoed'] * 100
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, crash_avoid, width, label='Crashes Avoided %', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, pct_vetoed, width, label='Time in Veto %', color='red', alpha=0.7)
    
    ax1.set_xlabel('Amihud Threshold')
    ax1.set_ylabel('Percentage')
    ax1.set_title('Veto System Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{t:.1f}' for t in thresholds])
    ax1.legend()
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    # 2. Correlation Heatmap
    ax2 = axes[0, 1]
    features = ['E4', 'Amihud', 'Kyle_Lambda', 'CARIA_SR']
    corr_subset = corr_ranked.loc[features, features]
    
    im = ax2.imshow(corr_subset, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(features)))
    ax2.set_yticks(range(len(features)))
    ax2.set_xticklabels(features, rotation=45, ha='right')
    ax2.set_yticklabels(features)
    ax2.set_title('Feature Correlation Matrix')
    
    # Add correlation values
    for i in range(len(features)):
        for j in range(len(features)):
            text = ax2.text(j, i, f'{corr_subset.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10)
    
    plt.colorbar(im, ax=ax2)
    
    # 3. Scatter: E4 vs Amihud
    ax3 = axes[1, 0]
    crashes = df[df['Target'] == 1]
    normal = df[df['Target'] == 0].sample(min(1000, len(df[df['Target'] == 0])))
    
    ax3.scatter(normal['E4'], normal['Amihud'], alpha=0.2, c='gray', s=10, label='Normal')
    ax3.scatter(crashes['E4'], crashes['Amihud'], alpha=0.5, c='red', s=20, label='Crash')
    
    # Add correlation line
    z = np.polyfit(df['E4'], df['Amihud'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax3.plot(x_line, p(x_line), 'b--', alpha=0.5, label=f'r={corr_ranked.loc["E4", "Amihud"]:.2f}')
    
    ax3.set_xlabel('E4 (Volatility)')
    ax3.set_ylabel('Amihud (Illiquidity)')
    ax3.set_title('E4 vs Amihud: Are They Redundant?')
    ax3.legend()
    
    # 4. Time series: E4 vs Amihud divergence
    ax4 = axes[1, 1]
    recent = df.tail(252*3)  # Last 3 years
    
    ax4.plot(recent.index, recent['E4'], 'b-', alpha=0.7, label='E4')
    ax4.plot(recent.index, recent['Amihud'], 'orange', alpha=0.7, label='Amihud')
    
    # Highlight divergences (when one is high, other is low)
    divergence = np.abs(recent['E4'] - recent['Amihud'])
    high_div = divergence > 0.3
    for idx in recent.index[high_div]:
        ax4.axvline(x=idx, color='purple', alpha=0.1, linewidth=2)
    
    ax4.set_ylabel('Rank (0-1)')
    ax4.set_title('E4 vs Amihud Over Time (Purple = Divergence)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_Veto_Correlation_Analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: Figure_Veto_Correlation_Analysis.png")


def final_recommendation(corr_ranked, veto_results):
    """Provide final recommendation."""
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)
    
    e4_amihud = corr_ranked.loc['E4', 'Amihud']
    amihud_kyle = corr_ranked.loc['Amihud', 'Kyle_Lambda']
    
    print(f"""
CORRELATION FINDINGS:
  • E4 vs Amihud:   {e4_amihud:.3f}
  • Amihud vs Kyle: {amihud_kyle:.3f}
  
INTERPRETATION:
""")
    
    if e4_amihud < 0.60:
        print("""  ✓ E4 and Amihud capture DIFFERENT risks!
     - E4: Volatility structure (realized vol across scales)
     - Amihud: Price impact / Liquidity stress
     
  → RECOMMENDATION: Use BOTH as complementary signals
     
     Option A - Blend:
       CARIA_v9 = 0.5 × E4 + 0.5 × Amihud
       
     Option B - Veto System:
       if Amihud > 0.8: STAY OUT (liquidity hole)
       else: use CARIA-SR for alpha""")
    
    elif e4_amihud < 0.85:
        print("""  ⚡ E4 and Amihud are MODERATELY correlated
     - Some overlap, but still capture different aspects
     
  → RECOMMENDATION: 
     - Use Amihud as PRIMARY (better AUC)
     - Keep E4 as SECONDARY/confirmation""")
    
    else:
        print("""  ⚠️ E4 and Amihud are HIGHLY redundant!
     
  → RECOMMENDATION: 
     - Use ONLY Amihud (simpler, cheaper)
     - Drop E4 from the model""")
    
    if amihud_kyle > 0.85:
        print(f"""
  ⚠️ Amihud and Kyle's Lambda are REDUNDANT ({amihud_kyle:.3f})
     - Both measure price impact
     - Kyle's Lambda adds no information
     
  → Use Amihud only (skip Kyle's Lambda)""")
    
    # Best veto threshold
    best_veto = veto_results.loc[veto_results['Crash_Avoid_Rate'].idxmax()]
    print(f"""
VETO SYSTEM:
  Best threshold: Amihud > {best_veto['Threshold']}
  - Avoids {best_veto['Crash_Avoid_Rate']:.1%} of crashes
  - Stays out {best_veto['Pct_Vetoed']:.1%} of the time
""")


def main():
    """Run complete analysis."""
    print("=" * 70)
    print("AMIHUD VETO SYSTEM + CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Load and prepare data
    r, p, v, r_hyg = load_data()
    df = compute_features(r, p, v, r_hyg)
    
    # Part 1: Veto System
    veto_results = test_veto_system(df)
    test_veto_auc(df)
    analyze_veto_regimes(df)
    
    # Part 2: Correlation Analysis
    corr_ranked, corr_raw = correlation_analysis(df)
    
    # Visualizations
    create_visualizations(df, veto_results, corr_ranked, OUTPUT_DIR)
    
    # Save results
    veto_results.to_csv(os.path.join(OUTPUT_DIR, 'Table_Veto_Results.csv'), index=False)
    corr_ranked.to_csv(os.path.join(OUTPUT_DIR, 'Table_Correlation_Matrix.csv'))
    print(f"✓ Saved: Table_Veto_Results.csv")
    print(f"✓ Saved: Table_Correlation_Matrix.csv")
    
    # Final recommendation
    final_recommendation(corr_ranked, veto_results)
    
    return df, veto_results, corr_ranked


if __name__ == "__main__":
    df, veto_results, corr_ranked = main()















