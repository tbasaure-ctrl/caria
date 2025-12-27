"""
Look-Ahead Bias Test for Amihud Illiquidity
============================================

CONCERN: Amihud = |Return_t| / (Volume_t × Price_t)
         If target involves Return_t, we have DATA LEAKAGE.

TEST: Add lags to Amihud and see if AUC drops significantly.
      If AUC 0.73 → 0.50 with 1-day lag: LEAKAGE CONFIRMED
      If AUC stays ~0.70 with lag: SIGNAL IS REAL

Author: Tomás Basaure
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
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


def load_data():
    """Load SPY data."""
    print("Loading data...")
    
    spy = yf.download("SPY", start=START_DATE, progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        p = spy["Close"].iloc[:, 0]
        v = spy["Volume"].iloc[:, 0]
    else:
        p = spy["Close"]
        v = spy["Volume"]
    
    r = p.pct_change()
    
    return r, p, v


def test_leakage(r, p, v):
    """Test for look-ahead bias with different lags."""
    print("\n" + "=" * 70)
    print("LOOK-AHEAD BIAS TEST")
    print("=" * 70)
    
    # === DEFINE TARGET ===
    # Target: Is the FORWARD 21-day return in the worst 5%?
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FWD_WINDOW)
    fwd_ret = r.rolling(window=indexer).sum()
    crash_threshold = fwd_ret.quantile(CRASH_QUANTILE)
    target = (fwd_ret < crash_threshold).astype(int)
    
    print(f"\nTarget Definition:")
    print(f"  Forward window: {FWD_WINDOW} days")
    print(f"  Crash threshold: {crash_threshold:.2%}")
    print(f"  Target uses: Return[t+1] to Return[t+{FWD_WINDOW}]")
    
    # === CALCULATE AMIHUD WITH DIFFERENT LAGS ===
    print("\n" + "-" * 50)
    print("AMIHUD AUC BY LAG:")
    print("-" * 50)
    
    dollar_vol = p * v
    amihud_raw = np.abs(r) / (dollar_vol + 1e-9)
    
    results = []
    
    # Test lags from 0 to 10 days
    for lag in range(0, 11):
        # Apply lag BEFORE smoothing
        amihud_lagged = amihud_raw.shift(lag)
        
        # Smooth and rank
        amihud_smooth = amihud_lagged.rolling(21).mean()
        amihud_rank = amihud_smooth.rolling(252).rank(pct=True)
        
        # Align with target
        valid = amihud_rank.dropna().index.intersection(target.dropna().index)
        
        if len(valid) > 500:
            auc = roc_auc_score(target.loc[valid], amihud_rank.loc[valid])
            results.append({'Lag': lag, 'AUC': auc})
            
            # Mark significant drops
            if lag == 0:
                baseline_auc = auc
                marker = "(baseline)"
            else:
                drop = baseline_auc - auc
                if drop > 0.10:
                    marker = "⚠️ MAJOR DROP - LIKELY LEAKAGE"
                elif drop > 0.05:
                    marker = "⚡ Moderate drop"
                elif drop > 0.02:
                    marker = "Minor drop"
                else:
                    marker = "✓ Stable (no leakage)"
            
            print(f"  Lag {lag:2d} days: AUC = {auc:.4f} {marker}")
    
    results_df = pd.DataFrame(results)
    
    # === INTERPRETATION ===
    print("\n" + "-" * 50)
    print("INTERPRETATION:")
    
    lag0 = results_df[results_df['Lag'] == 0]['AUC'].values[0]
    lag1 = results_df[results_df['Lag'] == 1]['AUC'].values[0]
    lag5 = results_df[results_df['Lag'] == 5]['AUC'].values[0]
    
    drop_1d = lag0 - lag1
    drop_5d = lag0 - lag5
    
    print(f"\n  AUC drop with 1-day lag: {drop_1d:+.4f}")
    print(f"  AUC drop with 5-day lag: {drop_5d:+.4f}")
    
    if drop_1d > 0.10:
        print("\n  ⚠️  LEAKAGE DETECTED!")
        print("      The high AUC is mostly due to same-day return information.")
        print("      Amihud is 'seeing' the crash as it happens, not predicting it.")
        leakage = True
    elif drop_1d > 0.05:
        print("\n  ⚡ PARTIAL LEAKAGE")
        print("      Some of the signal comes from same-day information.")
        print("      But there IS real predictive power with lag.")
        leakage = "partial"
    else:
        print("\n  ✓ NO SIGNIFICANT LEAKAGE")
        print("      The signal is robust to lagging.")
        print("      Amihud captures a real, persistent liquidity regime.")
        leakage = False
    
    return results_df, leakage


def test_e4_leakage(r, p, v):
    """Test E4 for leakage too."""
    print("\n" + "=" * 70)
    print("E4 (VOLATILITY) LEAKAGE TEST")
    print("=" * 70)
    
    # Target
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FWD_WINDOW)
    fwd_ret = r.rolling(window=indexer).sum()
    crash_threshold = fwd_ret.quantile(CRASH_QUANTILE)
    target = (fwd_ret < crash_threshold).astype(int)
    
    print("\nE4 uses rolling volatility which contains Return^2")
    print("Testing if E4 also has leakage...")
    
    results = []
    
    for lag in range(0, 6):
        # Calculate volatility with lag
        r_lagged = r.shift(lag)
        
        v5 = r_lagged.rolling(5).std() * np.sqrt(252)
        v21 = r_lagged.rolling(21).std() * np.sqrt(252)
        v63 = r_lagged.rolling(63).std() * np.sqrt(252)
        
        E4_raw = 0.25 * v5 + 0.35 * v21 + 0.40 * v63  # Simplified
        E4 = E4_raw.rolling(252).rank(pct=True)
        
        valid = E4.dropna().index.intersection(target.dropna().index)
        
        if len(valid) > 500:
            auc = roc_auc_score(target.loc[valid], E4.loc[valid])
            results.append({'Lag': lag, 'AUC': auc})
            print(f"  Lag {lag} days: AUC = {auc:.4f}")
    
    results_df = pd.DataFrame(results)
    
    lag0 = results_df[results_df['Lag'] == 0]['AUC'].values[0]
    lag1 = results_df[results_df['Lag'] == 1]['AUC'].values[0]
    
    print(f"\n  E4 AUC drop with 1-day lag: {lag0 - lag1:+.4f}")
    
    return results_df


def detailed_timing_check(r, p, v):
    """Check exact timing of calculations."""
    print("\n" + "=" * 70)
    print("DETAILED TIMING ANALYSIS")
    print("=" * 70)
    
    print("""
    CURRENT IMPLEMENTATION:
    
    Day t:
    ├── Amihud_t = |Return_t| / DollarVol_t
    │   └── Return_t = (Price_t - Price_{t-1}) / Price_{t-1}
    │   └── Smoothed over 21 days: mean(Amihud_{t-20} ... Amihud_t)
    │   └── Ranked over 252 days
    │
    └── Target_t = Is FwdReturn_t in worst 5%?
        └── FwdReturn_t = sum(Return_{t+1}, ..., Return_{t+21})
    
    QUESTION: Does Amihud_t contain information about Target_t?
    
    ANALYSIS:
    - Amihud_t uses: Return_t (today's return)
    - Target_t uses: Return_{t+1} to Return_{t+21} (tomorrow through 21 days)
    
    TECHNICALLY: No direct overlap!
    - Return_t is KNOWN at end of day t
    - Target_t starts from day t+1
    
    BUT: There could be AUTOCORRELATION
    - If a crash starts on day t (big |Return_t|)
    - And continues on days t+1 to t+21
    - Then Amihud_t "sees" the start of the crash
    
    THIS IS ACTUALLY DESIRABLE for an early warning system!
    """)
    
    # Test autocorrelation of returns
    print("-" * 50)
    print("RETURN AUTOCORRELATION:")
    
    for lag in [1, 5, 10, 21]:
        autocorr = r.autocorr(lag=lag)
        print(f"  Autocorr(lag={lag:2d}): {autocorr:+.4f}")
    
    # Test if big returns predict crashes
    print("\n" + "-" * 50)
    print("DO BIG RETURNS (|r|) PREDICT CRASHES?")
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FWD_WINDOW)
    fwd_ret = r.rolling(window=indexer).sum()
    crash_threshold = fwd_ret.quantile(CRASH_QUANTILE)
    target = (fwd_ret < crash_threshold).astype(int)
    
    # Pure |Return| as predictor
    abs_return = np.abs(r)
    abs_return_rank = abs_return.rolling(252).rank(pct=True)
    
    valid = abs_return_rank.dropna().index.intersection(target.dropna().index)
    auc_abs_ret = roc_auc_score(target.loc[valid], abs_return_rank.loc[valid])
    
    print(f"\n  Pure |Return_t| as predictor: AUC = {auc_abs_ret:.4f}")
    
    if auc_abs_ret > 0.65:
        print("  ⚠️ |Return| alone has high AUC - this is the 'leaking' component!")
    else:
        print("  ✓ |Return| alone has low AUC - Amihud adds value beyond |Return|")
    
    # Amihud vs |Return| comparison
    print("\n" + "-" * 50)
    print("DECOMPOSING AMIHUD'S PREDICTIVE POWER:")
    
    # 1. Only Volume component
    vol_rank = (1 / v).rolling(21).mean().rolling(252).rank(pct=True)
    valid = vol_rank.dropna().index.intersection(target.dropna().index)
    auc_vol = roc_auc_score(target.loc[valid], vol_rank.loc[valid])
    
    # 2. Only |Return| component (smoothed)
    ret_rank = abs_return.rolling(21).mean().rolling(252).rank(pct=True)
    valid = ret_rank.dropna().index.intersection(target.dropna().index)
    auc_ret = roc_auc_score(target.loc[valid], ret_rank.loc[valid])
    
    # 3. Full Amihud
    dollar_vol = p * v
    amihud = (np.abs(r) / (dollar_vol + 1e-9)).rolling(21).mean().rolling(252).rank(pct=True)
    valid = amihud.dropna().index.intersection(target.dropna().index)
    auc_amihud = roc_auc_score(target.loc[valid], amihud.loc[valid])
    
    print(f"\n  {'Component':<30} | {'AUC':>8}")
    print("  " + "-" * 45)
    print(f"  {'1/Volume (liquidity only)':<30} | {auc_vol:>8.4f}")
    print(f"  {'|Return| smoothed (vol proxy)':<30} | {auc_ret:>8.4f}")
    print(f"  {'Full Amihud (|Return|/DolVol)':<30} | {auc_amihud:>8.4f}")
    
    print("\n  INTERPRETATION:")
    if auc_ret > auc_vol and auc_ret > 0.6:
        print("    ⚠️ Most of Amihud's AUC comes from the |Return| component!")
        print("       This suggests volatility clustering, not true liquidity prediction.")
    elif auc_vol > auc_ret:
        print("    ✓ Volume component is more predictive than |Return|")
        print("       This suggests real liquidity information.")
    
    return auc_vol, auc_ret, auc_amihud


def create_visualization(amihud_results, output_dir):
    """Visualize leakage test results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. AUC by Lag
    ax1 = axes[0]
    ax1.plot(amihud_results['Lag'], amihud_results['AUC'], 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='red', linestyle='--', label='Random (0.5)')
    ax1.axhline(y=0.55, color='orange', linestyle='--', alpha=0.5, label='Minimal predictive (0.55)')
    
    ax1.set_xlabel('Lag (days)')
    ax1.set_ylabel('AUC')
    ax1.set_title('Amihud AUC by Feature Lag')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Annotate
    lag0_auc = amihud_results[amihud_results['Lag'] == 0]['AUC'].values[0]
    lag5_auc = amihud_results[amihud_results['Lag'] == 5]['AUC'].values[0]
    ax1.annotate(f'No lag: {lag0_auc:.3f}', xy=(0, lag0_auc), xytext=(1, lag0_auc + 0.02),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate(f'5-day lag: {lag5_auc:.3f}', xy=(5, lag5_auc), xytext=(6, lag5_auc + 0.02),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    # 2. Leakage interpretation
    ax2 = axes[1]
    drop = lag0_auc - lag5_auc
    
    categories = ['Lag 0\n(current)', 'Lag 5\n(lagged)']
    values = [lag0_auc, lag5_auc]
    colors = ['red' if drop > 0.1 else 'orange' if drop > 0.05 else 'green', 'blue']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0.5, color='red', linestyle='--')
    ax2.set_ylabel('AUC')
    ax2.set_title(f'Leakage Test: AUC Drop = {drop:.3f}')
    ax2.set_ylim(0.4, 0.8)
    
    # Add interpretation text
    if drop > 0.10:
        verdict = "⚠️ LEAKAGE DETECTED"
        color = 'red'
    elif drop > 0.05:
        verdict = "⚡ PARTIAL LEAKAGE"
        color = 'orange'
    else:
        verdict = "✓ NO LEAKAGE"
        color = 'green'
    
    ax2.text(0.5, 0.45, verdict, ha='center', fontsize=14, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_Leakage_Test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: Figure_Leakage_Test.png")


def main():
    """Run complete leakage analysis."""
    print("=" * 70)
    print("LOOK-AHEAD BIAS / DATA LEAKAGE ANALYSIS")
    print("=" * 70)
    
    # Load data
    r, p, v = load_data()
    
    # Test Amihud leakage
    amihud_results, leakage = test_leakage(r, p, v)
    
    # Test E4 leakage
    e4_results = test_e4_leakage(r, p, v)
    
    # Detailed timing analysis
    auc_vol, auc_ret, auc_amihud = detailed_timing_check(r, p, v)
    
    # Create visualization
    create_visualization(amihud_results, OUTPUT_DIR)
    
    # Save results
    amihud_results.to_csv(os.path.join(OUTPUT_DIR, 'Table_Leakage_Test.csv'), index=False)
    print(f"✓ Saved: Table_Leakage_Test.csv")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    lag0 = amihud_results[amihud_results['Lag'] == 0]['AUC'].values[0]
    lag5 = amihud_results[amihud_results['Lag'] == 5]['AUC'].values[0]
    
    print(f"""
    Amihud AUC (no lag):    {lag0:.4f}
    Amihud AUC (5-day lag): {lag5:.4f}
    Drop:                   {lag0 - lag5:.4f}
    
    |Return| component AUC: {auc_ret:.4f}
    Volume component AUC:   {auc_vol:.4f}
    """)
    
    if leakage == True:
        print("""
    ⚠️  CONCLUSION: SIGNIFICANT LEAKAGE DETECTED
    
    The high Amihud AUC is largely due to same-day return information.
    When properly lagged, the signal loses most of its predictive power.
    
    RECOMMENDATION:
    - Use Amihud with at least 1-day lag
    - Or focus on the Volume component alone
    - Re-validate all results with lagged features
        """)
    elif leakage == "partial":
        print("""
    ⚡ CONCLUSION: PARTIAL LEAKAGE
    
    Some of Amihud's AUC comes from same-day information,
    but there IS real predictive power that persists with lag.
    
    RECOMMENDATION:
    - Use 1-day lagged Amihud for conservative estimates
    - The 'true' AUC is the lagged version
        """)
    else:
        print("""
    ✓ CONCLUSION: NO SIGNIFICANT LEAKAGE
    
    Amihud's predictive power is ROBUST to lagging.
    The signal captures a persistent liquidity regime, not same-day noise.
    
    This is the IDEAL scenario for an early warning system:
    - High illiquidity TODAY predicts crashes TOMORROW
    - The signal is forward-looking, not coincident
        """)
    
    return amihud_results


if __name__ == "__main__":
    results = main()















