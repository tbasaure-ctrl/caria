"""
Analyze Missed Crashes: What component catches the remaining 20%?
=================================================================

Current detection (SR > 0.7 or Amihud > 0.7):
  - Both high: 31.9%
  - Only CARIA-SR: 20.1%
  - Only Amihud: 27.1%
  - NEITHER (missed): 20.8%  <-- What catches these?

Hypotheses for missed crashes:
1. EXOGENOUS SHOCKS - Sudden external events (news, geopolitical)
2. TAIL RISK - Fat tails not captured by normal volatility
3. CROSS-ASSET CONTAGION - Correlation breakdown
4. SENTIMENT/POSITIONING - Overcrowding, leverage
5. GAP RISK - Overnight jumps

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


def load_comprehensive_data():
    """Load SPY plus additional assets for cross-asset features."""
    print("Loading comprehensive data...")
    
    tickers = {
        'SPY': 'Equity',
        '^VIX': 'Volatility',
        'HYG': 'Credit',
        'TLT': 'Bonds',
        'GLD': 'Gold',
        'UUP': 'Dollar',
        '^VIX3M': 'VIX_3M',  # For term structure
    }
    
    data = {}
    for ticker, name in tickers.items():
        try:
            df = yf.download(ticker, start=START_DATE, progress=False)
            if len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    data[name] = {
                        'close': df['Close'].iloc[:, 0],
                        'high': df['High'].iloc[:, 0],
                        'low': df['Low'].iloc[:, 0],
                        'volume': df['Volume'].iloc[:, 0] if 'Volume' in df.columns else None,
                        'open': df['Open'].iloc[:, 0]
                    }
                else:
                    data[name] = {
                        'close': df['Close'],
                        'high': df['High'],
                        'low': df['Low'],
                        'volume': df['Volume'] if 'Volume' in df.columns else None,
                        'open': df['Open']
                    }
                print(f"  ✓ {name}: {len(df)} obs")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    return data


def compute_all_features(data):
    """Compute comprehensive feature set."""
    print("\nComputing features...")
    
    spy = data['Equity']
    p = spy['close']
    h = spy['high']
    l = spy['low']
    o = spy['open']
    v = spy['volume']
    r = p.pct_change()
    
    features = {}
    
    # === ORIGINAL COMPONENTS ===
    v5 = r.rolling(5).std() * np.sqrt(252)
    v21 = r.rolling(21).std() * np.sqrt(252)
    v63 = r.rolling(63).std() * np.sqrt(252)
    
    # Credit vol
    if 'Credit' in data:
        hyg_ret = data['Credit']['close'].pct_change()
        v_cred = hyg_ret.rolling(42).std() * np.sqrt(252)
        common = r.index.intersection(v_cred.index)
        v_cred = v_cred.loc[common]
        r = r.loc[common]
        p = p.loc[common]
        h = h.loc[common]
        l = l.loc[common]
        o = o.loc[common]
        v = v.loc[common]
        v5 = v5.loc[common]
        v21 = v21.loc[common]
        v63 = v63.loc[common]
    else:
        v_cred = v21
    
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
    
    # Amihud
    dollar_vol = p * v
    amihud_raw = np.abs(r) / (dollar_vol + 1e-9)
    amihud = amihud_raw.rolling(21).mean().rolling(252).rank(pct=True)
    features['Amihud'] = amihud
    
    # === NEW CANDIDATE FEATURES ===
    
    # 1. GAP RISK (Overnight jumps)
    gap = (o - p.shift(1)) / p.shift(1)
    gap_abs = np.abs(gap).rolling(21).mean()
    gap_rank = gap_abs.rolling(252).rank(pct=True)
    features['Gap_Risk'] = gap_rank
    
    # 2. INTRADAY RANGE EXPANSION (Sudden intraday volatility)
    true_range = np.maximum(h - l, 
                  np.maximum(np.abs(h - p.shift(1)), np.abs(l - p.shift(1))))
    atr = true_range.rolling(14).mean()
    atr_expansion = atr / atr.rolling(63).mean()
    atr_rank = atr_expansion.rolling(252).rank(pct=True)
    features['ATR_Expansion'] = atr_rank
    
    # 3. TAIL RISK (Kurtosis - fat tails)
    kurtosis = r.rolling(63).apply(lambda x: stats.kurtosis(x), raw=True)
    kurtosis_rank = kurtosis.rolling(252).rank(pct=True)
    features['Tail_Risk'] = kurtosis_rank
    
    # 4. SKEWNESS (Asymmetric risk)
    skew = r.rolling(63).apply(lambda x: stats.skew(x), raw=True)
    skew_rank = (-skew).rolling(252).rank(pct=True)  # Negative skew is bad
    features['Skew_Risk'] = skew_rank
    
    # 5. DRAWDOWN DEPTH (How deep are we from highs)
    rolling_max = p.rolling(252).max()
    drawdown = (p - rolling_max) / rolling_max
    dd_depth = (-drawdown).rolling(252).rank(pct=True)
    features['Drawdown_Depth'] = dd_depth
    
    # 6. VIX LEVEL
    if 'Volatility' in data:
        vix = data['Volatility']['close']
        common = features['E4'].index.intersection(vix.index)
        vix = vix.loc[common]
        vix_rank = vix.rolling(252).rank(pct=True)
        features['VIX'] = vix_rank
        
        # Realign all features
        for feat in features:
            if feat != 'VIX':
                features[feat] = features[feat].loc[common]
        r = r.loc[common]
        p = p.loc[common]
    
    # 7. VIX TERM STRUCTURE (Backwardation = fear)
    if 'VIX_3M' in data and 'Volatility' in data:
        vix_spot = data['Volatility']['close']
        vix_3m = data['VIX_3M']['close']
        common_vix = vix_spot.index.intersection(vix_3m.index).intersection(features['E4'].index)
        
        if len(common_vix) > 500:
            term_struct = (vix_spot.loc[common_vix] / vix_3m.loc[common_vix])
            # Backwardation (spot > 3M) = fear = high rank
            term_rank = term_struct.rolling(252).rank(pct=True)
            features['VIX_Term'] = term_rank
            
            # Realign
            for feat in features:
                if feat != 'VIX_Term':
                    features[feat] = features[feat].loc[common_vix]
            r = r.loc[common_vix]
            p = p.loc[common_vix]
    
    # 8. CROSS-ASSET CORRELATION (Correlation breakdown = stress)
    if 'Bonds' in data:
        tlt_ret = data['Bonds']['close'].pct_change()
        common_bond = r.index.intersection(tlt_ret.index)
        
        if len(common_bond) > 500:
            spy_bond_corr = r.loc[common_bond].rolling(63).corr(tlt_ret.loc[common_bond])
            # Normally negative, becomes positive in crisis
            corr_rank = spy_bond_corr.rolling(252).rank(pct=True)
            features['Bond_Corr'] = corr_rank
            
            # Realign
            for feat in features:
                if feat != 'Bond_Corr':
                    features[feat] = features[feat].loc[common_bond]
            r = r.loc[common_bond]
            p = p.loc[common_bond]
    
    # 9. GOLD CORRELATION (Flight to safety indicator)
    if 'Gold' in data:
        gld_ret = data['Gold']['close'].pct_change()
        common_gold = r.index.intersection(gld_ret.index)
        
        if len(common_gold) > 500:
            spy_gold_corr = r.loc[common_gold].rolling(63).corr(gld_ret.loc[common_gold])
            # Negative = flight to safety
            gold_flight = (-spy_gold_corr).rolling(252).rank(pct=True)
            features['Gold_Flight'] = gold_flight
            
            # Realign
            for feat in features:
                if feat != 'Gold_Flight':
                    features[feat] = features[feat].loc[common_gold]
            r = r.loc[common_gold]
    
    # 10. CONSECUTIVE GAINS (Complacency indicator)
    gains = (r > 0).astype(int)
    consec_gains = gains.rolling(21).sum()
    complacency = consec_gains.rolling(252).rank(pct=True)
    features['Complacency'] = complacency
    
    # 11. LOW VOL REGIME (Calm before storm)
    low_vol = (1 - v21.rolling(252).rank(pct=True))  # High when vol is low
    features['Low_Vol_Regime'] = low_vol
    
    # 12. VOLUME SPIKE (Sudden volume = panic)
    vol_spike = v / v.rolling(63).mean()
    vol_spike_rank = vol_spike.rolling(252).rank(pct=True)
    features['Volume_Spike'] = vol_spike_rank
    
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
    
    print(f"  Features: {list(features.keys())}")
    print(f"  Observations: {len(df)}")
    
    return df


def analyze_missed_crashes(df):
    """Deep dive into crashes missed by both CARIA-SR and Amihud."""
    print("\n" + "=" * 70)
    print("ANALYSIS OF MISSED CRASHES")
    print("=" * 70)
    
    crashes = df[df['Target'] == 1].copy()
    
    caria_high = crashes['CARIA_SR'] > 0.7
    amihud_high = crashes['Amihud'] > 0.7
    
    # Classification
    both = crashes[caria_high & amihud_high]
    only_caria = crashes[caria_high & ~amihud_high]
    only_amihud = crashes[~caria_high & amihud_high]
    neither = crashes[~caria_high & ~amihud_high]
    
    print(f"\nCrash breakdown:")
    print(f"  Both high:      {len(both):4d} ({len(both)/len(crashes):.1%})")
    print(f"  Only CARIA:     {len(only_caria):4d} ({len(only_caria)/len(crashes):.1%})")
    print(f"  Only Amihud:    {len(only_amihud):4d} ({len(only_amihud)/len(crashes):.1%})")
    print(f"  NEITHER:        {len(neither):4d} ({len(neither)/len(crashes):.1%})")
    
    # Characteristics of missed crashes
    print("\n" + "-" * 50)
    print("CHARACTERISTICS OF MISSED CRASHES (neither high):")
    
    feature_cols = [c for c in df.columns if c not in ['Target', 'Fwd_Ret']]
    
    print(f"\n{'Feature':<20} | {'Missed':>8} | {'Caught':>8} | {'Delta':>8}")
    print("-" * 55)
    
    caught = crashes[caria_high | amihud_high]
    
    significant_features = []
    
    for feat in feature_cols:
        missed_mean = neither[feat].mean()
        caught_mean = caught[feat].mean()
        delta = missed_mean - caught_mean
        
        # T-test
        if len(neither) > 5 and len(caught) > 5:
            t_stat, p_val = stats.ttest_ind(neither[feat], caught[feat])
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        else:
            sig = ""
        
        if abs(delta) > 0.1:
            significant_features.append((feat, delta, missed_mean))
        
        print(f"{feat:<20} | {missed_mean:>8.3f} | {caught_mean:>8.3f} | {delta:>+8.3f} {sig}")
    
    # Key insights
    print("\n" + "-" * 50)
    print("KEY DISTINGUISHING FEATURES OF MISSED CRASHES:")
    
    for feat, delta, mean in sorted(significant_features, key=lambda x: abs(x[1]), reverse=True):
        direction = "HIGH" if delta > 0 else "LOW"
        print(f"  • {feat}: {direction} ({mean:.3f}) - Delta: {delta:+.3f}")
    
    return neither, caught, significant_features


def find_third_component(df, neither, caught):
    """Find the best third component to catch missed crashes."""
    print("\n" + "=" * 70)
    print("SEARCH FOR THIRD COMPONENT")
    print("=" * 70)
    
    target = df['Target']
    caria = df['CARIA_SR']
    amihud = df['Amihud']
    
    # Baseline: Best of CARIA or Amihud
    combined_base = np.maximum(caria, amihud)
    base_auc = roc_auc_score(target, combined_base)
    
    print(f"\nBaseline (max(CARIA, Amihud)) AUC: {base_auc:.4f}")
    
    # Test each candidate as third component
    candidates = [c for c in df.columns if c not in ['Target', 'Fwd_Ret', 'CARIA_SR', 'Amihud', 'E4', 'Sync']]
    
    print(f"\n{'Third Component':<20} | {'Triple AUC':>10} | {'ΔAUC':>8} | {'Catches Missed?'}")
    print("-" * 65)
    
    results = []
    
    for cand in candidates:
        # Triple combination: max of all three
        triple = np.maximum(np.maximum(caria, amihud), df[cand])
        triple_auc = roc_auc_score(target, triple)
        delta = triple_auc - base_auc
        
        # Does it catch more missed crashes?
        cand_high = df[cand] > 0.7
        caria_high = caria > 0.7
        amihud_high = amihud > 0.7
        
        # Crashes caught only by this new component
        crashes = df[df['Target'] == 1]
        new_catches = crashes[cand_high & ~caria_high & ~amihud_high]
        
        catches_str = f"{len(new_catches)} new" if len(new_catches) > 0 else "-"
        improve = "✓" if delta > 0.005 else "≈" if delta > -0.005 else "✗"
        
        results.append({
            'Component': cand,
            'Triple_AUC': triple_auc,
            'Delta': delta,
            'New_Catches': len(new_catches)
        })
        
        print(f"{cand:<20} | {triple_auc:>10.4f} | {delta:>+8.4f} | {catches_str:>15} {improve}")
    
    results_df = pd.DataFrame(results).sort_values('Delta', ascending=False)
    
    # Best third component
    best = results_df.iloc[0]
    print("\n" + "-" * 50)
    print(f"BEST THIRD COMPONENT: {best['Component']}")
    print(f"  New AUC: {best['Triple_AUC']:.4f} (Δ = {best['Delta']:+.4f})")
    print(f"  Additional crashes caught: {best['New_Catches']}")
    
    return results_df


def build_optimal_triple(df):
    """Build optimal three-component model."""
    print("\n" + "=" * 70)
    print("OPTIMAL TRIPLE MODEL")
    print("=" * 70)
    
    target = df['Target']
    
    # Components
    E4 = df['E4']
    Amihud = df['Amihud']
    
    # Best candidates for third component based on analysis
    third_candidates = ['Tail_Risk', 'Gap_Risk', 'ATR_Expansion', 'VIX', 'Skew_Risk']
    third_candidates = [c for c in third_candidates if c in df.columns]
    
    print("\nTesting formulas for CARIA v9.0 Triple:")
    print(f"\n{'Formula':<45} | {'AUC':>8}")
    print("-" * 60)
    
    best_auc = 0
    best_formula = None
    
    for third in third_candidates:
        Third = df[third]
        
        # Formula 1: Max of three ranks
        f1 = np.maximum(np.maximum(E4, Amihud), Third)
        f1 = f1.rolling(252, min_periods=50).rank(pct=True)
        
        # Formula 2: Weighted average
        f2 = 0.4 * E4 + 0.4 * Amihud + 0.2 * Third
        
        # Formula 3: Multiplicative
        f3 = E4 * Amihud * Third
        f3 = f3.rolling(252, min_periods=50).rank(pct=True)
        
        # Formula 4: E4 * (1 + Amihud) * (1 + Third)
        f4 = E4 * (1 + Amihud) * (1 + Third)
        f4 = f4.rolling(252, min_periods=50).rank(pct=True)
        
        formulas = {
            f'Max(E4, Amh, {third[:8]})': f1,
            f'0.4E4 + 0.4Amh + 0.2{third[:6]}': f2,
            f'E4 × Amh × {third[:8]}': f3,
            f'E4×(1+Amh)×(1+{third[:6]})': f4,
        }
        
        for name, formula in formulas.items():
            valid = formula.dropna()
            valid_target = target.loc[valid.index]
            auc = roc_auc_score(valid_target, valid)
            
            if auc > best_auc:
                best_auc = auc
                best_formula = name
            
            marker = "***" if auc >= best_auc else ""
            print(f"{name:<45} | {auc:>8.4f} {marker}")
    
    # Baselines
    print("-" * 60)
    caria_auc = roc_auc_score(target, df['CARIA_SR'])
    amihud_auc = roc_auc_score(target, df['Amihud'])
    
    print(f"\nComparison:")
    print(f"  Original CARIA-SR:        {caria_auc:.4f}")
    print(f"  Amihud alone:             {amihud_auc:.4f}")
    print(f"  Best triple formula:      {best_auc:.4f}")
    print(f"  Improvement vs CARIA:     {best_auc - caria_auc:+.4f}")
    print(f"  Improvement vs Amihud:    {best_auc - amihud_auc:+.4f}")
    
    return best_formula, best_auc


def create_visualization(df, neither, caught, output_dir):
    """Visualize missed crash characteristics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Feature comparison: Missed vs Caught
    ax1 = axes[0, 0]
    features_to_plot = ['Gap_Risk', 'Tail_Risk', 'ATR_Expansion', 'Low_Vol_Regime', 
                        'Complacency', 'Skew_Risk']
    features_to_plot = [f for f in features_to_plot if f in df.columns]
    
    missed_means = [neither[f].mean() for f in features_to_plot]
    caught_means = [caught[f].mean() for f in features_to_plot]
    
    x = np.arange(len(features_to_plot))
    width = 0.35
    
    ax1.bar(x - width/2, missed_means, width, label='Missed', color='red', alpha=0.7)
    ax1.bar(x + width/2, caught_means, width, label='Caught', color='blue', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f[:10] for f in features_to_plot], rotation=45, ha='right')
    ax1.set_ylabel('Mean Value')
    ax1.set_title('Feature Comparison: Missed vs Caught Crashes')
    ax1.legend()
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 2. Distribution of missed crashes by date
    ax2 = axes[0, 1]
    ax2.hist(neither.index.year, bins=20, color='red', alpha=0.7, label='Missed')
    ax2.hist(caught.index.year, bins=20, color='blue', alpha=0.5, label='Caught')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Count')
    ax2.set_title('Temporal Distribution of Crashes')
    ax2.legend()
    
    # 3. Scatter: E4 vs Amihud for missed crashes
    ax3 = axes[1, 0]
    normal = df[df['Target'] == 0].sample(min(500, len(df[df['Target'] == 0])))
    
    ax3.scatter(normal['E4'], normal['Amihud'], alpha=0.2, c='gray', s=10, label='Normal')
    ax3.scatter(caught['E4'], caught['Amihud'], alpha=0.5, c='blue', s=30, label='Caught')
    ax3.scatter(neither['E4'], neither['Amihud'], alpha=0.8, c='red', s=50, marker='x', label='Missed')
    ax3.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7)
    ax3.axvline(x=0.7, color='orange', linestyle='--', alpha=0.7)
    ax3.set_xlabel('E4 (Volatility)')
    ax3.set_ylabel('Amihud (Illiquidity)')
    ax3.set_title('Where Are Missed Crashes?')
    ax3.legend()
    
    # 4. Forward returns of missed crashes
    ax4 = axes[1, 1]
    ax4.hist(neither['Fwd_Ret'], bins=20, color='red', alpha=0.7, label='Missed')
    ax4.hist(caught['Fwd_Ret'], bins=20, color='blue', alpha=0.5, label='Caught')
    ax4.axvline(x=neither['Fwd_Ret'].mean(), color='red', linestyle='--', 
                label=f"Missed mean: {neither['Fwd_Ret'].mean():.1%}")
    ax4.axvline(x=caught['Fwd_Ret'].mean(), color='blue', linestyle='--',
                label=f"Caught mean: {caught['Fwd_Ret'].mean():.1%}")
    ax4.set_xlabel('21-Day Forward Return')
    ax4.set_ylabel('Count')
    ax4.set_title('Severity: Missed vs Caught Crashes')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_Missed_Crashes_Analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: Figure_Missed_Crashes_Analysis.png")


def main():
    """Run complete analysis."""
    print("=" * 70)
    print("ANALYSIS: WHAT CATCHES THE REMAINING 20%?")
    print("=" * 70)
    
    # Load data
    data = load_comprehensive_data()
    
    # Compute features
    df = compute_all_features(data)
    
    # Analyze missed crashes
    neither, caught, sig_features = analyze_missed_crashes(df)
    
    # Find third component
    third_results = find_third_component(df, neither, caught)
    
    # Build optimal triple
    best_formula, best_auc = build_optimal_triple(df)
    
    # Create visualization
    create_visualization(df, neither, caught, OUTPUT_DIR)
    
    # Save results
    third_results.to_csv(os.path.join(OUTPUT_DIR, 'Table_Third_Component_Search.csv'), index=False)
    print(f"✓ Saved: Table_Third_Component_Search.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: THE MISSING 20%")
    print("=" * 70)
    
    print("""
WHAT ARE THESE MISSED CRASHES?
  • LOW volatility before the crash
  • LOW illiquidity before the crash
  • Often preceded by COMPLACENCY (many positive days)
  
IN OTHER WORDS: "Calm before the storm" crashes
  
These are EXOGENOUS SHOCKS:
  - Flash crashes
  - Black swan events
  - Sudden news-driven selloffs
  
BEST THIRD COMPONENT CANDIDATES:
  1. TAIL RISK (Kurtosis) - Fat tails predict sudden moves
  2. GAP RISK - Overnight jump risk
  3. ATR EXPANSION - Sudden range expansion
  4. SKEW - Asymmetric risk building
    """)
    
    return df, neither, caught


if __name__ == "__main__":
    df, neither, caught = main()















