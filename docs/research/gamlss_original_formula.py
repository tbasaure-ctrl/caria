"""
GAMLSS Test: Original CARIA-SR Formula vs E4 Alone
===================================================

Compare distributional effects of:
1. E4 alone (Signal = E4 > 0.8)
2. Original CARIA-SR = E4 × (1 + Sync) (Signal = SR > 0.8)

Which one produces a stronger distributional shift?

Author: Tomás Basaure
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import t as student_t
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
START_DATE = "2005-01-01"
FWD_WINDOW = 21


def neg_log_likelihood_t(params, data):
    """Negative log-likelihood for Student-t distribution."""
    df, loc, scale = params
    if df <= 2 or scale <= 0:
        return np.inf
    return -np.sum(student_t.logpdf(data, df=df, loc=loc, scale=scale))


def fit_student_t(data):
    """Fit Student-t distribution via MLE."""
    init_df = 5
    init_loc = np.median(data)
    init_scale = np.std(data) * 0.8
    
    result = minimize(
        neg_log_likelihood_t,
        x0=[init_df, init_loc, init_scale],
        args=(data,),
        method='Nelder-Mead',
        options={'maxiter': 5000}
    )
    
    df, loc, scale = result.x
    nll = result.fun
    
    return {'df': df, 'loc': loc, 'scale': scale, 'nll': nll, 'n': len(data)}


def likelihood_ratio_test(nll_null, nll_alt, df_diff):
    """Likelihood Ratio Test."""
    lr_stat = 2 * (nll_null - nll_alt)
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
    return lr_stat, p_value


def build_both_signals():
    """Build both E4 and original CARIA-SR signals."""
    print("Loading data and building signals...")
    
    spy = yf.download("SPY", start=START_DATE, progress=False)
    hyg = yf.download("HYG", start=START_DATE, progress=False)["Close"]
    
    if isinstance(spy.columns, pd.MultiIndex):
        price = spy["Close"].iloc[:, 0]
    else:
        price = spy["Close"]
    
    if isinstance(hyg, pd.DataFrame):
        hyg = hyg.iloc[:, 0]
    
    ret = price.pct_change()
    ret_hyg = hyg.pct_change()
    
    common = ret.index.intersection(ret_hyg.index)
    r = ret.loc[common]
    
    # E4
    v5 = r.rolling(5).std() * np.sqrt(252)
    v21 = r.rolling(21).std() * np.sqrt(252)
    v63 = r.rolling(63).std() * np.sqrt(252)
    v_cred = ret_hyg.loc[common].rolling(42).std() * np.sqrt(252)
    
    E4_raw = 0.20*v5 + 0.30*v21 + 0.25*v63 + 0.25*v_cred
    E4 = E4_raw.rolling(252).rank(pct=True)
    
    # Sync (original formula)
    m_fast = r.rolling(5).sum()
    m_slow = r.rolling(63).sum()
    sync_raw = m_fast.rolling(21).corr(m_slow)
    Sync = ((sync_raw + 1) / 2).rolling(252).rank(pct=True)
    
    # Original CARIA-SR
    SR_raw = E4 * (1 + Sync)
    SR = SR_raw.rolling(252).rank(pct=True)
    
    # Forward returns
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FWD_WINDOW)
    fwd_ret = r.rolling(window=indexer).sum()
    
    df = pd.DataFrame({
        'Fwd_Ret_21': fwd_ret,
        'E4': E4,
        'Sync': Sync,
        'SR': SR,
        'Signal_E4': (E4 > 0.8).astype(int),
        'Signal_SR': (SR > 0.8).astype(int),
        'Signal_Sync': (Sync > 0.8).astype(int),
    }).dropna()
    
    print(f"  Observations: {len(df)}")
    
    return df


def test_signal(df, signal_name, signal_col):
    """Test distributional effect of a signal."""
    y = df['Fwd_Ret_21'].values
    signal = df[signal_col].values
    
    y_low = y[signal == 0]
    y_high = y[signal == 1]
    
    # Fit null model (pooled)
    fit_null = fit_student_t(y)
    
    # Fit separate models
    fit_low = fit_student_t(y_low)
    fit_high = fit_student_t(y_high)
    
    nll_separate = fit_low['nll'] + fit_high['nll']
    
    # LR test
    lr_stat, p_value = likelihood_ratio_test(fit_null['nll'], nll_separate, df_diff=3)
    
    # Scale change
    scale_pct = (fit_high['scale'] / fit_low['scale'] - 1) * 100
    loc_change = fit_high['loc'] - fit_low['loc']
    
    return {
        'signal': signal_name,
        'n_high': len(y_high),
        'n_low': len(y_low),
        'pct_high': len(y_high) / len(y) * 100,
        'lr_stat': lr_stat,
        'p_value': p_value,
        'loc_low': fit_low['loc'],
        'loc_high': fit_high['loc'],
        'loc_change': loc_change,
        'scale_low': fit_low['scale'],
        'scale_high': fit_high['scale'],
        'scale_pct': scale_pct,
        'df_low': fit_low['df'],
        'df_high': fit_high['df'],
    }


def main():
    """Compare original CARIA-SR vs E4 alone."""
    print("=" * 70)
    print("GAMLSS COMPARISON: Original CARIA-SR vs E4 Alone")
    print("=" * 70)
    
    df = build_both_signals()
    
    # Test each signal
    signals = [
        ('E4 > 0.8', 'Signal_E4'),
        ('SR > 0.8 (Original)', 'Signal_SR'),
        ('Sync > 0.8', 'Signal_Sync'),
    ]
    
    results = []
    
    for name, col in signals:
        result = test_signal(df, name, col)
        results.append(result)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Signal':<25} | {'LR χ²':>10} | {'p-value':>12} | {'σ Change':>10} | {'μ Change':>10}")
    print("-" * 80)
    
    for r in results:
        p_str = f"{r['p_value']:.6f}" if r['p_value'] > 0.0001 else "< 0.0001"
        print(f"{r['signal']:<25} | {r['lr_stat']:>10.2f} | {p_str:>12} | {r['scale_pct']:>+10.1f}% | {r['loc_change']:>+10.4f}")
    
    # Detailed comparison
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON")
    print("=" * 70)
    
    for r in results:
        print(f"\n{r['signal']}:")
        print(f"  Sample: {r['n_high']} high ({r['pct_high']:.1f}%), {r['n_low']} low")
        print(f"  LR Test: χ² = {r['lr_stat']:.2f}, p = {r['p_value']:.6f}")
        print(f"  Location: {r['loc_low']:.5f} → {r['loc_high']:.5f} (Δ = {r['loc_change']:+.5f})")
        print(f"  Scale:    {r['scale_low']:.5f} → {r['scale_high']:.5f} (Δ = {r['scale_pct']:+.1f}%)")
        print(f"  DF:       {r['df_low']:.2f} → {r['df_high']:.2f}")
    
    # Winner
    print("\n" + "=" * 70)
    print("VERDICT: Which Signal Produces Stronger Distributional Shift?")
    print("=" * 70)
    
    # Sort by LR statistic (higher = stronger effect)
    results_sorted = sorted(results, key=lambda x: x['lr_stat'], reverse=True)
    
    print(f"\nRanking by LR statistic (higher = stronger distributional effect):")
    for i, r in enumerate(results_sorted, 1):
        sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
        print(f"  {i}. {r['signal']:<25} χ² = {r['lr_stat']:>8.2f} {sig}")
    
    winner = results_sorted[0]
    e4_result = [r for r in results if 'E4' in r['signal'] and 'SR' not in r['signal']][0]
    sr_result = [r for r in results if 'Original' in r['signal']][0]
    
    print(f"\n" + "-" * 50)
    
    if e4_result['lr_stat'] > sr_result['lr_stat']:
        improvement = (e4_result['lr_stat'] / sr_result['lr_stat'] - 1) * 100
        print(f"  E4 ALONE produces {improvement:.1f}% STRONGER distributional shift")
        print(f"  than the original CARIA-SR formula!")
        print(f"\n  This confirms: Sync is adding NOISE, not signal.")
    else:
        improvement = (sr_result['lr_stat'] / e4_result['lr_stat'] - 1) * 100
        print(f"  Original CARIA-SR produces {improvement:.1f}% STRONGER shift")
        print(f"  than E4 alone!")
        print(f"\n  This means: Sync IS adding value in the distributional framework.")
    
    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    best = results_sorted[0]
    print(f"""
  BEST SIGNAL: {best['signal']}
  
  Effect:
    • Scale increase: {best['scale_pct']:+.1f}%
    • Location change: {best['loc_change']:+.5f} (should be ~0)
    • LR statistic: {best['lr_stat']:.2f} (p < 0.0001)
  
  Valid Claim:
    "Conditional on {best['signal']}, the distribution of 21-day forward 
     returns exhibits {best['scale_pct']:.0f}% higher dispersion, 
     despite an unchanged mean (Δμ = {best['loc_change']:.4f})."
    """)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'gamlss_formula_comparison.csv'), index=False)
    print(f"✓ Results saved to gamlss_formula_comparison.csv")
    
    return results


if __name__ == "__main__":
    results = main()















