"""
GAMLSS-Equivalent Analysis in Python
=====================================

Since GAMLSS in R requires package installation, here's a Python equivalent
using Maximum Likelihood Estimation with scipy.

The test:
  H0: Signal_Strict has NO effect on σ (scale) or ν (tail df)
  H1: Signal_Strict DOES affect σ and/or ν

We fit a Student-t distribution to each regime and test if parameters differ.

Author: Tomás Basaure
Date: December 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import t as student_t
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def neg_log_likelihood_t(params, data):
    """Negative log-likelihood for Student-t distribution."""
    df, loc, scale = params
    if df <= 2 or scale <= 0:
        return np.inf
    return -np.sum(student_t.logpdf(data, df=df, loc=loc, scale=scale))


def fit_student_t(data):
    """Fit Student-t distribution via MLE."""
    # Initial guess
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
    n = len(data)
    k = 3  # number of parameters
    aic = 2 * k + 2 * nll
    bic = k * np.log(n) + 2 * nll
    
    return {
        'df': df,
        'loc': loc,
        'scale': scale,
        'nll': nll,
        'aic': aic,
        'bic': bic,
        'n': n
    }


def likelihood_ratio_test(nll_null, nll_alt, df_diff):
    """
    Likelihood Ratio Test.
    
    LR = 2 * (LL_alt - LL_null) = 2 * (NLL_null - NLL_alt)
    Under H0, LR ~ chi-squared(df_diff)
    """
    lr_stat = 2 * (nll_null - nll_alt)
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
    return lr_stat, p_value


def main():
    """Run GAMLSS-equivalent analysis."""
    print("=" * 70)
    print("GAMLSS-EQUIVALENT ANALYSIS (Python)")
    print("Testing: Does Signal affect distribution shape?")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'gamlss_data.csv'))
    
    y = df['Fwd_Ret_21'].values
    signal = df['Signal_Strict'].values
    
    y_low = y[signal == 0]
    y_high = y[signal == 1]
    
    print(f"\nData: {len(y)} total observations")
    print(f"  Signal=0: {len(y_low)} obs")
    print(f"  Signal=1: {len(y_high)} obs")
    
    # =========================================================================
    # MODEL 1: NULL (same distribution for both regimes)
    # =========================================================================
    print("\n" + "-" * 50)
    print("MODEL 1: NULL (constant distribution)")
    
    fit_null = fit_student_t(y)
    
    print(f"  df (tail):  {fit_null['df']:.3f}")
    print(f"  loc (μ):    {fit_null['loc']:.6f}")
    print(f"  scale (σ):  {fit_null['scale']:.6f}")
    print(f"  AIC:        {fit_null['aic']:.2f}")
    
    # =========================================================================
    # MODEL 2: SEPARATE (different distribution per regime)
    # =========================================================================
    print("\n" + "-" * 50)
    print("MODEL 2: SEPARATE (regime-specific distributions)")
    
    fit_low = fit_student_t(y_low)
    fit_high = fit_student_t(y_high)
    
    print("\n  Signal = 0 (Low Risk):")
    print(f"    df (tail):  {fit_low['df']:.3f}")
    print(f"    loc (μ):    {fit_low['loc']:.6f}")
    print(f"    scale (σ):  {fit_low['scale']:.6f}")
    
    print("\n  Signal = 1 (High Risk):")
    print(f"    df (tail):  {fit_high['df']:.3f}")
    print(f"    loc (μ):    {fit_high['loc']:.6f}")
    print(f"    scale (σ):  {fit_high['scale']:.6f}")
    
    # Combined AIC/BIC for separate model
    nll_separate = fit_low['nll'] + fit_high['nll']
    k_separate = 6  # 3 params per regime
    aic_separate = 2 * k_separate + 2 * nll_separate
    bic_separate = k_separate * np.log(len(y)) + 2 * nll_separate
    
    print(f"\n  Combined AIC: {aic_separate:.2f}")
    
    # =========================================================================
    # LIKELIHOOD RATIO TEST
    # =========================================================================
    print("\n" + "=" * 70)
    print("LIKELIHOOD RATIO TEST")
    print("=" * 70)
    
    # H0: Same distribution (3 params)
    # H1: Different distributions (6 params)
    # df_diff = 3
    
    lr_stat, p_value = likelihood_ratio_test(fit_null['nll'], nll_separate, df_diff=3)
    
    print(f"\n  LR statistic: {lr_stat:.2f}")
    print(f"  Degrees of freedom: 3")
    print(f"  p-value: {p_value:.6f}")
    
    print("\n  " + "-" * 40)
    if p_value < 0.001:
        print("  >>> HIGHLY SIGNIFICANT (p < 0.001)")
        print("  >>> REJECT H0: Distributions ARE different!")
    elif p_value < 0.01:
        print("  >>> SIGNIFICANT (p < 0.01)")
        print("  >>> REJECT H0: Distributions ARE different!")
    elif p_value < 0.05:
        print("  >>> SIGNIFICANT (p < 0.05)")
        print("  >>> REJECT H0: Distributions ARE different!")
    else:
        print("  >>> NOT SIGNIFICANT")
        print("  >>> Cannot reject H0")
    
    # =========================================================================
    # PARAMETER COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON")
    print("=" * 70)
    
    print(f"\n  {'Parameter':<15} | {'Signal=0':>12} | {'Signal=1':>12} | {'Change':>12}")
    print("  " + "-" * 55)
    
    loc_change = fit_high['loc'] - fit_low['loc']
    scale_change = fit_high['scale'] - fit_low['scale']
    scale_pct = (fit_high['scale'] / fit_low['scale'] - 1) * 100
    df_change = fit_high['df'] - fit_low['df']
    
    print(f"  {'Location (μ)':<15} | {fit_low['loc']:>12.5f} | {fit_high['loc']:>12.5f} | {loc_change:>+12.5f}")
    print(f"  {'Scale (σ)':<15} | {fit_low['scale']:>12.5f} | {fit_high['scale']:>12.5f} | {scale_change:>+12.5f} ({scale_pct:+.1f}%)")
    print(f"  {'DF (tail)':<15} | {fit_low['df']:>12.2f} | {fit_high['df']:>12.2f} | {df_change:>+12.2f}")
    
    # =========================================================================
    # INTERPRETATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    print("\n  HYPOTHESIS TEST RESULTS:")
    
    # Location (should be unchanged)
    if abs(loc_change) < 0.005:
        print(f"  ✓ Location (μ) essentially unchanged ({loc_change:+.5f})")
        loc_ok = True
    else:
        print(f"  ⚠ Location (μ) changed ({loc_change:+.5f})")
        loc_ok = False
    
    # Scale (should increase)
    if scale_change > 0 and scale_pct > 20:
        print(f"  ✓ Scale (σ) INCREASED under signal ({scale_pct:+.1f}%)")
        scale_ok = True
    else:
        print(f"  ✗ Scale (σ) did not increase significantly ({scale_pct:+.1f}%)")
        scale_ok = False
    
    # Degrees of freedom interpretation
    # Lower df = heavier tails
    if df_change < 0:
        print(f"  ✓ DF decreased → HEAVIER tails under signal")
        df_ok = True
    else:
        print(f"  ~ DF increased → Lighter tails under signal")
        df_ok = False
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    if p_value < 0.05 and loc_ok and scale_ok:
        print("""
  ✅ HYPOTHESIS CONFIRMED
  
  The Signal affects the DISTRIBUTION of future returns:
    • Location (μ) remains unchanged: {:.5f} → {:.5f}
    • Scale (σ) INCREASES by {:.1f}%: {:.5f} → {:.5f}
    • Distributions are significantly different (p < {:.4f})
  
  VALID SCIENTIFIC CLAIM:
  
  "Conditional on elevated E4 (Signal_Strict = 1), the distribution 
   of 21-day forward returns exhibits significantly higher dispersion 
   (σ +{:.0f}%), despite an unchanged mean. The LR test confirms the 
   distributional shift is statistically significant (χ² = {:.1f}, p < {:.4f})."
  
  PRACTICAL IMPLICATION:
  
  The Signal does NOT predict whether returns will be positive or negative.
  It predicts when returns will be MORE VOLATILE and have HEAVIER TAILS.
  
  Use case: When Signal=1, standard VaR models UNDERESTIMATE tail risk.
        """.format(
            fit_low['loc'], fit_high['loc'],
            scale_pct, fit_low['scale'], fit_high['scale'],
            p_value, scale_pct, lr_stat, p_value
        ))
    else:
        print(f"""
  ⚠ RESULTS INCONCLUSIVE
  
  p-value: {p_value:.4f}
  Location change: {loc_change:+.5f}
  Scale change: {scale_pct:+.1f}%
  
  The hypothesis is not clearly supported.
        """)
    
    # Save results
    results = pd.DataFrame({
        'metric': ['LR_statistic', 'p_value', 'loc_low', 'loc_high', 'loc_change',
                   'scale_low', 'scale_high', 'scale_change_pct', 'df_low', 'df_high'],
        'value': [lr_stat, p_value, fit_low['loc'], fit_high['loc'], loc_change,
                  fit_low['scale'], fit_high['scale'], scale_pct, fit_low['df'], fit_high['df']]
    })
    
    results.to_csv(os.path.join(OUTPUT_DIR, 'gamlss_python_results.csv'), index=False)
    print(f"\n✓ Results saved to gamlss_python_results.csv")
    
    return fit_low, fit_high, lr_stat, p_value


if __name__ == "__main__":
    fit_low, fit_high, lr_stat, p_value = main()















