"""
Prepare Data for GAMLSS Analysis
================================

Export data from Python to CSV for R analysis.

The CORRECT hypothesis:
  - Signal does NOT affect μ (mean returns)
  - Signal DOES affect σ (volatility) and τ (tail heaviness)

This is a DISTRIBUTIONAL test, not a mean prediction test.

Author: Tomás Basaure
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
START_DATE = "2005-01-01"
FWD_WINDOW = 21


def build_gamlss_data():
    """Build dataset for GAMLSS analysis."""
    print("=" * 70)
    print("PREPARING DATA FOR GAMLSS ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    
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
    
    # Align
    common = ret.index.intersection(ret_hyg.index)
    r = ret.loc[common]
    
    # E4 (the only component that matters)
    v5 = r.rolling(5).std() * np.sqrt(252)
    v21 = r.rolling(21).std() * np.sqrt(252)
    v63 = r.rolling(63).std() * np.sqrt(252)
    v_cred = ret_hyg.loc[common].rolling(42).std() * np.sqrt(252)
    
    E4_raw = 0.20*v5 + 0.30*v21 + 0.25*v63 + 0.25*v_cred
    E4 = E4_raw.rolling(252).rank(pct=True)
    
    # Forward returns (the target)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FWD_WINDOW)
    fwd_ret = r.rolling(window=indexer).sum()
    
    # Signal_Strict (binary: E4 > 0.8)
    signal_strict = (E4 > 0.8).astype(int)
    
    # Also create continuous E4 for more power
    # And quartile version
    
    # Build DataFrame
    df = pd.DataFrame({
        'Date': r.index,
        'Return': r.values,
        'Fwd_Ret_21': fwd_ret.values,
        'E4': E4.values,
        'E4_raw': E4_raw.values,
        'Signal_Strict': signal_strict.values,
    }).dropna()
    
    # Also create E4 quartiles
    df['E4_Q'] = pd.qcut(df['E4'], 4, labels=[1, 2, 3, 4]).astype(int)
    
    print(f"\nDataset created: {len(df)} observations")
    print(f"\nSignal distribution:")
    print(df['Signal_Strict'].value_counts())
    
    # Quick empirical check
    print("\n" + "-" * 50)
    print("EMPIRICAL PREVIEW (what GAMLSS will test):")
    print("-" * 50)
    
    low_signal = df[df['Signal_Strict'] == 0]['Fwd_Ret_21']
    high_signal = df[df['Signal_Strict'] == 1]['Fwd_Ret_21']
    
    print(f"\n{'Statistic':<20} | {'Signal=0':>12} | {'Signal=1':>12}")
    print("-" * 50)
    print(f"{'N':<20} | {len(low_signal):>12} | {len(high_signal):>12}")
    print(f"{'Mean':<20} | {low_signal.mean():>12.4f} | {high_signal.mean():>12.4f}")
    print(f"{'Std Dev':<20} | {low_signal.std():>12.4f} | {high_signal.std():>12.4f}")
    print(f"{'Skewness':<20} | {stats.skew(low_signal):>12.4f} | {stats.skew(high_signal):>12.4f}")
    print(f"{'Kurtosis':<20} | {stats.kurtosis(low_signal):>12.4f} | {stats.kurtosis(high_signal):>12.4f}")
    print(f"{'5th percentile':<20} | {low_signal.quantile(0.05):>12.4f} | {high_signal.quantile(0.05):>12.4f}")
    print(f"{'1st percentile':<20} | {low_signal.quantile(0.01):>12.4f} | {high_signal.quantile(0.01):>12.4f}")
    print(f"{'Min':<20} | {low_signal.min():>12.4f} | {high_signal.min():>12.4f}")
    
    # Key test: are the distributions different?
    print("\n" + "-" * 50)
    print("PRELIMINARY TESTS:")
    
    # Levene's test for equal variances
    stat_levene, p_levene = stats.levene(low_signal, high_signal)
    print(f"\nLevene's test (equal variances):")
    print(f"  Statistic: {stat_levene:.4f}, p-value: {p_levene:.4f}")
    if p_levene < 0.05:
        print(f"  >>> VARIANCES ARE DIFFERENT! (σ effect)")
    
    # Kolmogorov-Smirnov test for distribution difference
    stat_ks, p_ks = stats.ks_2samp(low_signal, high_signal)
    print(f"\nKS test (distribution equality):")
    print(f"  Statistic: {stat_ks:.4f}, p-value: {p_ks:.4f}")
    if p_ks < 0.05:
        print(f"  >>> DISTRIBUTIONS ARE DIFFERENT!")
    
    # T-test for means (should NOT be significant per hypothesis)
    stat_t, p_t = stats.ttest_ind(low_signal, high_signal)
    print(f"\nT-test (equal means):")
    print(f"  Statistic: {stat_t:.4f}, p-value: {p_t:.4f}")
    if p_t > 0.05:
        print(f"  >>> Means are NOT significantly different (as hypothesized)")
    else:
        print(f"  >>> Means ARE different")
    
    # Save for R
    output_path = os.path.join(OUTPUT_DIR, 'gamlss_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Data saved to: {output_path}")
    
    # Summary for R
    print("\n" + "=" * 70)
    print("READY FOR GAMLSS ANALYSIS")
    print("=" * 70)
    print("""
To run the GAMLSS test in R:

1. Open R or RStudio
2. Set working directory: setwd("{}") 
3. Run: source("gamlss_test.R")

The test will determine if Signal_Strict affects:
  - σ (volatility) of future returns
  - τ (tail heaviness) of future returns

WITHOUT affecting μ (mean).
    """.format(OUTPUT_DIR.replace("\\", "/")))
    
    return df


def quick_python_gamlss_approximation(df):
    """
    Quick approximation of GAMLSS using Python.
    NOT as rigorous as R, but gives preliminary results.
    """
    print("\n" + "=" * 70)
    print("PYTHON APPROXIMATION (Quick Check)")
    print("=" * 70)
    
    from scipy.stats import t as student_t
    
    low = df[df['Signal_Strict'] == 0]['Fwd_Ret_21']
    high = df[df['Signal_Strict'] == 1]['Fwd_Ret_21']
    
    # Fit Student-t to each regime
    print("\nFitting Student-t distribution to each regime...")
    
    # Low signal regime
    df_low, loc_low, scale_low = student_t.fit(low)
    print(f"\nSignal = 0 (Low risk):")
    print(f"  df (tail thickness): {df_low:.2f}")
    print(f"  location (μ proxy):  {loc_low:.4f}")
    print(f"  scale (σ proxy):     {scale_low:.4f}")
    
    # High signal regime
    df_high, loc_high, scale_high = student_t.fit(high)
    print(f"\nSignal = 1 (High risk):")
    print(f"  df (tail thickness): {df_high:.2f}")
    print(f"  location (μ proxy):  {loc_high:.4f}")
    print(f"  scale (σ proxy):     {scale_high:.4f}")
    
    # Compare
    print("\n" + "-" * 50)
    print("PARAMETER CHANGES:")
    print(f"  Location change: {loc_high - loc_low:+.4f} (should be ~0)")
    print(f"  Scale change:    {scale_high - scale_low:+.4f} (should be positive)")
    print(f"  DF change:       {df_high - df_low:+.2f} (lower df = heavier tails)")
    
    # Interpretation
    print("\n" + "-" * 50)
    print("PRELIMINARY VERDICT:")
    
    if abs(loc_high - loc_low) < 0.01 and scale_high > scale_low * 1.1:
        print("  ✓ SUPPORTS HYPOTHESIS")
        print("    - Location (mean) essentially unchanged")
        print("    - Scale (volatility) HIGHER under signal")
        if df_high < df_low:
            print("    - Degrees of freedom LOWER (heavier tails)")
    else:
        print("  ? Inconclusive - run full GAMLSS in R")
    
    return {
        'df_low': df_low, 'loc_low': loc_low, 'scale_low': scale_low,
        'df_high': df_high, 'loc_high': loc_high, 'scale_high': scale_high
    }


if __name__ == "__main__":
    df = build_gamlss_data()
    fit_params = quick_python_gamlss_approximation(df)















