"""
Reviewer Response Analysis
==========================
This script addresses key reviewer concerns with new empirical analyses.

Run with: python reviewer_response_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: ROLLING THRESHOLD ROBUSTNESS
# ============================================================================
# Reviewer concern: Threshold drifts from 0.22 to 0.12. How to use in real-time?
# Response: Show OOS performance using only prior-period threshold estimates

def compute_spectral_entropy(returns_df, window=63):
    """Compute spectral entropy from rolling correlation matrix."""
    entropy_series = []
    dates = []
    
    for i in range(window, len(returns_df)):
        window_data = returns_df.iloc[i-window:i]
        if window_data.dropna().shape[0] < window * 0.8:
            entropy_series.append(np.nan)
            dates.append(returns_df.index[i])
            continue
            
        corr_matrix = window_data.corr()
        eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
        eigenvalues = eigenvalues[eigenvalues > 0]
        
        # Normalize to probabilities
        p = eigenvalues / eigenvalues.sum()
        # Compute entropy
        entropy = -np.sum(p * np.log(p)) / np.log(len(p))
        
        entropy_series.append(entropy)
        dates.append(returns_df.index[i])
    
    return pd.Series(entropy_series, index=dates, name='Entropy')

def compute_asf(entropy, theta=0.995):
    """Compute Accumulated Spectral Fragility."""
    fragility = 1 - entropy
    asf = fragility.ewm(alpha=1-theta, adjust=False).mean()
    return asf

def estimate_threshold_hansen(asf, forward_risk, connectivity, grid_size=100):
    """
    Estimate threshold using Hansen (2000) grid search.
    Returns: threshold, beta_low, beta_high, test_statistic
    """
    # Grid of potential thresholds (15th to 85th percentile)
    c_grid = np.percentile(connectivity.dropna(), np.linspace(15, 85, grid_size))
    
    best_ssr = np.inf
    best_tau = None
    
    for tau in c_grid:
        low_mask = connectivity <= tau
        high_mask = connectivity > tau
        
        if low_mask.sum() < 30 or high_mask.sum() < 30:
            continue
        
        # Regime-specific regressions
        try:
            # Low regime
            X_low = asf[low_mask].values.reshape(-1, 1)
            y_low = forward_risk[low_mask].values
            valid_low = ~np.isnan(X_low.flatten()) & ~np.isnan(y_low)
            if valid_low.sum() < 20:
                continue
            beta_low = np.linalg.lstsq(
                np.column_stack([np.ones(valid_low.sum()), X_low[valid_low]]),
                y_low[valid_low], rcond=None
            )[0]
            resid_low = y_low[valid_low] - (beta_low[0] + beta_low[1] * X_low[valid_low].flatten())
            
            # High regime
            X_high = asf[high_mask].values.reshape(-1, 1)
            y_high = forward_risk[high_mask].values
            valid_high = ~np.isnan(X_high.flatten()) & ~np.isnan(y_high)
            if valid_high.sum() < 20:
                continue
            beta_high = np.linalg.lstsq(
                np.column_stack([np.ones(valid_high.sum()), X_high[valid_high]]),
                y_high[valid_high], rcond=None
            )[0]
            resid_high = y_high[valid_high] - (beta_high[0] + beta_high[1] * X_high[valid_high].flatten())
            
            # Total SSR
            ssr = np.sum(resid_low**2) + np.sum(resid_high**2)
            
            if ssr < best_ssr:
                best_ssr = ssr
                best_tau = tau
                best_beta_low = beta_low[1]
                best_beta_high = beta_high[1]
                
        except:
            continue
    
    return best_tau, best_beta_low, best_beta_high

def rolling_threshold_analysis(returns_df, estimation_window_years=5):
    """
    Main analysis: Estimate threshold using rolling window, apply to next period.
    This addresses the reviewer's concern about real-time usability.
    """
    print("=" * 70)
    print("ROLLING THRESHOLD ROBUSTNESS ANALYSIS")
    print("=" * 70)
    print(f"\nEstimation window: {estimation_window_years} years")
    print("Method: Estimate τ on [t-5y, t], apply to predict [t, t+1y]\n")
    
    # Compute base metrics
    entropy = compute_spectral_entropy(returns_df)
    asf = compute_asf(entropy)
    connectivity = returns_df.rolling(63).corr().groupby(level=0).mean().mean(axis=1)
    
    # Forward 1-month max drawdown
    cumret = (1 + returns_df.mean(axis=1)).cumprod()
    running_max = cumret.expanding().max()
    drawdown = (cumret - running_max) / running_max
    forward_risk = drawdown.rolling(21).min().shift(-21).abs()
    
    # Align all series
    df = pd.DataFrame({
        'asf': asf,
        'connectivity': connectivity,
        'forward_risk': forward_risk
    }).dropna()
    
    # Rolling estimation
    results = []
    estimation_days = estimation_window_years * 252
    
    years = df.index.year.unique()
    
    for year in years[estimation_window_years:]:
        # Estimation period: prior 5 years
        est_end = pd.Timestamp(f"{year-1}-12-31")
        est_start = pd.Timestamp(f"{year-estimation_window_years}-01-01")
        
        est_data = df[(df.index >= est_start) & (df.index <= est_end)]
        
        if len(est_data) < 252 * 3:
            continue
        
        # Estimate threshold on historical data
        tau, beta_low, beta_high = estimate_threshold_hansen(
            est_data['asf'], 
            est_data['forward_risk'],
            est_data['connectivity']
        )
        
        if tau is None:
            continue
        
        # Apply to current year (out-of-sample)
        oos_data = df[df.index.year == year]
        
        if len(oos_data) < 50:
            continue
        
        # Classify regimes using estimated threshold
        low_regime = oos_data['connectivity'] <= tau
        high_regime = oos_data['connectivity'] > tau
        
        # Compute OOS correlations by regime
        if low_regime.sum() > 10:
            corr_low = oos_data.loc[low_regime, ['asf', 'forward_risk']].corr().iloc[0, 1]
        else:
            corr_low = np.nan
            
        if high_regime.sum() > 10:
            corr_high = oos_data.loc[high_regime, ['asf', 'forward_risk']].corr().iloc[0, 1]
        else:
            corr_high = np.nan
        
        results.append({
            'year': year,
            'estimated_tau': tau,
            'beta_low': beta_low,
            'beta_high': beta_high,
            'oos_corr_low': corr_low,
            'oos_corr_high': corr_high,
            'sign_inverts': (beta_low > 0 and beta_high < 0)
        })
    
    results_df = pd.DataFrame(results)
    
    print("\nRolling Threshold Estimates by Year:")
    print("-" * 70)
    print(results_df.to_string(index=False))
    
    print("\n\nSUMMARY:")
    print("-" * 70)
    print(f"Mean estimated τ: {results_df['estimated_tau'].mean():.3f}")
    print(f"Std of τ: {results_df['estimated_tau'].std():.3f}")
    print(f"% of years with sign inversion: {results_df['sign_inverts'].mean()*100:.1f}%")
    print(f"Mean OOS correlation (low regime): {results_df['oos_corr_low'].mean():.3f}")
    print(f"Mean OOS correlation (high regime): {results_df['oos_corr_high'].mean():.3f}")
    
    return results_df

# ============================================================================
# SECTION 2: DATA SPLICING ANALYSIS
# ============================================================================
# Reviewer concern: Did H_t shift due to universe change?

def data_splicing_analysis(global_macro_returns, etf_returns, overlap_start='2007-01-01'):
    """
    Compare entropy between Global Macro and ETF datasets in overlap period.
    """
    print("\n" + "=" * 70)
    print("DATA SPLICING ANALYSIS: OVERLAP PERIOD COMPARISON")
    print("=" * 70)
    
    # Compute entropy for both datasets
    entropy_macro = compute_spectral_entropy(global_macro_returns)
    entropy_etf = compute_spectral_entropy(etf_returns)
    
    # Overlap period
    overlap_start = pd.Timestamp(overlap_start)
    
    entropy_macro_overlap = entropy_macro[entropy_macro.index >= overlap_start]
    entropy_etf_overlap = entropy_etf[entropy_etf.index >= overlap_start]
    
    # Align to common dates
    common_dates = entropy_macro_overlap.index.intersection(entropy_etf_overlap.index)
    
    if len(common_dates) < 100:
        print("WARNING: Insufficient overlap for robust comparison")
        return None
    
    macro_aligned = entropy_macro_overlap.loc[common_dates]
    etf_aligned = entropy_etf_overlap.loc[common_dates]
    
    print(f"\nOverlap period: {common_dates[0].date()} to {common_dates[-1].date()}")
    print(f"Number of observations: {len(common_dates)}")
    
    print("\nDescriptive Statistics:")
    print("-" * 50)
    print(f"{'Statistic':<20} {'Global Macro':>15} {'ETF':>15}")
    print("-" * 50)
    print(f"{'Mean':>20} {macro_aligned.mean():>15.4f} {etf_aligned.mean():>15.4f}")
    print(f"{'Std':>20} {macro_aligned.std():>15.4f} {etf_aligned.std():>15.4f}")
    print(f"{'Min':>20} {macro_aligned.min():>15.4f} {etf_aligned.min():>15.4f}")
    print(f"{'Max':>20} {macro_aligned.max():>15.4f} {etf_aligned.max():>15.4f}")
    
    # Correlation between the two
    correlation = macro_aligned.corr(etf_aligned)
    print(f"\nCorrelation: {correlation:.4f}")
    
    # Test for significant difference in means
    t_stat, p_value = stats.ttest_rel(macro_aligned, etf_aligned)
    print(f"\nPaired t-test (H0: means are equal):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print("  → Cannot reject H0: No significant difference in entropy levels")
    else:
        print("  → Reject H0: Significant difference detected")
        print("  → Note: Include panel-standardization in robustness")
    
    return {
        'correlation': correlation,
        't_stat': t_stat,
        'p_value': p_value,
        'mean_diff': macro_aligned.mean() - etf_aligned.mean()
    }

# ============================================================================
# SECTION 3: HYSTERESIS SIGNIFICANCE TEST
# ============================================================================

def hysteresis_significance_test(asf, drawdowns):
    """
    Test if the "loading phase" significantly precedes the "unloading phase".
    """
    print("\n" + "=" * 70)
    print("HYSTERESIS SIGNIFICANCE TEST")
    print("=" * 70)
    
    # Compute changes
    delta_asf = asf.diff()
    delta_dd = drawdowns.diff()
    
    # Cross-correlation analysis
    print("\nCross-Correlation: Δ(ASF) leads Δ(Drawdowns)")
    print("-" * 50)
    print(f"{'Lag (days)':<15} {'Correlation':>15} {'p-value':>15}")
    print("-" * 50)
    
    for lag in [-21, -10, -5, 0, 5, 10, 21]:
        if lag < 0:
            x = delta_asf.shift(-lag).dropna()
            y = delta_dd.loc[x.index].dropna()
        else:
            x = delta_asf.dropna()
            y = delta_dd.shift(lag).loc[x.index].dropna()
        
        common = x.index.intersection(y.index)
        if len(common) < 50:
            continue
        
        corr, pval = stats.pearsonr(x.loc[common], y.loc[common])
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"{lag:>15} {corr:>15.4f} {pval:>14.4f} {sig}")
    
    # Granger causality interpretation
    print("\n\nInterpretation (based on existing Granger causality results):")
    print("-" * 50)
    print("From Table A.3 (Granger Causality):")
    print("  - ASF → Tail Risk significant at lags 2-5 (F > 3.78, p < 0.002)")
    print("  - This confirms loading (ASF accumulation) precedes unloading (crises)")
    print("  - The asymmetry is not due to simple mean reversion")
    
    return True

# ============================================================================
# SECTION 4: PERSISTENCE PARAMETER JUSTIFICATION
# ============================================================================

def persistence_justification():
    """
    Provide economic rationale for theta = 0.995 (6-month half-life).
    """
    print("\n" + "=" * 70)
    print("ECONOMIC JUSTIFICATION FOR theta = 0.995")
    print("=" * 70)
    
    theta = 0.995
    half_life_days = np.log(0.5) / np.log(theta)
    half_life_months = half_life_days / 21
    
    print(f"\nPersistence parameter: theta = {theta}")
    print(f"Half-life: {half_life_days:.0f} trading days (approx {half_life_months:.1f} months)")
    
    print("\n\nEconomic Rationale:")
    print("-" * 50)
    print("""
1. INSTITUTIONAL REPORTING CYCLES
   - Quarterly reports (13F filings): 3 months
   - Semi-annual rebalancing: 6 months
   - Annual performance reviews: 12 months
   -> 6-month half-life captures semi-annual adjustment cycle

2. BALANCE SHEET ADJUSTMENT SPEED
   - He & Krishnamurthy (2013) document that intermediary 
     balance-sheet constraints adjust over months, not days
   - Adrian & Shin (2010) show leverage targeting with 
     similar adjustment horizons
   -> theta = 0.995 consistent with intermediary dynamics

3. POSITION BUILDUP DYNAMICS
   - Crowded trades develop over quarters as momentum 
     investors accumulate positions
   - Leverage constraints bind gradually as VaR limits 
     are approached
   -> Slow memory appropriate for "stored fragility"

4. EMPIRICAL ROBUSTNESS
   - From sensitivity analysis (Appendix Figure A.6):
     Sign inversion robust for theta in [0.90, 0.999]
   - Optimal theta depends on forecast horizon
   -> theta = 0.995 is central tendency, not knife-edge
""")
    
    return {
        'theta': theta,
        'half_life_days': half_life_days,
        'half_life_months': half_life_months
    }

# ============================================================================
# SECTION 5: COORDINATION REGIME IMPLICATIONS
# ============================================================================

def coordination_implications():
    """
    Clarify practical implications for diversification.
    """
    print("\n" + "=" * 70)
    print("PRACTICAL IMPLICATIONS: COORDINATION REGIME")
    print("=" * 70)
    
    print("""
REVIEWER CONCERN:
"Does high ASF mean investors should avoid diversifying?"

CLARIFICATION:
The finding is NOT that diversification is bad, but that:

1. DIVERSIFICATION BENEFITS ARE REGIME-DEPENDENT
   - In low-connectivity regimes: Traditional diversification works
   - In high-connectivity regimes: Correlations are already high,
     so "diversification" provides less marginal benefit

2. THE RISK IS COORDINATION BREAKDOWN, NOT DIVERSIFICATION
   - High ASF indicates synchronized markets
   - The danger is when synchronization breaks down
   - Diversification doesn't protect against this specific risk

3. PRACTICAL IMPLICATION
   NOT: "Avoid diversifying when ASF is high"
   BUT: "Recognize that standard diversification metrics may 
        overstate protection when ASF is high"

4. MONITORING RECOMMENDATION
   - Track ASF alongside VaR/volatility
   - When ASF is elevated, consider:
     a) Stress-testing for correlation breakdown
     b) Reducing leverage (not diversification)
     c) Maintaining liquidity buffers

5. WHAT INVESTORS SHOULD DO
   - Continue diversifying (it's never harmful)
   - But adjust risk models to account for regime
   - Consider structural indicators in addition to volatility
""")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# REVIEWER RESPONSE ANALYSIS")
    print("# Addresses key concerns from peer review")
    print("#" * 70)
    
    # Note: This script requires price data to run the quantitative analyses.
    # The functions above are ready to use with appropriate data inputs.
    
    # For now, run the qualitative responses:
    persistence_justification()
    coordination_implications()
    
    print("\n" + "=" * 70)
    print("QUANTITATIVE ANALYSES (require data)")
    print("=" * 70)
    print("""
To run the quantitative analyses, load your price data and call:

1. Rolling Threshold:
   results = rolling_threshold_analysis(returns_df, estimation_window_years=5)

2. Data Splicing:
   comparison = data_splicing_analysis(global_macro_returns, etf_returns)

3. Hysteresis Test:
   hysteresis_significance_test(asf_series, drawdown_series)
""")
    
    print("\n" + "#" * 70)
    print("# END OF ANALYSIS")
    print("#" * 70)
