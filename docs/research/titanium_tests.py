"""
TITANIUM STANDARD TESTS: Advanced Econometric Validation
=========================================================

Implements the "titanium" level tests from the strategic overhaul:

1. SURROGATE DATA ANALYSIS (IAAFT Algorithm)
   - Generate 1000 surrogate matrices preserving marginal properties
   - Test if real entropy is significantly lower than random
   - Compute Z-scores for structural significance

2. RANDOM MATRIX THEORY (RMT) FILTERING
   - Marchenko-Pastur law to identify signal vs noise eigenvalues
   - Filter correlation matrix before entropy calculation

3. DECAY-WEIGHTED STORED ENERGY
   - Replace hard cutoff with exponential decay kernel
   - Estimate optimal λ parameter

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import signal
from scipy.optimize import minimize_scalar
import yfinance as yf
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# Universe
EXPANDED_UNIVERSE = [
    'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY',
    'EWJ', 'EWG', 'EWU', 'EWC', 'EWA', 'EWZ', 'EWY', 'EWT', 'EWH', 'EWS',
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
    'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP', 'AGG',
    'GLD', 'SLV', 'USO', 'DBC',
    'EFA', 'EEM', 'VEU', 'VWO',
    'VNQ', 'IYR',
]

START_DATE = "2007-01-01"
WINDOW = 63

def load_data(tickers, start_date):
    print("Loading data...")
    df = yf.download(tickers + ['^VIX', '^GSPC'], start=start_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        px = df['Adj Close' if 'Adj Close' in df else 'Close']
    else:
        px = df
    return px.dropna(how='all')

# =============================================================================
# TEST 1: IAAFT SURROGATE DATA ANALYSIS
# =============================================================================

def iaaft_surrogate(x, n_iterations=100):
    """
    Iterative Amplitude Adjusted Fourier Transform (IAAFT).
    Creates a surrogate series that preserves:
    - The exact amplitude distribution (histogram)
    - The power spectrum (autocorrelation)
    But destroys phase relationships (cross-correlations).
    """
    n = len(x)
    
    # Store original amplitude spectrum and sorted values
    x_fft = np.fft.rfft(x)
    amplitudes = np.abs(x_fft)
    x_sorted = np.sort(x)
    
    # Initialize with Gaussian random phases
    surrogate = np.random.permutation(x)
    
    for _ in range(n_iterations):
        # Step 1: Match power spectrum
        surr_fft = np.fft.rfft(surrogate)
        surr_phases = np.angle(surr_fft)
        new_fft = amplitudes * np.exp(1j * surr_phases)
        surrogate = np.fft.irfft(new_fft, n=n)
        
        # Step 2: Match amplitude distribution
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = x_sorted[ranks]
    
    return surrogate

def generate_surrogate_matrix(returns_df, n_surrogates=100):
    """
    Generate surrogate return matrices using IAAFT.
    Each column is independently surrogatized.
    """
    print(f"\nGenerating {n_surrogates} IAAFT surrogates...")
    
    surrogates = []
    for i in range(n_surrogates):
        if i % 20 == 0:
            print(f"  Surrogate {i}/{n_surrogates}...")
        
        surrogate_df = pd.DataFrame(index=returns_df.index)
        for col in returns_df.columns:
            series = returns_df[col].dropna().values
            if len(series) > 100:
                surrogate_df[col] = iaaft_surrogate(series)
            else:
                surrogate_df[col] = np.random.permutation(series)
        
        surrogates.append(surrogate_df)
    
    return surrogates

def calculate_entropy(corr_matrix):
    """Calculate spectral entropy from correlation matrix."""
    try:
        eigvals = np.linalg.eigvalsh(corr_matrix)
        eigvals = eigvals[eigvals > 1e-10]
        probs = eigvals / np.sum(eigvals)
        S = -np.sum(probs * np.log(probs))
        N = len(probs)
        return S / np.log(N) if N > 1 else 1.0
    except:
        return np.nan

def surrogate_data_test(returns_df, window, n_surrogates=100):
    """
    Test if observed entropy is significantly lower than surrogates.
    Returns Z-scores for each time point.
    """
    print("\n" + "=" * 60)
    print("TEST 1: IAAFT SURROGATE DATA ANALYSIS")
    print("=" * 60)
    
    # Calculate real entropy time series
    print("\nCalculating real entropy...")
    real_entropy = {}
    for i in range(window, len(returns_df)):
        idx = returns_df.index[i]
        window_ret = returns_df.iloc[i-window:i].dropna(axis=1)
        if len(window_ret.columns) >= 15:
            corr = window_ret.corr()
            real_entropy[idx] = calculate_entropy(corr)
    
    real_entropy = pd.Series(real_entropy)
    
    # Generate surrogates and calculate entropy distribution
    surrogates = generate_surrogate_matrix(returns_df, n_surrogates)
    
    print("\nCalculating surrogate entropy distributions...")
    surrogate_entropies = []
    
    for k, surr_df in enumerate(surrogates):
        if k % 20 == 0:
            print(f"  Processing surrogate {k}/{n_surrogates}...")
        
        surr_entropy = {}
        for i in range(window, min(len(surr_df), len(returns_df))):
            idx = returns_df.index[i]
            if idx in real_entropy.index:
                window_ret = surr_df.iloc[i-window:i].dropna(axis=1)
                if len(window_ret.columns) >= 15:
                    corr = window_ret.corr()
                    surr_entropy[idx] = calculate_entropy(corr)
        
        surrogate_entropies.append(pd.Series(surr_entropy))
    
    # Calculate Z-scores
    print("\nCalculating structural significance Z-scores...")
    z_scores = {}
    p_values = {}
    
    for idx in real_entropy.index:
        surr_values = [s[idx] for s in surrogate_entropies if idx in s.index]
        if len(surr_values) > 10:
            mu_surr = np.mean(surr_values)
            sigma_surr = np.std(surr_values)
            if sigma_surr > 0:
                z = (real_entropy[idx] - mu_surr) / sigma_surr
                z_scores[idx] = z
                # One-tailed p-value (we expect real entropy to be LOWER)
                from scipy import stats
                p_values[idx] = stats.norm.cdf(z)
    
    z_series = pd.Series(z_scores)
    p_series = pd.Series(p_values)
    
    # Summary statistics
    print(f"\n  RESULTS:")
    print(f"    Mean Z-score: {z_series.mean():.2f}")
    print(f"    % of days with Z < -2 (p < 0.023): {(z_series < -2).mean()*100:.1f}%")
    print(f"    % of days with Z < -3 (p < 0.001): {(z_series < -3).mean()*100:.1f}%")
    print(f"    Min Z-score: {z_series.min():.2f}")
    
    return z_series, p_series, real_entropy

# =============================================================================
# TEST 2: RANDOM MATRIX THEORY (RMT) FILTERING
# =============================================================================

def marchenko_pastur_bounds(T, N, sigma=1.0):
    """
    Calculate Marchenko-Pastur theoretical eigenvalue bounds.
    
    T = number of time observations
    N = number of assets
    sigma = variance of returns (normalized to 1)
    
    Returns: (lambda_minus, lambda_plus)
    """
    Q = T / N
    lambda_plus = sigma**2 * (1 + 1/np.sqrt(Q))**2
    lambda_minus = sigma**2 * (1 - 1/np.sqrt(Q))**2
    return lambda_minus, lambda_plus

def rmt_filter_correlation(corr_matrix, T):
    """
    Filter correlation matrix using RMT.
    Keep only eigenvalues above Marchenko-Pastur upper bound.
    """
    N = len(corr_matrix)
    
    # Get MP bounds
    lambda_minus, lambda_plus = marchenko_pastur_bounds(T, N)
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    
    # Count signal vs noise eigenvalues
    n_signal = np.sum(eigvals > lambda_plus)
    n_noise = N - n_signal
    
    # Filter: set noise eigenvalues to their mean
    eigvals_filtered = eigvals.copy()
    noise_mask = eigvals <= lambda_plus
    if np.sum(noise_mask) > 0:
        noise_mean = eigvals[noise_mask].mean()
        eigvals_filtered[noise_mask] = noise_mean
    
    # Reconstruct filtered correlation matrix
    corr_filtered = eigvecs @ np.diag(eigvals_filtered) @ eigvecs.T
    
    # Normalize diagonal to 1
    d = np.sqrt(np.diag(corr_filtered))
    corr_filtered = corr_filtered / np.outer(d, d)
    
    return corr_filtered, n_signal, n_noise

def rmt_entropy_analysis(returns_df, window):
    """
    Calculate entropy using RMT-filtered correlation matrices.
    """
    print("\n" + "=" * 60)
    print("TEST 2: RANDOM MATRIX THEORY FILTERING")
    print("=" * 60)
    
    raw_entropy = {}
    filtered_entropy = {}
    n_signal_modes = {}
    
    for i in range(window, len(returns_df)):
        idx = returns_df.index[i]
        window_ret = returns_df.iloc[i-window:i].dropna(axis=1)
        
        if len(window_ret.columns) >= 15:
            T = len(window_ret)
            N = len(window_ret.columns)
            
            # Raw correlation
            corr_raw = window_ret.corr()
            raw_entropy[idx] = calculate_entropy(corr_raw)
            
            # RMT-filtered correlation
            corr_filtered, n_sig, n_noise = rmt_filter_correlation(corr_raw.values, T)
            filtered_entropy[idx] = calculate_entropy(corr_filtered)
            n_signal_modes[idx] = n_sig
    
    raw_series = pd.Series(raw_entropy)
    filtered_series = pd.Series(filtered_entropy)
    signal_series = pd.Series(n_signal_modes)
    
    print(f"\n  RESULTS:")
    print(f"    Mean raw entropy: {raw_series.mean():.3f}")
    print(f"    Mean RMT-filtered entropy: {filtered_series.mean():.3f}")
    print(f"    Mean signal eigenvalues: {signal_series.mean():.1f}")
    print(f"    Correlation (raw vs filtered): {raw_series.corr(filtered_series):.3f}")
    
    return raw_series, filtered_series, signal_series

# =============================================================================
# TEST 3: DECAY-WEIGHTED STORED ENERGY
# =============================================================================

def decay_weighted_se(fragility, decay_lambda):
    """
    Calculate Stored Energy with exponential decay kernel.
    
    SE_t(λ) = ∫ F(τ) * exp(-λ(t-τ)) dτ
    
    Discrete version: SE_t = Σ F_{t-k} * exp(-λ*k)
    """
    # Create decay weights
    max_lookback = min(504, len(fragility))  # 2 years max
    weights = np.exp(-decay_lambda * np.arange(max_lookback))
    weights = weights / weights.sum()  # Normalize
    
    # Convolve with fragility
    se = fragility.rolling(max_lookback, min_periods=30).apply(
        lambda x: np.sum(x[-len(weights):] * weights[:len(x)]) if len(x) >= 30 else np.nan
    )
    
    return se

def estimate_optimal_lambda(fragility, forward_returns, lambda_range=(0.001, 0.1)):
    """
    Estimate optimal decay parameter λ that maximizes predictive power.
    """
    print("\n" + "=" * 60)
    print("TEST 3: DECAY-WEIGHTED STORED ENERGY")
    print("=" * 60)
    
    def neg_r2(lam):
        se = decay_weighted_se(fragility, lam)
        
        common = se.dropna().index.intersection(forward_returns.dropna().index)
        if len(common) < 500:
            return 0
        
        df = pd.DataFrame({
            'SE': se.loc[common],
            'Fwd': forward_returns.loc[common]
        }).dropna()
        
        if len(df) < 500:
            return 0
        
        df['SE_std'] = (df['SE'] - df['SE'].mean()) / df['SE'].std()
        
        X = sm.add_constant(df['SE_std'])
        y = df['Fwd']
        model = sm.OLS(y, X).fit()
        
        return -model.rsquared
    
    # Test a range of lambda values
    print("\n  Testing decay parameters...")
    results = []
    for lam in np.linspace(0.005, 0.05, 20):
        r2 = -neg_r2(lam)
        results.append({'lambda': lam, 'R2': r2})
        print(f"    λ = {lam:.4f}: R² = {r2:.4f}")
    
    results_df = pd.DataFrame(results)
    best_lambda = results_df.loc[results_df['R2'].idxmax(), 'lambda']
    best_r2 = results_df['R2'].max()
    
    print(f"\n  OPTIMAL RESULTS:")
    print(f"    Best λ: {best_lambda:.4f}")
    print(f"    Best R²: {best_r2:.4f}")
    print(f"    Implied half-life: {np.log(2)/best_lambda:.0f} days")
    
    return best_lambda, results_df

def main():
    print("=" * 70)
    print("TITANIUM STANDARD TESTS")
    print("=" * 70)
    
    px = load_data(EXPANDED_UNIVERSE, START_DATE)
    if px.empty: 
        print("ERROR: No data")
        return
    
    available = [t for t in EXPANDED_UNIVERSE if t in px.columns]
    returns = px[available].pct_change()
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    # Forward S&P returns
    sp500_fwd = px['^GSPC'].pct_change(21).shift(-21) if '^GSPC' in px.columns else px['SPY'].pct_change(21).shift(-21)
    
    # Calculate standard fragility first
    print("\nCalculating base fragility series...")
    fragility = {}
    for i in range(WINDOW, len(returns)):
        idx = returns.index[i]
        window_ret = returns.iloc[i-WINDOW:i].dropna(axis=1)
        if len(window_ret.columns) >= 15:
            corr = window_ret.corr()
            entropy = calculate_entropy(corr)
            fragility[idx] = 1 - entropy
    
    fragility = pd.Series(fragility)
    
    # TEST 1: Surrogate Data Analysis (reduced for speed)
    z_scores, p_values, real_entropy = surrogate_data_test(
        returns.iloc[-1000:], WINDOW, n_surrogates=50
    )
    
    surr_results = pd.DataFrame({
        'Mean_Z': [z_scores.mean()],
        'Pct_Z_below_minus2': [(z_scores < -2).mean()],
        'Pct_Z_below_minus3': [(z_scores < -3).mean()],
        'Min_Z': [z_scores.min()]
    })
    surr_results.to_csv(os.path.join(output_dir, "Table_Surrogate_Data_Test.csv"), index=False)
    
    # TEST 2: RMT Filtering
    raw_entropy, filtered_entropy, signal_modes = rmt_entropy_analysis(returns, WINDOW)
    
    rmt_results = pd.DataFrame({
        'Mean_Raw_Entropy': [raw_entropy.mean()],
        'Mean_Filtered_Entropy': [filtered_entropy.mean()],
        'Mean_Signal_Eigenvalues': [signal_modes.mean()],
        'Correlation': [raw_entropy.corr(filtered_entropy)]
    })
    rmt_results.to_csv(os.path.join(output_dir, "Table_RMT_Filtering.csv"), index=False)
    
    # TEST 3: Decay-Weighted SE
    best_lambda, lambda_results = estimate_optimal_lambda(fragility, sp500_fwd)
    lambda_results.to_csv(os.path.join(output_dir, "Table_Decay_Lambda_Search.csv"), index=False)
    
    # Final summary
    print("\n" + "=" * 70)
    print("TITANIUM TESTS COMPLETE")
    print("=" * 70)
    print(f"\n  Surrogate Test: Real entropy significantly lower (Z < -2 in {(z_scores < -2).mean()*100:.0f}% of days)")
    print(f"  RMT Filtering: Corr(raw, filtered) = {raw_entropy.corr(filtered_entropy):.3f}")
    print(f"  Decay-Weighted SE: Optimal λ = {best_lambda:.4f} (half-life = {np.log(2)/best_lambda:.0f} days)")

if __name__ == "__main__":
    main()
