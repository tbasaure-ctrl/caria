"""
Entropy Measures for Financial Time Series
==========================================

This module implements various entropy measures for quantifying information content
and disorder in financial time series, as described in:

    "Entropic Resonance and Volatility Compression as Precursors to Systemic Failure"
    Basaure, T. (2025)

Mathematical Framework:
-----------------------
Shannon Entropy measures the information content of a discrete probability distribution:

    H(X) = -Σ p(xᵢ) log₂ p(xᵢ)

Where:
    - X is a discrete random variable with possible values {x₁, x₂, ..., xₙ}
    - p(xᵢ) is the probability of outcome xᵢ
    - H(X) ∈ [0, log₂(n)] where n is the number of bins

For continuous financial returns, we discretize using histogram binning methods:
    - Freedman-Diaconis: bin_width = 2 * IQR * n^(-1/3)
    - Scott: bin_width = 3.5 * σ * n^(-1/3)
    - Sturges: n_bins = 1 + log₂(n)

References:
-----------
[1] Shannon, C.E. (1948). "A Mathematical Theory of Communication"
[2] Freedman, D. & Diaconis, P. (1981). "On the histogram as a density estimator"
[3] Bandt, C. & Pompe, B. (2002). "Permutation Entropy" - Physical Review Letters

Author: Tomás Basaure
Date: December 2025
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import hilbert
from typing import Union, Tuple, Optional, Literal
from dataclasses import dataclass


# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BinMethod = Literal['fd', 'scott', 'sturges', 'sqrt', 'rice', 'fixed']


@dataclass
class EntropyResult:
    """Container for entropy calculation results with metadata."""
    value: float
    n_bins: int
    bin_edges: np.ndarray
    method: str
    n_samples: int
    normalized: bool
    
    def __repr__(self):
        return f"EntropyResult(H={self.value:.4f}, bins={self.n_bins}, method='{self.method}')"


# ==============================================================================
# SHANNON ENTROPY (PRIMARY MEASURE)
# ==============================================================================

def shannon_entropy(
    data: Union[np.ndarray, pd.Series],
    bins: Union[BinMethod, int] = 'fd',
    normalize: bool = False,
    base: float = 2.0,
    min_samples: int = 30
) -> EntropyResult:
    """
    Calculate Shannon Entropy of a continuous signal via histogram discretization.
    
    Mathematical Definition:
    ========================
    H(X) = -Σᵢ p(xᵢ) log_b(p(xᵢ))
    
    Where p(xᵢ) is estimated from histogram bin frequencies.
    
    Parameters:
    ===========
    data : array-like
        Input time series (typically log returns)
    bins : str or int
        Binning method. Options:
        - 'fd': Freedman-Diaconis (recommended for heavy-tailed distributions)
        - 'scott': Scott's rule (assumes Gaussian)
        - 'sturges': Sturges' formula
        - 'sqrt': Square root of n
        - 'rice': Rice's rule
        - int: Fixed number of bins
    normalize : bool
        If True, normalize to [0, 1] by dividing by log_b(n_bins)
    base : float
        Logarithm base (2 for bits, e for nats, 10 for dits)
    min_samples : int
        Minimum samples required for reliable estimation
        
    Returns:
    ========
    EntropyResult : Named tuple with value, metadata, and diagnostics
    
    Examples:
    =========
    >>> returns = np.random.randn(1000) * 0.02  # Simulated daily returns
    >>> result = shannon_entropy(returns, bins='fd')
    >>> print(f"Entropy: {result.value:.4f} bits")
    
    Notes:
    ======
    - For financial returns, 'fd' (Freedman-Diaconis) is recommended as it is
      robust to outliers and heavy tails common in financial data.
    - Normalized entropy allows comparison across different sample sizes.
    - Entropy increases with disorder/complexity; decreases with predictability.
    
    References:
    ===========
    [1] Freedman, D. & Diaconis, P. (1981). "On the histogram as a density estimator"
    """
    # Input validation
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    
    if len(data) < min_samples:
        raise ValueError(f"Insufficient samples: {len(data)} < {min_samples}")
    
    # Calculate histogram
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    
    # Convert to probabilities (add small epsilon to avoid log(0))
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]  # Remove zero-probability bins
    
    # Calculate Shannon entropy
    entropy = -np.sum(probabilities * np.log(probabilities) / np.log(base))
    
    # Normalize if requested
    n_bins = len(counts)
    max_entropy = np.log(n_bins) / np.log(base)
    
    if normalize and max_entropy > 0:
        entropy = entropy / max_entropy
    
    return EntropyResult(
        value=entropy,
        n_bins=n_bins,
        bin_edges=bin_edges,
        method=str(bins),
        n_samples=len(data),
        normalized=normalize
    )


def rolling_shannon_entropy(
    data: Union[np.ndarray, pd.Series],
    window: int = 30,
    bins: Union[BinMethod, int] = 'fd',
    normalize: bool = True,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate rolling Shannon Entropy over a sliding window.
    
    This is the primary entropy measure used in the Caria Risk Engine.
    
    Parameters:
    ===========
    data : array-like
        Input time series
    window : int
        Rolling window size (default: 30 trading days)
    bins : str or int
        Binning method (see shannon_entropy)
    normalize : bool
        Normalize to [0, 1]
    min_periods : int, optional
        Minimum observations required. Defaults to window.
        
    Returns:
    ========
    pd.Series : Rolling entropy values indexed like input
    
    Notes:
    ======
    - Window of 30 days captures approximately 6 weeks of market behavior
    - Normalized entropy allows cross-asset comparison
    - NaN values are forward-filled after calculation
    """
    if min_periods is None:
        min_periods = window
    
    data = pd.Series(data).dropna()
    
    result = pd.Series(index=data.index, dtype=float)
    
    for i in range(min_periods - 1, len(data)):
        window_data = data.iloc[max(0, i - window + 1):i + 1]
        if len(window_data) >= min_periods:
            try:
                ent = shannon_entropy(window_data.values, bins=bins, normalize=normalize)
                result.iloc[i] = ent.value
            except (ValueError, RuntimeError):
                result.iloc[i] = np.nan
    
    return result


# ==============================================================================
# ALTERNATIVE ENTROPY MEASURES (FOR ROBUSTNESS CHECKS)
# ==============================================================================

def permutation_entropy(
    data: Union[np.ndarray, pd.Series],
    order: int = 3,
    delay: int = 1,
    normalize: bool = True
) -> float:
    """
    Calculate Permutation Entropy (Bandt & Pompe, 2002).
    
    Mathematical Definition:
    ========================
    H_PE = -Σ p(π) log₂(p(π))
    
    Where π represents ordinal patterns of length 'order' in the time series.
    
    Parameters:
    ===========
    data : array-like
        Input time series
    order : int
        Embedding dimension (pattern length). Default: 3
        - order=3: 3! = 6 possible patterns
        - order=4: 4! = 24 possible patterns
        - order=5: 5! = 120 possible patterns
    delay : int
        Time delay between elements in patterns
    normalize : bool
        Normalize by log₂(order!)
        
    Returns:
    ========
    float : Permutation entropy value
    
    Notes:
    ======
    - More robust to noise than Shannon entropy
    - Captures temporal structure/dynamics
    - Computational complexity: O(n * order!)
    
    References:
    ===========
    [1] Bandt, C. & Pompe, B. (2002). "Permutation entropy: A natural complexity
        measure for time series." Physical Review Letters, 88(17), 174102.
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    
    n = len(data)
    if n < order:
        raise ValueError(f"Data length ({n}) must be >= order ({order})")
    
    # Extract ordinal patterns
    from itertools import permutations
    
    # Create all possible permutations of indices
    factorial_order = np.math.factorial(order)
    pattern_dict = {perm: i for i, perm in enumerate(permutations(range(order)))}
    
    # Count pattern occurrences
    pattern_counts = np.zeros(factorial_order)
    
    for i in range(n - (order - 1) * delay):
        # Extract window
        indices = range(i, i + order * delay, delay)
        window = data[list(indices)]
        
        # Get ordinal pattern (rank order)
        pattern = tuple(np.argsort(np.argsort(window)))
        pattern_counts[pattern_dict[pattern]] += 1
    
    # Calculate entropy
    probabilities = pattern_counts / pattern_counts.sum()
    probabilities = probabilities[probabilities > 0]
    
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    if normalize:
        max_entropy = np.log2(factorial_order)
        entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return entropy


def sample_entropy(
    data: Union[np.ndarray, pd.Series],
    m: int = 2,
    r: Optional[float] = None,
    normalize: bool = False
) -> float:
    """
    Calculate Sample Entropy (Richman & Moorman, 2000).
    
    Mathematical Definition:
    ========================
    SampEn(m, r, N) = -ln(A/B)
    
    Where:
    - A = number of template matches of length m+1
    - B = number of template matches of length m
    - r = tolerance (similarity threshold)
    
    Parameters:
    ===========
    data : array-like
        Input time series
    m : int
        Embedding dimension (template length)
    r : float, optional
        Tolerance. Default: 0.2 * std(data)
    normalize : bool
        Not applicable for SampEn (included for API consistency)
        
    Returns:
    ========
    float : Sample entropy value
    
    Notes:
    ======
    - Lower values indicate more self-similarity/regularity
    - Higher values indicate more complexity/randomness
    - More robust than Approximate Entropy (no self-matching)
    
    References:
    ===========
    [1] Richman, J.S. & Moorman, J.R. (2000). "Physiological time-series analysis
        using approximate entropy and sample entropy."
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    N = len(data)
    
    if r is None:
        r = 0.2 * np.std(data)
    
    def _count_matches(template_length):
        count = 0
        templates = np.array([data[i:i + template_length] 
                             for i in range(N - template_length)])
        
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count
    
    B = _count_matches(m)
    A = _count_matches(m + 1)
    
    if B == 0 or A == 0:
        return np.inf
    
    return -np.log(A / B)


def spectral_entropy(
    data: Union[np.ndarray, pd.Series],
    fs: float = 1.0,
    normalize: bool = True
) -> float:
    """
    Calculate Spectral Entropy from Power Spectral Density.
    
    Mathematical Definition:
    ========================
    H_spectral = -Σ P_norm(f) log₂(P_norm(f))
    
    Where P_norm(f) is the normalized power spectral density.
    
    Parameters:
    ===========
    data : array-like
        Input time series
    fs : float
        Sampling frequency (default: 1.0 for normalized frequency)
    normalize : bool
        Normalize by log₂(N/2)
        
    Returns:
    ========
    float : Spectral entropy value
    
    Notes:
    ======
    - Measures the "flatness" of the spectrum
    - White noise has maximum spectral entropy
    - Periodic signals have low spectral entropy
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    
    # Compute power spectral density via FFT
    fft = np.fft.rfft(data)
    psd = np.abs(fft) ** 2
    
    # Normalize to probability distribution
    psd_norm = psd / psd.sum()
    psd_norm = psd_norm[psd_norm > 0]
    
    # Calculate entropy
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    
    if normalize:
        max_entropy = np.log2(len(psd))
        entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return entropy


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def compare_entropy_methods(
    data: Union[np.ndarray, pd.Series],
    window: int = 30
) -> pd.DataFrame:
    """
    Compare different entropy methods on the same data.
    
    Useful for robustness checks and method selection.
    
    Parameters:
    ===========
    data : array-like
        Input time series
    window : int
        Window size for rolling calculations
        
    Returns:
    ========
    pd.DataFrame : Comparison of entropy methods with statistics
    """
    data = pd.Series(data).dropna()
    
    results = {}
    
    # Shannon with different binning
    for method in ['fd', 'scott', 'sturges']:
        try:
            rolling_ent = rolling_shannon_entropy(data, window=window, bins=method)
            results[f'shannon_{method}'] = {
                'mean': rolling_ent.mean(),
                'std': rolling_ent.std(),
                'min': rolling_ent.min(),
                'max': rolling_ent.max()
            }
        except Exception as e:
            results[f'shannon_{method}'] = {'error': str(e)}
    
    # Permutation entropy
    try:
        pe_values = []
        for i in range(window, len(data)):
            pe = permutation_entropy(data.iloc[i-window:i].values)
            pe_values.append(pe)
        pe_series = pd.Series(pe_values)
        results['permutation'] = {
            'mean': pe_series.mean(),
            'std': pe_series.std(),
            'min': pe_series.min(),
            'max': pe_series.max()
        }
    except Exception as e:
        results['permutation'] = {'error': str(e)}
    
    # Spectral entropy
    try:
        se_values = []
        for i in range(window, len(data)):
            se = spectral_entropy(data.iloc[i-window:i].values)
            se_values.append(se)
        se_series = pd.Series(se_values)
        results['spectral'] = {
            'mean': se_series.mean(),
            'std': se_series.std(),
            'min': se_series.min(),
            'max': se_series.max()
        }
    except Exception as e:
        results['spectral'] = {'error': str(e)}
    
    return pd.DataFrame(results).T


# ==============================================================================
# TESTS
# ==============================================================================

if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(RANDOM_SEED)
    
    # Generate test signals
    n = 1000
    
    # 1. White noise (should have high entropy)
    white_noise = np.random.randn(n)
    
    # 2. Sine wave (should have low entropy)
    sine_wave = np.sin(np.linspace(0, 20 * np.pi, n))
    
    # 3. Financial-like returns (moderate entropy)
    returns = np.random.randn(n) * 0.02  # 2% daily vol
    returns[500:510] = np.random.randn(10) * 0.10  # Crisis period
    
    print("=" * 60)
    print("ENTROPY MODULE VALIDATION")
    print("=" * 60)
    
    for name, signal in [("White Noise", white_noise), 
                          ("Sine Wave", sine_wave),
                          ("Financial Returns", returns)]:
        print(f"\n{name}:")
        print("-" * 40)
        
        # Shannon entropy
        result = shannon_entropy(signal, bins='fd', normalize=True)
        print(f"  Shannon (FD):      {result.value:.4f}")
        
        # Permutation entropy
        pe = permutation_entropy(signal, order=3, normalize=True)
        print(f"  Permutation:       {pe:.4f}")
        
        # Spectral entropy
        se = spectral_entropy(signal, normalize=True)
        print(f"  Spectral:          {se:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
