"""
Synchronization Measures for Multi-Scale Financial Systems
===========================================================

This module implements temporal synchronization metrics based on the Kuramoto model
of coupled oscillators, as described in:

    "Entropic Resonance and Volatility Compression as Precursors to Systemic Failure"
    Basaure, T. (2025)

Theoretical Framework:
----------------------
Financial markets exhibit multi-scale dynamics where different agents operate at
different temporal horizons:
    - HFT/Algorithms: < 1 day (Ultra-Fast)
    - Day Traders: 1-10 days (Short)
    - Hedge Funds: 10-60 days (Medium) 
    - Institutions: 60-250 days (Long)
    - Central Banks: > 250 days (Ultra-Long)

The Kuramoto Order Parameter (r) measures phase synchronization across these scales:

    r(t) = |1/N Σₖ exp(i·φₖ(t))|

Where:
    - φₖ(t) is the instantaneous phase of the k-th frequency band
    - r = 0: Complete desynchronization (healthy independence)
    - r = 1: Perfect synchronization (dangerous herding)

Phase extraction uses the Hilbert Transform:
    φ(t) = arctan(H[x(t)] / x(t))

Where H[x] is the Hilbert transform of signal x.

Key Insight:
------------
Crisis emerges not from chaos (high entropy, low sync) but from "Entropic Resonance":
a state where the system maintains high information content (entropy) while all
temporal scales lock into synchronous behavior.

References:
-----------
[1] Kuramoto, Y. (1984). "Chemical Oscillations, Waves, and Turbulence"
[2] Strogatz, S.H. (2000). "From Kuramoto to Crawford: exploring the onset of
    synchronization in populations of coupled oscillators"
[3] Morales, R. et al. (2012). "Dynamical generalized Hurst exponent as a tool
    to monitor unstable periods in financial time series"

Author: Tomás Basaure
Date: December 2025
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt
from typing import Union, Tuple, Optional, List, Dict
from dataclasses import dataclass
import warnings


# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Default frequency bands (in trading days)
DEFAULT_BANDS = {
    'ultra_fast': (1, 5),      # HFT/Algorithms: 1-5 days
    'short': (5, 20),          # Day/Swing traders: 1-4 weeks
    'medium': (20, 60),        # Hedge funds: 1-3 months (RESONANCE ZONE)
    'long': (60, 252),         # Institutions: 3-12 months
    'ultra_long': (252, 504)   # Central banks: 1-2 years
}

# Physics-first weights (medium band is the "fuse")
PHYSICS_WEIGHTS = {
    'ultra_fast': 0.05,
    'short': 0.10,
    'medium': 0.35,    # Critical resonance zone
    'long': 0.25,
    'ultra_long': 0.25
}


@dataclass
class SyncResult:
    """Container for synchronization calculation results."""
    order_parameter: float  # Kuramoto r ∈ [0, 1]
    desynchronization: float  # D = 1 - r
    phases: np.ndarray  # Individual band phases
    band_amplitudes: Dict[str, float]  # Energy per band
    weighted_sync: float  # Physics-weighted synchronization
    
    def __repr__(self):
        return f"SyncResult(r={self.order_parameter:.4f}, D={self.desynchronization:.4f})"


# ==============================================================================
# PHASE EXTRACTION VIA HILBERT TRANSFORM
# ==============================================================================

def extract_instantaneous_phase(
    signal_data: np.ndarray,
    unwrap: bool = True
) -> np.ndarray:
    """
    Extract instantaneous phase using the Hilbert Transform.
    
    Mathematical Definition:
    ========================
    For a real signal x(t), the analytic signal is:
        z(t) = x(t) + i·H[x(t)]
    
    Where H[x] is the Hilbert transform. The instantaneous phase is:
        φ(t) = arctan2(H[x(t)], x(t)) = arg(z(t))
    
    Parameters:
    ===========
    signal_data : np.ndarray
        Input time series (should be centered/detrended)
    unwrap : bool
        If True, unwrap phase to avoid discontinuities at ±π
        
    Returns:
    ========
    np.ndarray : Instantaneous phase φ(t) in radians
    
    Notes:
    ======
    - Input should be detrended to avoid edge effects
    - Phase is meaningful only for narrow-band signals
    - For broad-band signals, decompose first then extract phase per band
    """
    # Compute analytic signal
    analytic = hilbert(signal_data)
    
    # Extract instantaneous phase
    phase = np.angle(analytic)
    
    if unwrap:
        phase = np.unwrap(phase)
    
    return phase


def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float = 1.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.
    
    Parameters:
    ===========
    data : np.ndarray
        Input signal
    lowcut : float
        Lower cutoff frequency (cycles per sample)
    highcut : float
        Upper cutoff frequency
    fs : float
        Sampling frequency
    order : int
        Filter order
        
    Returns:
    ========
    np.ndarray : Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Clamp to valid range
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    try:
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    except ValueError:
        # If filter design fails, return smoothed signal
        warnings.warn(f"Bandpass filter failed for [{lowcut}, {highcut}], using original signal")
        return data


def decompose_signal_bands(
    data: np.ndarray,
    bands: Optional[Dict[str, Tuple[int, int]]] = None,
    method: str = 'bandpass'
) -> Dict[str, np.ndarray]:
    """
    Decompose signal into frequency bands.
    
    Parameters:
    ===========
    data : np.ndarray
        Input time series (e.g., log returns or prices)
    bands : dict, optional
        Dictionary mapping band names to (low_period, high_period) in samples
    method : str
        Decomposition method: 'bandpass' or 'moving_average'
        
    Returns:
    ========
    Dict[str, np.ndarray] : Dictionary of band-filtered signals
    
    Notes:
    ======
    - 'bandpass' uses Butterworth filter (better for phase analysis)
    - 'moving_average' uses cascading MAs (simpler, more interpretable)
    """
    if bands is None:
        bands = DEFAULT_BANDS
    
    decomposed = {}
    
    if method == 'bandpass':
        for name, (low_period, high_period) in bands.items():
            # Convert periods to frequencies
            low_freq = 1.0 / high_period  # Note: inverted
            high_freq = 1.0 / low_period
            
            if len(data) > max(low_period, high_period) * 2:
                decomposed[name] = bandpass_filter(data, low_freq, high_freq)
            else:
                decomposed[name] = np.zeros_like(data)
    
    elif method == 'moving_average':
        # Cascading moving average decomposition
        df = pd.Series(data)
        prev_ma = df.copy()
        
        sorted_bands = sorted(bands.items(), key=lambda x: x[1][0])
        
        for name, (low_period, high_period) in sorted_bands:
            ma_slow = df.rolling(window=high_period, min_periods=1).mean()
            ma_fast = df.rolling(window=low_period, min_periods=1).mean()
            decomposed[name] = (ma_fast - ma_slow).values
    
    return decomposed


# ==============================================================================
# KURAMOTO ORDER PARAMETER
# ==============================================================================

def kuramoto_order_parameter(
    phases: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Calculate Kuramoto Order Parameter from phase array.
    
    Mathematical Definition:
    ========================
    r(t) = |1/N Σₖ wₖ exp(i·φₖ(t))|
    
    This measures the degree of phase coherence:
    - r = 0: Phases uniformly distributed (no synchronization)
    - r = 1: All phases identical (perfect synchronization)
    
    Parameters:
    ===========
    phases : np.ndarray
        Array of phases in radians, shape (n_oscillators,) or (n_time, n_oscillators)
    weights : np.ndarray, optional
        Weights for each oscillator. Default: uniform weights.
        
    Returns:
    ========
    Tuple[float, float] : (order_parameter r, mean_phase ψ)
    
    Example:
    ========
    >>> phases = np.array([0.1, 0.15, 0.12, 0.08])  # Similar phases
    >>> r, psi = kuramoto_order_parameter(phases)
    >>> print(f"Synchronization: {r:.3f}")  # Should be close to 1
    
    References:
    ===========
    [1] Kuramoto, Y. (1984). "Chemical Oscillations, Waves, and Turbulence"
    """
    phases = np.asarray(phases)
    
    if weights is None:
        weights = np.ones(phases.shape[-1]) / phases.shape[-1]
    else:
        weights = np.asarray(weights)
        weights = weights / weights.sum()
    
    # Calculate complex order parameter
    # z = Σ wₖ exp(i·φₖ)
    if phases.ndim == 1:
        z = np.sum(weights * np.exp(1j * phases))
    else:
        z = np.sum(weights * np.exp(1j * phases), axis=-1)
    
    # Order parameter is the magnitude
    r = np.abs(z)
    
    # Mean phase is the argument
    psi = np.angle(z)
    
    return r, psi


def calculate_temporal_sync(
    data: Union[np.ndarray, pd.Series],
    bands: Optional[Dict[str, Tuple[int, int]]] = None,
    weights: Optional[Dict[str, float]] = None,
    method: str = 'bandpass'
) -> SyncResult:
    """
    Calculate multi-scale temporal synchronization.
    
    This is the core synchronization metric for the Caria Risk Engine.
    
    Parameters:
    ===========
    data : array-like
        Input time series (prices or returns)
    bands : dict, optional
        Frequency bands definition
    weights : dict, optional
        Physics-inspired weights per band
    method : str
        Decomposition method
        
    Returns:
    ========
    SyncResult : Container with synchronization metrics
    
    Algorithm:
    ==========
    1. Decompose signal into frequency bands
    2. Extract instantaneous phase for each band using Hilbert transform
    3. Calculate Kuramoto order parameter across bands
    4. Apply physics-based weighting
    """
    if bands is None:
        bands = DEFAULT_BANDS
    if weights is None:
        weights = PHYSICS_WEIGHTS
    
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    
    # Decompose into bands
    band_signals = decompose_signal_bands(data, bands, method)
    
    # Extract phases and amplitudes
    phases = []
    amplitudes = {}
    weight_array = []
    
    for name in bands.keys():
        if name in band_signals:
            band_data = band_signals[name]
            
            # Extract phase (take last value for point-in-time)
            phase = extract_instantaneous_phase(band_data)[-1]
            phases.append(phase)
            
            # Calculate band energy (amplitude)
            amplitudes[name] = np.std(band_data)
            
            # Get weight
            weight_array.append(weights.get(name, 1.0 / len(bands)))
    
    phases = np.array(phases)
    weight_array = np.array(weight_array)
    weight_array = weight_array / weight_array.sum()
    
    # Calculate Kuramoto order parameter
    r, _ = kuramoto_order_parameter(phases, weight_array)
    
    # Desynchronization measure
    D = 1.0 - r
    
    # Physics-weighted synchronization
    weighted_sync = r
    
    return SyncResult(
        order_parameter=r,
        desynchronization=D,
        phases=phases,
        band_amplitudes=amplitudes,
        weighted_sync=weighted_sync
    )


def rolling_synchronization(
    data: Union[np.ndarray, pd.Series],
    window: int = 30,
    bands: Optional[Dict[str, Tuple[int, int]]] = None,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate rolling synchronization metrics.
    
    Parameters:
    ===========
    data : array-like
        Input time series
    window : int
        Rolling window size
    bands : dict, optional
        Frequency bands definition
    min_periods : int, optional
        Minimum observations required
        
    Returns:
    ========
    pd.DataFrame : Rolling sync metrics (r, D, weighted_sync)
    """
    if min_periods is None:
        min_periods = window
    
    data = pd.Series(data).dropna()
    
    results = {
        'order_parameter': [],
        'desynchronization': [],
        'weighted_sync': []
    }
    indices = []
    
    for i in range(min_periods - 1, len(data)):
        window_data = data.iloc[max(0, i - window + 1):i + 1]
        
        if len(window_data) >= min_periods:
            try:
                sync = calculate_temporal_sync(window_data.values, bands)
                results['order_parameter'].append(sync.order_parameter)
                results['desynchronization'].append(sync.desynchronization)
                results['weighted_sync'].append(sync.weighted_sync)
                indices.append(data.index[i])
            except Exception:
                results['order_parameter'].append(np.nan)
                results['desynchronization'].append(np.nan)
                results['weighted_sync'].append(np.nan)
                indices.append(data.index[i])
    
    return pd.DataFrame(results, index=indices)


# ==============================================================================
# CORRELATION-BASED SYNCHRONIZATION (SIMPLER ALTERNATIVE)
# ==============================================================================

def rolling_correlation_sync(
    data: Union[np.ndarray, pd.Series],
    window: int = 30,
    reference_window: int = 5
) -> pd.Series:
    """
    Simple synchronization measure using rolling correlation.
    
    This is a computationally simpler alternative to Kuramoto
    that measures how "locked-in" recent behavior is with longer trends.
    
    Parameters:
    ===========
    data : array-like
        Input time series
    window : int
        Long-term window for comparison
    reference_window : int
        Short-term window (recent behavior)
        
    Returns:
    ========
    pd.Series : Correlation-based synchronization |r| ∈ [0, 1]
    
    Notes:
    ======
    - D = 1 - |r| is the desynchronization measure
    - When |r| → 1: Short and long-term trends aligned (danger)
    - When |r| → 0: No alignment (healthy diversity)
    """
    data = pd.Series(data).dropna()
    
    # Calculate returns for correlation
    returns = data.pct_change().dropna()
    
    # Short-term momentum
    short_ma = returns.rolling(window=reference_window).mean()
    
    # Long-term momentum
    long_ma = returns.rolling(window=window).mean()
    
    # Rolling correlation between short and long term
    sync = short_ma.rolling(window=window).corr(long_ma).abs()
    
    return sync


# ==============================================================================
# BIFURCATION RISK
# ==============================================================================

def calculate_bifurcation_risk(
    sync: float,
    volatility: float,
    sync_threshold: float = 0.7,
    vol_threshold: float = 0.02
) -> float:
    """
    Calculate bifurcation risk using geometric mean.
    
    Mathematical Definition:
    ========================
    BifRisk = √(Sync_norm × Vol_norm)
    
    Geometric mean ensures ALL conditions must be met:
    - High synchronization alone is not enough
    - High volatility alone is not enough
    - Both must be elevated for true bifurcation risk
    
    Parameters:
    ===========
    sync : float
        Synchronization measure (Kuramoto r or correlation)
    volatility : float
        Realized volatility measure
    sync_threshold : float
        Synchronization threshold for normalization
    vol_threshold : float
        Volatility threshold for normalization
        
    Returns:
    ========
    float : Bifurcation risk ∈ [0, 1]
    """
    # Normalize to [0, 1]
    sync_norm = min(sync / sync_threshold, 1.0)
    vol_norm = min(volatility / vol_threshold, 1.0)
    
    # Geometric mean
    bif_risk = np.sqrt(sync_norm * vol_norm)
    
    return bif_risk


# ==============================================================================
# TESTS
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SYNCHRONIZATION MODULE VALIDATION")
    print("=" * 60)
    
    np.random.seed(RANDOM_SEED)
    n = 500
    
    # Test 1: Synchronized signal (all bands in phase)
    t = np.linspace(0, 10 * np.pi, n)
    synchronized = np.sin(t) + 0.5 * np.sin(2*t) + 0.3 * np.sin(0.5*t)
    
    # Test 2: Desynchronized signal (random phases)
    desynchronized = np.sin(t) + 0.5 * np.sin(2*t + np.pi/3) + 0.3 * np.sin(0.5*t + np.pi/2)
    
    # Test 3: Random noise (maximum desync)
    noise = np.random.randn(n)
    
    print("\n1. Phase Extraction Test:")
    print("-" * 40)
    phase = extract_instantaneous_phase(np.sin(t))
    print(f"   Sine wave phase range: [{phase.min():.2f}, {phase.max():.2f}] rad")
    
    print("\n2. Kuramoto Order Parameter Test:")
    print("-" * 40)
    
    for name, sig in [("Synchronized", synchronized), 
                       ("Desynchronized", desynchronized),
                       ("Random Noise", noise)]:
        try:
            sync = calculate_temporal_sync(sig)
            print(f"   {name}:")
            print(f"      r = {sync.order_parameter:.4f}")
            print(f"      D = {sync.desynchronization:.4f}")
        except Exception as e:
            print(f"   {name}: Error - {e}")
    
    print("\n3. Correlation-based Sync Test:")
    print("-" * 40)
    for name, sig in [("Synchronized", synchronized), 
                       ("Random Noise", noise)]:
        corr_sync = rolling_correlation_sync(sig, window=30).dropna()
        print(f"   {name}: mean |r| = {corr_sync.mean():.4f}")
    
    print("\n4. Bifurcation Risk Test:")
    print("-" * 40)
    scenarios = [
        ("Low Sync, Low Vol", 0.3, 0.01),
        ("High Sync, Low Vol", 0.9, 0.01),
        ("Low Sync, High Vol", 0.3, 0.05),
        ("High Sync, High Vol", 0.9, 0.05),
    ]
    for name, sync, vol in scenarios:
        risk = calculate_bifurcation_risk(sync, vol)
        print(f"   {name}: BifRisk = {risk:.4f}")
    
    print("\n" + "=" * 60)
    print("All synchronization tests completed!")
