"""
Volatility and Crisis Detection Measures
=========================================

This module implements multiple crisis definitions and volatility metrics for
robust validation of the Caria Risk Engine, as required for academic publication.

Crisis Definitions Implemented:
-------------------------------
1. Tail Events (EVT): Based on Extreme Value Theory
2. Structural Drawdowns: Persistent price declines
3. Jump Detection: Barndorff-Nielsen & Shephard test
4. Percentile-based: Original paper definition (return/vol > p90)

Key Finding from Research:
--------------------------
The "Volatility Compression" paradox: Crises are NOT preceded by high volatility,
but by LOW volatility combined with HIGH synchronization. This aligns with
Minsky's hypothesis: "Stability breeds instability."

References:
-----------
[1] McNeil, A.J. et al. (2015). "Quantitative Risk Management"
[2] Barndorff-Nielsen, O.E. & Shephard, N. (2006). "Econometrics of Testing for
    Jumps in Financial Economics Using Bipower Variation"
[3] Cont, R. (2001). "Empirical properties of asset returns: stylized facts and
    statistical issues"

Author: Tomás Basaure
Date: December 2025
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma
from typing import Union, Tuple, Optional, List, Literal
from dataclasses import dataclass
import warnings


# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CrisisMethod = Literal['tail_evt', 'drawdown', 'jump_bns', 'percentile', 'composite']


@dataclass
class CrisisLabel:
    """Container for crisis detection results."""
    is_crisis: bool
    method: str
    severity: float  # 0-1 scale
    details: dict
    
    def __repr__(self):
        return f"CrisisLabel(crisis={self.is_crisis}, method='{self.method}', severity={self.severity:.3f})"


@dataclass
class VolatilityMetrics:
    """Container for volatility calculation results."""
    realized_vol: float
    rolling_vol: pd.Series
    vol_of_vol: float
    compression_ratio: float  # Current vol / historical vol
    is_compressed: bool
    
    def __repr__(self):
        return f"VolatilityMetrics(vol={self.realized_vol:.4f}, compressed={self.is_compressed})"


# ==============================================================================
# REALIZED VOLATILITY
# ==============================================================================

def realized_volatility(
    returns: Union[np.ndarray, pd.Series],
    window: int = 30,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculate rolling realized volatility.
    
    Mathematical Definition:
    ========================
    σ_realized = √(Σᵢ rᵢ²)  (for high-frequency)
    
    Or for daily data:
    σ_realized = √(252 × Var(r))  (annualized)
    
    Parameters:
    ===========
    returns : array-like
        Return series (simple or log)
    window : int
        Rolling window for calculation
    annualize : bool
        If True, multiply by √252 for annualized volatility
    trading_days : int
        Number of trading days per year
        
    Returns:
    ========
    pd.Series : Rolling realized volatility
    """
    returns = pd.Series(returns).dropna()
    
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(trading_days)
    
    return vol


def bipower_variation(
    returns: Union[np.ndarray, pd.Series],
    window: int = 30
) -> pd.Series:
    """
    Calculate Bipower Variation (robust to jumps).
    
    Mathematical Definition:
    ========================
    BV = (π/2) × Σᵢ |rᵢ| × |rᵢ₋₁|
    
    This estimator is robust to jumps and converges to integrated variance.
    
    Parameters:
    ===========
    returns : array-like
        Return series
    window : int
        Rolling window
        
    Returns:
    ========
    pd.Series : Bipower variation estimates
    
    References:
    ===========
    [1] Barndorff-Nielsen, O.E. & Shephard, N. (2004)
    """
    returns = pd.Series(returns).dropna()
    
    abs_returns = returns.abs()
    abs_returns_lag = abs_returns.shift(1)
    
    # Bipower contribution
    bp_contrib = abs_returns * abs_returns_lag
    
    # Scaling constant (π/2)
    mu1 = np.sqrt(2 / np.pi)
    scaling = 1 / (mu1 ** 2)
    
    bv = scaling * bp_contrib.rolling(window=window).sum()
    
    return bv


def calculate_volatility_metrics(
    prices: Union[np.ndarray, pd.Series],
    window: int = 30,
    historical_window: int = 252
) -> VolatilityMetrics:
    """
    Calculate comprehensive volatility metrics.
    
    Parameters:
    ===========
    prices : array-like
        Price series
    window : int
        Short-term window for current vol
    historical_window : int
        Long-term window for historical comparison
        
    Returns:
    ========
    VolatilityMetrics : Container with all metrics
    """
    prices = pd.Series(prices).dropna()
    returns = prices.pct_change().dropna()
    
    # Rolling volatility
    rolling_vol = realized_volatility(returns, window=window, annualize=True)
    
    # Current realized volatility
    current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else np.nan
    
    # Historical volatility
    historical_vol = returns.iloc[-historical_window:].std() * np.sqrt(252) if len(returns) >= historical_window else returns.std() * np.sqrt(252)
    
    # Volatility of volatility
    vol_of_vol = rolling_vol.std() if len(rolling_vol) > 20 else np.nan
    
    # Compression ratio
    compression_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
    
    # Is volatility compressed? (below 50% of historical)
    is_compressed = compression_ratio < 0.5
    
    return VolatilityMetrics(
        realized_vol=current_vol,
        rolling_vol=rolling_vol,
        vol_of_vol=vol_of_vol,
        compression_ratio=compression_ratio,
        is_compressed=is_compressed
    )


# ==============================================================================
# CRISIS DEFINITION 1: TAIL EVENTS (EVT)
# ==============================================================================

def tail_crisis_evt(
    returns: Union[np.ndarray, pd.Series],
    threshold_percentile: float = 5,
    persistence_days: int = 3,
    horizon: int = 5
) -> pd.Series:
    """
    Detect crisis using Extreme Value Theory approach.
    
    Definition:
    ===========
    Crisis = Returns in bottom `threshold_percentile`% persisting for
             `persistence_days` consecutive days within `horizon` window.
    
    Mathematical Basis:
    ===================
    Uses the Generalized Pareto Distribution (GPD) for tail modeling:
    
    G_ξ,β(x) = 1 - (1 + ξx/β)^(-1/ξ)  for ξ ≠ 0
    
    Parameters:
    ===========
    returns : array-like
        Return series
    threshold_percentile : float
        Percentile for defining extreme returns (default: 5th = bottom 5%)
    persistence_days : int
        Minimum consecutive days of extreme returns
    horizon : int
        Forward-looking window for crisis label
        
    Returns:
    ========
    pd.Series : Binary crisis labels (1 = crisis, 0 = no crisis)
    
    Notes:
    ======
    - More conservative than simple percentile
    - Captures "persistent stress" vs one-off events
    """
    returns = pd.Series(returns).dropna()
    
    # Calculate threshold
    threshold = np.percentile(returns, threshold_percentile)
    
    # Mark extreme days
    extreme = (returns < threshold).astype(int)
    
    # Check for persistence
    crisis = pd.Series(0, index=returns.index)
    
    for i in range(len(returns) - horizon):
        # Look at forward window
        forward_extreme = extreme.iloc[i:i+horizon]
        
        # Check for consecutive extreme days
        consecutive = 0
        max_consecutive = 0
        for val in forward_extreme:
            if val == 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        if max_consecutive >= persistence_days:
            crisis.iloc[i] = 1
    
    return crisis


# ==============================================================================
# CRISIS DEFINITION 2: STRUCTURAL DRAWDOWN
# ==============================================================================

def drawdown_crisis(
    prices: Union[np.ndarray, pd.Series],
    threshold: float = 0.10,
    window: int = 20,
    horizon: int = 5
) -> pd.Series:
    """
    Detect crisis using structural drawdown approach.
    
    Definition:
    ===========
    Crisis = Maximum drawdown exceeds `threshold` within `window` days.
    
    Mathematical Definition:
    ========================
    Drawdown(t) = (Peak(t) - Price(t)) / Peak(t)
    
    Where Peak(t) = max(Price(s)) for s ≤ t
    
    Parameters:
    ===========
    prices : array-like
        Price series
    threshold : float
        Drawdown threshold (default: 10%)
    window : int
        Rolling window for drawdown calculation
    horizon : int
        Forward-looking window for crisis label
        
    Returns:
    ========
    pd.Series : Binary crisis labels
    
    Notes:
    ======
    - Captures sustained price declines
    - Less sensitive to single-day spikes
    """
    prices = pd.Series(prices).dropna()
    
    # Calculate rolling maximum (peak)
    rolling_peak = prices.rolling(window=window, min_periods=1).max()
    
    # Calculate drawdown
    drawdown = (rolling_peak - prices) / rolling_peak
    
    # Forward-looking crisis label
    crisis = pd.Series(0, index=prices.index)
    
    for i in range(len(prices) - horizon):
        # Check if drawdown exceeds threshold in forward window
        forward_dd = drawdown.iloc[i:i+horizon]
        if forward_dd.max() > threshold:
            crisis.iloc[i] = 1
    
    return crisis


# ==============================================================================
# CRISIS DEFINITION 3: JUMP DETECTION (BARNDORFF-NIELSEN & SHEPHARD)
# ==============================================================================

def jump_crisis_bns(
    returns: Union[np.ndarray, pd.Series],
    window: int = 30,
    significance: float = 0.01,
    horizon: int = 5
) -> pd.Series:
    """
    Detect crisis using jump detection test (BNS statistic).
    
    Mathematical Definition:
    ========================
    The test compares Realized Variance (RV) with Bipower Variation (BV):
    
    Z_BNS = (RV - BV) / √(θ × max(QP - BV², 0))
    
    Where:
    - RV = Σ rᵢ² (realized variance)
    - BV = (π/2) Σ |rᵢ||rᵢ₋₁| (bipower variation)
    - QP = quad-power quarticity (for variance estimation)
    
    Under H₀ (no jumps): Z_BNS ~ N(0,1)
    
    Parameters:
    ===========
    returns : array-like
        Return series
    window : int
        Rolling window for calculations
    significance : float
        Significance level for jump detection
    horizon : int
        Forward-looking window
        
    Returns:
    ========
    pd.Series : Binary crisis labels (jump detected = 1)
    
    References:
    ===========
    [1] Barndorff-Nielsen, O.E. & Shephard, N. (2006)
    """
    returns = pd.Series(returns).dropna()
    
    # Realized variance
    rv = (returns ** 2).rolling(window=window).sum()
    
    # Bipower variation
    bv = bipower_variation(returns, window=window)
    
    # Tri-power quarticity for variance estimation
    mu1 = np.sqrt(2 / np.pi)
    abs_returns = returns.abs()
    
    # Simplified variance estimate
    theta = (np.pi ** 2 / 4 + np.pi - 5) * (mu1 ** -4)
    
    # BNS statistic
    numerator = rv - bv
    
    # Quad-power quarticity approximation
    qp = (abs_returns ** (4/3)).rolling(window=window).sum() ** 3
    
    var_estimate = theta * np.maximum(qp - bv ** 2, 1e-10)
    
    z_bns = numerator / np.sqrt(var_estimate)
    
    # Critical value
    z_critical = stats.norm.ppf(1 - significance)
    
    # Jump detected if Z > critical value
    jump_detected = (z_bns > z_critical).astype(int)
    
    # Forward-looking label
    crisis = pd.Series(0, index=returns.index)
    
    for i in range(len(returns) - horizon):
        if jump_detected.iloc[i:i+horizon].sum() > 0:
            crisis.iloc[i] = 1
    
    return crisis


# ==============================================================================
# CRISIS DEFINITION 4: PERCENTILE-BASED (ORIGINAL PAPER)
# ==============================================================================

def percentile_crisis(
    returns: Union[np.ndarray, pd.Series],
    volatility: Union[np.ndarray, pd.Series],
    return_percentile: float = 10,
    vol_percentile: float = 90,
    horizon: int = 5,
    method: str = 'or'
) -> pd.Series:
    """
    Detect crisis using percentile-based approach (original paper definition).
    
    Definition:
    ===========
    Crisis at t+horizon if:
    - Return < p(return_percentile) OR
    - Volatility > p(vol_percentile)
    
    Parameters:
    ===========
    returns : array-like
        Return series
    volatility : array-like
        Volatility series
    return_percentile : float
        Percentile for negative returns (default: bottom 10%)
    vol_percentile : float
        Percentile for high volatility (default: top 90%)
    horizon : int
        Forward-looking window
    method : str
        'or' = either condition, 'and' = both conditions
        
    Returns:
    ========
    pd.Series : Binary crisis labels
    """
    returns = pd.Series(returns).dropna()
    volatility = pd.Series(volatility).dropna()
    
    # Align series
    common_idx = returns.index.intersection(volatility.index)
    returns = returns.loc[common_idx]
    volatility = volatility.loc[common_idx]
    
    # Calculate thresholds
    return_threshold = np.percentile(returns, return_percentile)
    vol_threshold = np.percentile(volatility, vol_percentile)
    
    # Forward-looking crisis label
    crisis = pd.Series(0, index=returns.index)
    
    for i in range(len(returns) - horizon):
        forward_returns = returns.iloc[i:i+horizon]
        forward_vol = volatility.iloc[i:i+horizon]
        
        # Check conditions
        return_condition = (forward_returns < return_threshold).any()
        vol_condition = (forward_vol > vol_threshold).any()
        
        if method == 'or':
            if return_condition or vol_condition:
                crisis.iloc[i] = 1
        else:  # 'and'
            if return_condition and vol_condition:
                crisis.iloc[i] = 1
    
    return crisis


# ==============================================================================
# COMPOSITE CRISIS DETECTOR
# ==============================================================================

def detect_crisis(
    prices: Union[np.ndarray, pd.Series],
    method: CrisisMethod = 'composite',
    horizon: int = 5,
    **kwargs
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Unified crisis detection function supporting multiple methods.
    
    Parameters:
    ===========
    prices : array-like
        Price series
    method : str
        Detection method:
        - 'tail_evt': Extreme Value Theory
        - 'drawdown': Structural drawdown
        - 'jump_bns': Jump detection
        - 'percentile': Original paper method
        - 'composite': Ensemble of all methods
    horizon : int
        Forward-looking window for labels
    **kwargs : dict
        Method-specific parameters
        
    Returns:
    ========
    Tuple[pd.Series, pd.DataFrame]:
        - Crisis labels (binary)
        - Individual method results (for composite)
    """
    prices = pd.Series(prices).dropna()
    returns = prices.pct_change().dropna()
    vol = realized_volatility(returns, window=30, annualize=False)
    
    results = {}
    
    if method in ['tail_evt', 'composite']:
        results['tail_evt'] = tail_crisis_evt(
            returns, 
            threshold_percentile=kwargs.get('tail_percentile', 5),
            persistence_days=kwargs.get('persistence_days', 3),
            horizon=horizon
        )
    
    if method in ['drawdown', 'composite']:
        results['drawdown'] = drawdown_crisis(
            prices,
            threshold=kwargs.get('dd_threshold', 0.10),
            window=kwargs.get('dd_window', 20),
            horizon=horizon
        )
    
    if method in ['jump_bns', 'composite']:
        results['jump_bns'] = jump_crisis_bns(
            returns,
            window=kwargs.get('bns_window', 30),
            significance=kwargs.get('bns_significance', 0.01),
            horizon=horizon
        )
    
    if method in ['percentile', 'composite']:
        results['percentile'] = percentile_crisis(
            returns, vol,
            return_percentile=kwargs.get('return_pct', 10),
            vol_percentile=kwargs.get('vol_pct', 90),
            horizon=horizon
        )
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Final crisis label
    if method == 'composite':
        # Majority vote: crisis if >= 2 methods agree
        crisis = (results_df.sum(axis=1) >= 2).astype(int)
    else:
        crisis = results[method]
    
    return crisis, results_df


# ==============================================================================
# VOLATILITY COMPRESSION DETECTION
# ==============================================================================

def detect_volatility_compression(
    prices: Union[np.ndarray, pd.Series],
    short_window: int = 20,
    long_window: int = 60,
    compression_threshold: float = 0.5
) -> pd.Series:
    """
    Detect volatility compression (the "Silence Paradox").
    
    Key Research Finding:
    =====================
    Crises are preceded NOT by high volatility, but by LOW volatility
    (compression) combined with high synchronization. This is the
    "Calm Before the Storm" effect.
    
    Parameters:
    ===========
    prices : array-like
        Price series
    short_window : int
        Window for current volatility
    long_window : int
        Window for historical volatility
    compression_threshold : float
        Ratio below which vol is "compressed"
        
    Returns:
    ========
    pd.Series : Boolean series (True = compressed)
    """
    prices = pd.Series(prices).dropna()
    returns = prices.pct_change().dropna()
    
    # Short-term volatility
    short_vol = returns.rolling(window=short_window).std()
    
    # Long-term volatility
    long_vol = returns.rolling(window=long_window).std()
    
    # Compression ratio
    compression = short_vol / long_vol
    
    # Is compressed?
    is_compressed = compression < compression_threshold
    
    return is_compressed


# ==============================================================================
# CARIA RISK ENGINE (FINAL MODEL)
# ==============================================================================

def caria_risk_signal(
    entropy: float,
    desync: float,
    volatility: float,
    entropy_threshold: float = 0.6,
    sync_critical: float = 0.13,
    sync_alert: float = 0.31,
    vol_compressed: float = 0.08,
    vol_fragile: float = 0.06
) -> Tuple[str, dict]:
    """
    The Caria Risk Formula - Decision Tree derived from empirical optimization.
    
    Three Risk Regimes:
    ===================
    
    Regime A: HYPERSYNCHRONIZATION (CRITICAL)
        Condition: Desync ≤ 0.13 (i.e., |r| > 0.87)
        Interpretation: Total market lockstep. Immediate danger.
        
    Regime B: VOLATILITY COMPRESSION (ALERT)
        Condition: 0.13 < Desync ≤ 0.31 AND Volatility ≤ 0.08
        Interpretation: "Calm Before the Storm"
        
    Regime C: COMPLACENCY (FRAGILITY)
        Condition: Desync > 0.31 AND Volatility ≤ 0.06
        Interpretation: Extreme calm indicates liquidity fragility
    
    Parameters:
    ===========
    entropy : float
        Shannon entropy (normalized 0-1)
    desync : float
        Desynchronization = 1 - |r|
    volatility : float
        Realized volatility
    *_threshold : float
        Decision thresholds from tree
        
    Returns:
    ========
    Tuple[str, dict]:
        - Risk regime: 'CRITICAL', 'ALERT', 'FRAGILE', or 'NORMAL'
        - Details dictionary with reasoning
    """
    details = {
        'entropy': entropy,
        'desync': desync,
        'volatility': volatility
    }
    
    # Regime A: Hypersynchronization
    if desync <= sync_critical:
        return 'CRITICAL', {
            **details,
            'regime': 'A_HYPERSYNC',
            'reason': f'Desync={desync:.3f} ≤ {sync_critical} (total lockstep)'
        }
    
    # Regime B: Volatility Compression
    if desync <= sync_alert and volatility <= vol_compressed:
        return 'ALERT', {
            **details,
            'regime': 'B_COMPRESSION',
            'reason': f'Desync={desync:.3f} ≤ {sync_alert} AND Vol={volatility:.3f} ≤ {vol_compressed}'
        }
    
    # Regime C: Complacency
    if desync > sync_alert and volatility <= vol_fragile:
        return 'FRAGILE', {
            **details,
            'regime': 'C_COMPLACENCY',
            'reason': f'Desync={desync:.3f} > {sync_alert} BUT Vol={volatility:.3f} ≤ {vol_fragile} (liquidity risk)'
        }
    
    # Normal
    return 'NORMAL', {
        **details,
        'regime': 'NORMAL',
        'reason': 'No risk conditions met'
    }


# ==============================================================================
# TESTS
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VOLATILITY & CRISIS MODULE VALIDATION")
    print("=" * 60)
    
    np.random.seed(RANDOM_SEED)
    n = 1000
    
    # Generate synthetic price series with a crisis
    returns = np.random.randn(n) * 0.01  # 1% daily vol
    
    # Inject crisis at day 500
    returns[500:520] = np.random.randn(20) * 0.05 - 0.02  # 5% vol + negative drift
    
    prices = pd.Series(100 * np.cumprod(1 + returns))
    returns = pd.Series(returns)
    
    print("\n1. Volatility Metrics Test:")
    print("-" * 40)
    metrics = calculate_volatility_metrics(prices)
    print(f"   Realized Vol: {metrics.realized_vol:.2%}")
    print(f"   Compression: {metrics.compression_ratio:.2f}")
    print(f"   Is Compressed: {metrics.is_compressed}")
    
    print("\n2. Crisis Detection Methods:")
    print("-" * 40)
    
    for method in ['tail_evt', 'drawdown', 'jump_bns', 'percentile']:
        crisis, _ = detect_crisis(prices, method=method, horizon=5)
        n_crisis = crisis.sum()
        pct_crisis = n_crisis / len(crisis) * 100
        print(f"   {method:15s}: {n_crisis:4d} crisis days ({pct_crisis:.1f}%)")
    
    # Composite
    crisis, methods_df = detect_crisis(prices, method='composite', horizon=5)
    print(f"   {'composite':15s}: {crisis.sum():4d} crisis days ({crisis.sum()/len(crisis)*100:.1f}%)")
    
    print("\n3. Volatility Compression Detection:")
    print("-" * 40)
    compressed = detect_volatility_compression(prices)
    print(f"   Compressed days: {compressed.sum()} ({compressed.sum()/len(compressed)*100:.1f}%)")
    
    print("\n4. Caria Risk Signal Test:")
    print("-" * 40)
    test_cases = [
        (0.7, 0.10, 0.05),  # High sync, low vol -> CRITICAL
        (0.7, 0.25, 0.06),  # Moderate sync, compressed vol -> ALERT
        (0.7, 0.40, 0.04),  # Low sync, very low vol -> FRAGILE
        (0.7, 0.40, 0.15),  # Low sync, high vol -> NORMAL
    ]
    
    for ent, desync, vol in test_cases:
        signal, details = caria_risk_signal(ent, desync, vol)
        print(f"   H={ent:.2f}, D={desync:.2f}, σ={vol:.2f} -> {signal}")
    
    print("\n" + "=" * 60)
    print("All volatility tests completed!")
