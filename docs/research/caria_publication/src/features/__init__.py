This module provides the core feature calculations for the paper:
"Entropic Resonance and Volatility Compression as Precursors to Systemic Failure"

Modules:
--------
- entropy: Shannon entropy and alternative entropy measures
- synchronization: Kuramoto order parameter and temporal sync metrics
- volatility: Realized volatility and crisis detection methods
- spectral: cross-sectional spectral structure measures (AR, eigen-entropy, eRank)
- caria_sr_spectral: CARIA-SR built from spectral structure + hysteresis

Usage:
------
>>> from caria_publication.src.features import (
...     shannon_entropy, rolling_shannon_entropy,
...     calculate_temporal_sync, rolling_synchronization,
...     detect_crisis, caria_risk_signal
... )
"""

from .entropy import (
    shannon_entropy,
    rolling_shannon_entropy,
    permutation_entropy,
    sample_entropy,
    spectral_entropy,
    compare_entropy_methods,
    EntropyResult
)

from .synchronization import (
    kuramoto_order_parameter,
    calculate_temporal_sync,
    rolling_synchronization,
    rolling_correlation_sync,
    calculate_bifurcation_risk,
    extract_instantaneous_phase,
    decompose_signal_bands,
    SyncResult,
    DEFAULT_BANDS,
    PHYSICS_WEIGHTS
)

from .volatility import (
    realized_volatility,
    bipower_variation,
    calculate_volatility_metrics,
    tail_crisis_evt,
    drawdown_crisis,
    jump_crisis_bns,
    percentile_crisis,
    detect_crisis,
    detect_volatility_compression,
    caria_risk_signal,
    CrisisLabel,
    VolatilityMetrics
)

from .spectral import (
    SpectralStats,
    pairwise_correlation,
    eigenvalues_of_correlation,
    absorption_ratio,
    spectral_shannon_entropy,
    effective_rank,
    spectral_stats_from_returns,
)

from .caria_sr_spectral import (
    CariaSRSpectralConfig,
    compute_caria_sr_spectral,
)

__all__ = [
    # Entropy
    'shannon_entropy',
    'rolling_shannon_entropy',
    'permutation_entropy',
    'sample_entropy',
    'spectral_entropy',
    'compare_entropy_methods',
    'EntropyResult',
    
    # Synchronization
    'kuramoto_order_parameter',
    'calculate_temporal_sync',
    'rolling_synchronization',
    'rolling_correlation_sync',
    'calculate_bifurcation_risk',
    'extract_instantaneous_phase',
    'decompose_signal_bands',
    'SyncResult',
    'DEFAULT_BANDS',
    'PHYSICS_WEIGHTS',
    
    # Volatility & Crisis
    'realized_volatility',
    'bipower_variation',
    'calculate_volatility_metrics',
    'tail_crisis_evt',
    'drawdown_crisis',
    'jump_crisis_bns',
    'percentile_crisis',
    'detect_crisis',
    'detect_volatility_compression',
    'caria_risk_signal',
    'CrisisLabel',
    'VolatilityMetrics',
    
    # Spectral (cross-sectional)
    'SpectralStats',
    'pairwise_correlation',
    'eigenvalues_of_correlation',
    'absorption_ratio',
    'spectral_shannon_entropy',
    'effective_rank',
    'spectral_stats_from_returns',

    # CARIA-SR spectral
    'CariaSRSpectralConfig',
    'compute_caria_sr_spectral',
]
