"""
Spectral (Cross-Sectional) Structure Measures
=============================================

This module implements *cross-sectional* (multi-asset) spectral diagnostics used
in the CARIA-SR framing discussed in the research notes:

- Absorption Ratio (AR): fraction of total variance absorbed by the top-K
  principal components (eigenvalues) of the *correlation* matrix.
- Spectral Shannon Entropy of eigenvalues: measures dispersion of systematic
  risk across factors.
- Effective Rank (eRank): exp(entropy), a scale-free "effective dimension".

Critical reproducibility notes (addressing common empirical pitfalls):
----------------------------------------------------------------------
1) **No forward-fill of prices** for correlation/covariance estimation.
   Forward-filling creates artificial 0 returns and biases correlations.
2) **Entropy must be comparable across time** when N (number of assets in the
   window) varies. We therefore expose normalized entropy:
       H_norm = H / log(N)
   and effective-rank-based normalization.
3) Pairwise correlations with missing data can yield non-PSD matrices. We
   provide an optional PSD projection via eigenvalue clipping.

The functions are intentionally small and composable for auditability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd

EntropyNorm = Literal["logN", "none"]


@dataclass(frozen=True)
class SpectralStats:
    """Container for spectral statistics at a point in time."""

    n_assets: int
    eigenvalues: np.ndarray  # sorted desc
    absorption_ratio: float
    entropy: float
    entropy_norm: float
    effective_rank: float


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def nearest_psd_eigen_clip(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest PSD matrix (simple eigen clipping).

    This is *not* a full Higham nearest correlation implementation, but it is:
    - deterministic,
    - fast,
    - sufficient to prevent negative eigenvalues from numerical/pairwise artifacts.
    """
    a = np.asarray(matrix, dtype=float)
    a = 0.5 * (a + a.T)
    w, v = np.linalg.eigh(a)
    w = np.clip(w, eps, None)
    out = (v * w) @ v.T
    out = 0.5 * (out + out.T)
    return out


def pairwise_correlation(
    returns: pd.DataFrame,
    min_pair_obs: int = 30,
    ensure_psd: bool = True,
    psd_eps: float = 1e-10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute a pairwise-complete correlation matrix with missing data.

    Parameters
    ----------
    returns:
        DataFrame of returns with DatetimeIndex and asset columns.
        Missing values are allowed (NO forward-fill should be applied upstream).
    min_pair_obs:
        Minimum overlapping observations required to compute corr(i,j).
        Pairs with fewer observations will be set to NaN.
    ensure_psd:
        If True, project the resulting correlation matrix to PSD via eigen clipping.
        NaNs are filled with 0 before projection (conservative "uncorrelated" prior).
    psd_eps:
        Minimum eigenvalue after clipping.

    Returns
    -------
    corr:
        Correlation matrix (DataFrame).
    pair_counts:
        Overlap counts per pair (DataFrame of ints).
    """
    if returns.empty:
        raise ValueError("returns is empty")

    r = returns.copy()
    # Keep numeric columns only
    r = r.apply(pd.to_numeric, errors="coerce")

    cols = r.columns
    n = len(cols)
    corr = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    counts = pd.DataFrame(0, index=cols, columns=cols, dtype=int)

    # Diagonal
    np.fill_diagonal(corr.values, 1.0)
    np.fill_diagonal(counts.values, int(r.shape[0]))

    for i in range(n):
        for j in range(i + 1, n):
            x = r.iloc[:, i]
            y = r.iloc[:, j]
            common = x.notna() & y.notna()
            m = int(common.sum())
            counts.iat[i, j] = m
            counts.iat[j, i] = m
            if m < min_pair_obs:
                continue
            rho = x[common].corr(y[common])
            corr.iat[i, j] = float(rho) if rho is not None else np.nan
            corr.iat[j, i] = float(rho) if rho is not None else np.nan

    if ensure_psd:
        # Conservative fill: treat unknown pairs as uncorrelated.
        filled = corr.fillna(0.0).values
        psd = nearest_psd_eigen_clip(filled, eps=psd_eps)
        corr = pd.DataFrame(psd, index=cols, columns=cols)
        np.fill_diagonal(corr.values, 1.0)

    return corr, counts


def eigenvalues_of_correlation(corr: pd.DataFrame) -> np.ndarray:
    """Return eigenvalues of a correlation matrix sorted descending."""
    a = np.asarray(corr.values, dtype=float)
    a = 0.5 * (a + a.T)
    w = np.linalg.eigvalsh(a)
    w = np.sort(w)[::-1]
    return w


def absorption_ratio(
    eigenvalues: np.ndarray,
    k: Optional[int] = None,
    k_fraction: float = 0.2,
) -> float:
    """
    Absorption Ratio (Kritzman et al.) using top-K eigenvalues.

    For a correlation matrix, sum(eigenvalues) = N (up to numerical error).
    """
    w = np.asarray(eigenvalues, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return np.nan
    n = w.size
    if k is None:
        k = int(np.ceil(k_fraction * n))
    k = max(1, min(int(k), n))
    return float(w[:k].sum() / w.sum())


def spectral_shannon_entropy(
    eigenvalues: np.ndarray,
    normalize: EntropyNorm = "logN",
    base: float = np.e,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Shannon entropy over normalized eigenvalues p_i = λ_i / Σ λ.

    Returns (H, H_norm) where:
      - H is in units of log-base 'base'
      - H_norm is divided by log_base(N) if normalize='logN', else equals H
    """
    w = np.asarray(eigenvalues, dtype=float)
    w = w[w > 0]
    n = w.size
    if n == 0:
        return np.nan, np.nan
    p = w / w.sum()
    logb = np.log(base)
    H = float(-np.sum(p * (_safe_log(p, eps=eps) / logb)))
    if normalize == "logN":
        H_max = float(np.log(n) / logb) if n > 1 else 0.0
        H_norm = float(H / H_max) if H_max > 0 else 0.0
    else:
        H_norm = H
    return H, H_norm


def effective_rank(eigenvalues: np.ndarray, eps: float = 1e-12) -> float:
    """
    Effective rank (Roy & Vetterli): eRank = exp(H_nats),
    where H_nats is Shannon entropy computed with natural log.
    """
    H, _ = spectral_shannon_entropy(eigenvalues, normalize="none", base=np.e, eps=eps)
    if not np.isfinite(H):
        return np.nan
    return float(np.exp(H))


def spectral_stats_from_returns(
    returns: pd.DataFrame,
    min_pair_obs: int = 30,
    ensure_psd: bool = True,
    k: Optional[int] = None,
    k_fraction: float = 0.2,
    entropy_normalize: EntropyNorm = "logN",
) -> SpectralStats:
    """
    Convenience wrapper: returns -> corr -> eigen -> (AR, entropy, eRank).
    """
    corr, _ = pairwise_correlation(
        returns=returns,
        min_pair_obs=min_pair_obs,
        ensure_psd=ensure_psd,
    )
    w = eigenvalues_of_correlation(corr)
    ar = absorption_ratio(w, k=k, k_fraction=k_fraction)
    H, Hn = spectral_shannon_entropy(w, normalize=entropy_normalize)
    er = effective_rank(w)
    return SpectralStats(
        n_assets=int(len(returns.columns)),
        eigenvalues=w,
        absorption_ratio=float(ar),
        entropy=float(H),
        entropy_norm=float(Hn),
        effective_rank=float(er),
    )

