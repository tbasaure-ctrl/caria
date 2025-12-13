"""
CARIA-SR (Spectral) — reproducible, audit-friendly implementation
==================================================================

This module implements a *cross-sectional* CARIA-SR signal derived from the
spectral structure of the correlation matrix (absorption ratio + eigen-entropy)
and a memory (hysteresis) component.

It is designed to address common fatal issues in empirical asset-pricing code:

- **No forward-fill of prices** for correlation estimation.
- **Entropy comparable over time** even when N varies:
    use H_norm = H / log(N) (or effective-rank).
- **Point-in-time** rolling computation (no in-sample thresholding/leakage).

Important limitation:
---------------------
This module does NOT solve survivorship bias by itself. To avoid survivorship,
the user must provide a historical constituents panel (date → constituents).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Literal

import numpy as np
import pandas as pd

from .spectral import spectral_stats_from_returns, EntropyNorm

ScoreNorm = Literal["zscore", "robust_zscore"]


@dataclass(frozen=True)
class CariaSRSpectralConfig:
    window: int = 252
    min_pair_obs: int = 60
    min_assets: int = 50

    # Absorption ratio definition
    k: Optional[int] = None
    k_fraction: float = 0.2

    # Entropy normalization for comparability across varying N
    entropy_normalize: EntropyNorm = "logN"

    # CARIA raw formula
    eps: float = 1e-8  # protects division by zero

    # Standardization
    score_norm: ScoreNorm = "robust_zscore"
    norm_lookback: int = 252

    # Hysteresis / memory
    memory_halflife: int = 63  # ~quarter


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.apply(pd.to_numeric, errors="coerce")
    return np.log(p).diff()


def _robust_zscore(x: pd.Series, lookback: int) -> pd.Series:
    """
    Rolling robust z-score using median / MAD.
    z = (x - median) / (1.4826 * MAD)
    """
    med = x.rolling(lookback, min_periods=max(20, lookback // 5)).median()
    mad = (x - med).abs().rolling(lookback, min_periods=max(20, lookback // 5)).median()
    scale = 1.4826 * mad
    return (x - med) / scale.replace(0.0, np.nan)


def _zscore(x: pd.Series, lookback: int) -> pd.Series:
    mu = x.rolling(lookback, min_periods=max(20, lookback // 5)).mean()
    sd = x.rolling(lookback, min_periods=max(20, lookback // 5)).std()
    return (x - mu) / sd.replace(0.0, np.nan)


def _ewma_memory(x: pd.Series, halflife: int) -> pd.Series:
    """
    Simple hysteresis: exponential memory of the standardized score.
    """
    # Pandas ewm uses alpha = 1 - exp(log(0.5)/halflife)
    return x.ewm(halflife=halflife, adjust=False, min_periods=1).mean()


def compute_caria_sr_spectral(
    prices: pd.DataFrame,
    config: Optional[CariaSRSpectralConfig] = None,
) -> pd.DataFrame:
    """
    Compute CARIA-SR spectral series from a wide price panel.

    Parameters
    ----------
    prices:
        DataFrame (dates x assets) of *prices* (not returns), with NaNs allowed.
        Do NOT forward-fill gaps. Missingness is handled point-in-time.
    config:
        Hyperparameters for rolling window, AR definition, normalization, memory.

    Returns
    -------
    DataFrame indexed by date with columns:
      - n_assets_used
      - absorption_ratio
      - entropy_norm
      - effective_rank
      - caria_raw
      - caria_z
      - caria_memory
    """
    if config is None:
        config = CariaSRSpectralConfig()

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    rets = _log_returns(prices)

    out = []
    dates = rets.index

    for end_idx in range(config.window - 1, len(dates)):
        end_date = dates[end_idx]
        window_rets = rets.iloc[end_idx - config.window + 1 : end_idx + 1]

        # Drop assets with all missing in the window
        window_rets = window_rets.dropna(axis=1, how="all")

        # Enforce a minimum amount of data per asset to reduce instability
        min_obs_per_asset = max(20, int(0.7 * config.window))
        ok_cols = window_rets.columns[window_rets.notna().sum(axis=0) >= min_obs_per_asset]
        window_rets = window_rets[ok_cols]

        if window_rets.shape[1] < config.min_assets:
            out.append(
                {
                    "date": end_date,
                    "n_assets_used": int(window_rets.shape[1]),
                    "absorption_ratio": np.nan,
                    "entropy_norm": np.nan,
                    "effective_rank": np.nan,
                    "caria_raw": np.nan,
                }
            )
            continue

        stats = spectral_stats_from_returns(
            returns=window_rets,
            min_pair_obs=config.min_pair_obs,
            ensure_psd=True,
            k=config.k,
            k_fraction=config.k_fraction,
            entropy_normalize=config.entropy_normalize,
        )

        # Core raw score (explicit and reproducible):
        # Higher AR => more common factor dominance (systemic coupling)
        # Lower entropy_norm => more concentration (lower effective dimension)
        caria_raw = stats.absorption_ratio / (stats.entropy_norm + config.eps)

        out.append(
            {
                "date": end_date,
                "n_assets_used": int(stats.n_assets),
                "absorption_ratio": float(stats.absorption_ratio),
                "entropy_norm": float(stats.entropy_norm),
                "effective_rank": float(stats.effective_rank),
                "caria_raw": float(caria_raw),
            }
        )

    df = pd.DataFrame(out).set_index("date")

    # Standardize point-in-time (rolling)
    if config.score_norm == "robust_zscore":
        df["caria_z"] = _robust_zscore(df["caria_raw"], config.norm_lookback)
    else:
        df["caria_z"] = _zscore(df["caria_raw"], config.norm_lookback)

    # Hysteresis / memory state (EWMA on standardized score)
    df["caria_memory"] = _ewma_memory(df["caria_z"], config.memory_halflife)

    return df


def split_by_date(
    df: pd.DataFrame,
    splits: Dict[str, tuple[str, str]],
) -> Dict[str, pd.DataFrame]:
    """Utility: split a time series DataFrame into named date ranges."""
    out: Dict[str, pd.DataFrame] = {}
    for name, (start, end) in splits.items():
        out[name] = df.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()
    return out

