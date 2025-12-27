"""
Run CARIA-SR structural (cross-sectional) hysteresis signal + simple hedge backtest.

Uses local cross-sectional price panel:
  docs/research/caria_publication/data/sp500_universe_fmp.parquet

Downloads (via yfinance):
  - ^VIX
  - SPY
  - TLT

Outputs:
  - Deep-calm robustness stats (Δq05)
  - Simple "Minsky hedge" backtest: switch SPY -> TLT when fragile
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


def _require(pkg: str):
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg}'. Install it (pip install {pkg}). Original error: {e}"
        )


_require("pyarrow")
_require("yfinance")
_require("sklearn")

import yfinance as yf  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402


DATA_PATH_DEFAULT = os.path.join("data", "sp500_universe_fmp.parquet")


@dataclass
class Perf:
    cagr: float
    vol: float
    sharpe: float
    max_dd: float
    calmar: float


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) == 0:
        return float("nan")
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def perf_stats(ret: pd.Series, ann: int = 252) -> Perf:
    r = ret.dropna()
    if len(r) < 30:
        return Perf(np.nan, np.nan, np.nan, np.nan, np.nan)
    eq = (1 + r).cumprod()
    years = len(r) / ann
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(r.std() * np.sqrt(ann))
    sharpe = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(ann))
    mdd = max_drawdown(eq)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    return Perf(cagr, vol, sharpe, mdd, calmar)


def forward_min(x: pd.Series, h: int = 22) -> pd.Series:
    a = x.to_numpy()
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        j0, j1 = i + 1, min(len(a), i + 1 + h)
        if j0 < j1:
            out[i] = np.nanmin(a[j0:j1])
    return pd.Series(out, index=x.index)


def eig_metrics_from_corr(C: np.ndarray, k_frac: float = 0.2) -> Tuple[float, float]:
    w = np.linalg.eigvalsh(C)
    w = np.sort(w)[::-1]
    n = len(w)
    k = max(1, int(np.ceil(k_frac * n)))
    ar = float(np.sum(w[:k]) / np.sum(w))
    p = w / np.sum(w)
    p = p[p > 0]
    ent = -np.sum(p * np.log(p))
    ent_norm = float(ent / np.log(n)) if n > 1 else np.nan
    return ar, ent_norm


def cov_to_corr(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(S))
    d = np.where(d == 0, np.nan, d)
    C = S / np.outer(d, d)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    # force symmetry
    return (C + C.T) / 2.0


def rolling_structural_metrics(
    ret_mat: pd.DataFrame,
    window: int = 252,
    k_frac: float = 0.2,
    min_assets: int = 120,
    step: int = 5,
    coverage_in_window: float = 0.90,
) -> pd.DataFrame:
    idx = ret_mat.index
    out = pd.DataFrame(index=idx, columns=["AR", "E_eig", "N_assets"], dtype=float)

    lw = LedoitWolf()

    for t in range(window, len(idx), step):
        W = ret_mat.iloc[t - window + 1 : t + 1]
        good = W.notna().mean() >= coverage_in_window
        W = W.loc[:, good]
        if W.shape[1] < min_assets:
            continue

        # fill residual nans with col mean and de-mean per col
        W = W.apply(lambda s: s.fillna(s.mean()), axis=0)
        X = W.to_numpy()
        X = X - np.nanmean(X, axis=0, keepdims=True)

        try:
            S = lw.fit(X).covariance_
            C = cov_to_corr(S)
        except Exception:
            C = np.corrcoef(X, rowvar=False)
            C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
            C = (C + C.T) / 2.0

        ar, ent = eig_metrics_from_corr(C, k_frac=k_frac)
        out.iloc[t] = [ar, ent, W.shape[1]]

    return out.ffill().bfill()


def zscore(s: pd.Series, w: int = 252) -> pd.Series:
    mu = s.rolling(w).mean()
    sd = s.rolling(w).std().replace(0, np.nan)
    return (s - mu) / sd


def download_adj_close(symbol: str, start: str, end: str) -> pd.Series:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance download empty for {symbol}")
    # yfinance can return either flat columns or MultiIndex columns even for 1 ticker
    if isinstance(df.columns, pd.MultiIndex):
        # try common layouts: ('Adj Close',) level 0
        if ("Adj Close" in df.columns.get_level_values(0)):
            s = df.loc[:, ("Adj Close",)].iloc[:, 0]
        elif ("Close" in df.columns.get_level_values(0)):
            s = df.loc[:, ("Close",)].iloc[:, 0]
        else:
            raise RuntimeError(f"Unexpected MultiIndex columns for {symbol}: {df.columns}")
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            raise RuntimeError(f"Unexpected columns for {symbol}: {df.columns}")
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    return s.rename(symbol)


def main(
    data_path: str = DATA_PATH_DEFAULT,
    start: str = "2000-01-01",
    coverage_min: float = 0.90,
    window: int = 252,
    step: int = 5,
    k_frac: float = 0.2,
    min_assets: int = 120,
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}")

    px = pd.read_parquet(data_path)
    if "date" in px.columns:
        px["date"] = pd.to_datetime(px["date"])
        px = px.set_index("date")
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    px = px.loc[start:].copy()

    ret = np.log(px).diff()
    coverage = 1.0 - ret.isna().mean()
    keep = coverage[coverage >= coverage_min].index.tolist()
    ret = ret[keep]

    # Download external series and align calendar
    start_dl = str(ret.index.min().date())
    end_dl = str((ret.index.max() + pd.Timedelta(days=1)).date())
    vix = download_adj_close("^VIX", start_dl, end_dl).rename("VIX")
    spy = download_adj_close("SPY", start_dl, end_dl).rename("SPY")
    tlt = download_adj_close("TLT", start_dl, end_dl).rename("TLT")

    idx = ret.index.intersection(vix.index).intersection(spy.index).intersection(tlt.index)
    ret_cs = ret.loc[idx].copy()
    vix = vix.loc[idx].copy()
    spy = spy.loc[idx].copy()
    tlt = tlt.loc[idx].copy()

    spy_ret = np.log(spy).diff()
    tlt_ret = np.log(tlt).diff()

    print(f"Aligned dates: {idx.min().date()} → {idx.max().date()} (n={len(idx)})")
    print(f"Cross-sectional universe: {ret_cs.shape[1]} tickers (coverage>={coverage_min})")

    struct = rolling_structural_metrics(
        ret_cs,
        window=window,
        k_frac=k_frac,
        min_assets=min_assets,
        step=step,
        coverage_in_window=coverage_min,
    )
    struct["CARIA_SR"] = zscore(struct["AR"], window) + zscore(1.0 - struct["E_eig"], window)
    struct["Peak60"] = struct["CARIA_SR"].rolling(60).max()
    struct["PeakZ"] = zscore(struct["Peak60"], window)

    y = forward_min(spy_ret, 22).rename("future_min_22d")

    # Deep calm robustness: Δq05 for Peak memory
    H_list = [20, 40, 60, 90, 120]
    thr_list = [15, 18, 20, 22, 25]
    out_rows: List[Dict] = []
    for H in H_list:
        peak = struct["CARIA_SR"].rolling(H).max()
        for thr in thr_list:
            calm = vix < thr
            df = pd.concat([y, peak.rename("peak")], axis=1).loc[calm].dropna()
            if len(df) < 400:
                continue
            cut = df["peak"].quantile(0.80)
            dq = float(df[df["peak"] >= cut]["future_min_22d"].quantile(0.05) - df[df["peak"] < cut]["future_min_22d"].quantile(0.05))
            out_rows.append({"H": H, "VIX_thr": thr, "n": len(df), "dq05": dq})

    if out_rows:
        df_out = pd.DataFrame(out_rows).sort_values(["VIX_thr", "H"])
        print("\nDeep Calm Δq05 (HIGH PeakMemory - LOW), negative=worse tail under memory:")
        for thr in thr_list:
            sub = df_out[df_out["VIX_thr"] == thr]
            if sub.empty:
                continue
            print(f"  VIX<{thr}: " + ", ".join([f"H={int(r.H)}:{r.dq05:+.4f}" for r in sub.itertuples(index=False)]))

    # Strategy: Minsky hedge (switch SPY->TLT) when fragile
    # Default fragile rule: deep-calm AND PeakZ > 1.5 (sustained structural memory while VIX low)
    # We report both VIX<15 and VIX<20 variants.
    for vix_thr in [15, 20]:
        fragile = (vix < vix_thr) & (struct["PeakZ"] > 1.5)
        strat_ret = pd.Series(np.where(fragile, tlt_ret, spy_ret), index=idx).rename("strategy")
        bh_ret = spy_ret.rename("buyhold_spy")
        tlt_only = tlt_ret.rename("tlt_only")

        ps = perf_stats(strat_ret)
        pb = perf_stats(bh_ret)
        pt = perf_stats(tlt_only)

        eq_s = (1 + strat_ret.fillna(0)).cumprod()
        eq_b = (1 + bh_ret.fillna(0)).cumprod()

        print(f"\n=== Minsky Hedge (SPY->TLT) rule: (VIX<{vix_thr}) & (PeakZ>1.5) ===")
        print(f"Time in hedge: {float(fragile.mean()*100):.2f}%")
        print(f"Strategy: CAGR={ps.cagr:.3%} Vol={ps.vol:.3%} Sharpe={ps.sharpe:.2f} MaxDD={ps.max_dd:.1%} Calmar={ps.calmar:.2f}")
        print(f"Buy&Hold SPY: CAGR={pb.cagr:.3%} Vol={pb.vol:.3%} Sharpe={pb.sharpe:.2f} MaxDD={pb.max_dd:.1%} Calmar={pb.calmar:.2f}")
        print(f"TLT only: CAGR={pt.cagr:.3%} Vol={pt.vol:.3%} Sharpe={pt.sharpe:.2f} MaxDD={pt.max_dd:.1%} Calmar={pt.calmar:.2f}")
        # Event drawdowns
        for label, (a, b) in {"GFC": ("2007-10-01", "2009-06-30"), "COVID": ("2020-02-01", "2020-06-30")}.items():
            mdd_s = max_drawdown(eq_s.loc[a:b])
            mdd_b = max_drawdown(eq_b.loc[a:b])
            print(f"  {label} MaxDD: strategy={mdd_s:.1%} vs SPY={mdd_b:.1%}")

if __name__ == "__main__":
    main()


