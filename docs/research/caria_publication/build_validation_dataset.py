import pandas as pd
import numpy as np
from pathlib import Path

INFILE = Path("full_validation_dataset.csv")
OUTFILE = Path("validation_clean_for_gamlss.csv")

HORIZON = 22         # trading days ahead
ZWIN = 252           # zscore window (1y)
PEAK_WIN = 60        # memory window

def find_date_col(df: pd.DataFrame) -> str:
    for c in ["date", "Date", "datetime", "time", "timestamp"]:
        if c in df.columns:
            return c
    # fallback: first column if it parses as datetime
    c0 = df.columns[0]
    try:
        pd.to_datetime(df[c0])
        return c0
    except Exception:
        raise ValueError("No encontré columna fecha. Renombra a 'date' o similar.")

def rolling_zscore(x: pd.Series, win: int) -> pd.Series:
    m = x.rolling(win, min_periods=win).mean()
    s = x.rolling(win, min_periods=win).std(ddof=0)
    return (x - m) / s

def main():
    df = pd.read_csv(INFILE)

    # --- date index ---
    date_col = find_date_col(df)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # --- sanity: required columns ---
    required = ["price", "volatility"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida: '{c}' en el CSV")

    # --- target: future returns computed on FULL continuous series ---
    df["Future_Ret_22d"] = df["price"].shift(-HORIZON) / df["price"] - 1.0
    df["Future_LogRet_22d"] = np.log(df["price"].shift(-HORIZON) / df["price"])

    # --- structural feature base ---
    # Prefer absorption_ratio if exists; else try 'absorption' or similar
    if "absorption_ratio" in df.columns:
        base = df["absorption_ratio"].astype(float)
    elif "absorption" in df.columns:
        base = df["absorption"].astype(float)
    else:
        raise ValueError("No encuentro absorption_ratio (ni absorption). Necesito eso para absorp_z/peak.")

    # NO backfill. Warmup is expected.
    df["absorp_z"] = rolling_zscore(base, ZWIN)

    # memory peak: trailing mean of zscore
    df["peak_60"] = df["absorp_z"].rolling(PEAK_WIN, min_periods=PEAK_WIN).mean()

    # Optional: keep your existing peak_60 if you want to compare
    # but DO NOT use duplicates together in the same model.
    # if "peak_60" in df.columns: ...

    # --- clean set ---
    cols_keep = ["Future_Ret_22d", "Future_LogRet_22d", "volatility", "peak_60", "absorp_z"]
    # add other covariates if present (entropy, rates, etc.) but only if you will use them
    for c in ["entropy", "treasury_10y", "tlt"]:
        if c in df.columns:
            cols_keep.append(c)

    out = df[cols_keep].copy()

    # drop rows where any required piece is missing (warmup + horizon tail)
    out = out.dropna(subset=["Future_Ret_22d", "volatility", "peak_60"])

    # Quick diagnostics to avoid “edge spline madness”
    # (not deleting, just reporting)
    q01, q99 = out["peak_60"].quantile([0.01, 0.99])
    print("Rows:", len(out))
    print("peak_60 p01/p99:", float(q01), float(q99))
    print("volatility min/max:", float(out["volatility"].min()), float(out["volatility"].max()))

    out.to_csv(OUTFILE, index=True)
    print(f"OK -> {OUTFILE.resolve()}")

if __name__ == "__main__":
    main()
