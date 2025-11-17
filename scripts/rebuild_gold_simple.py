"""Rebuild gold dataset without Prefect - simple version"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

SILVER = BASE_DIR / "data/silver"
GOLD = BASE_DIR / "data/gold"

def merge_with_asof(base, df, base_name, df_name):
    """Merge using merge_asof for mixed frequency data"""
    if df.empty:
        return base
    if base is None or base.empty:
        return df

    # Check if we have date and ticker
    if "date" not in base.columns or "date" not in df.columns:
        return base.merge(df, on=["date", "ticker"], how="left")

    base_dates = base["date"].nunique()
    df_dates = df["date"].nunique()

    LOGGER.info(f"Merging {base_name} ({base_dates} dates) with {df_name} ({df_dates} dates)")

    # If frequencies are very different, use merge_asof
    if base_dates / df_dates > 10 or df_dates / base_dates > 10:
        LOGGER.info(f"  Using merge_asof (mixed frequency)")

        base["date"] = pd.to_datetime(base["date"])
        df["date"] = pd.to_datetime(df["date"])

        # Remove rows with NaN in join columns
        base = base.dropna(subset=["date", "ticker"])
        df = df.dropna(subset=["date", "ticker"])

        # merge_asof by ticker requires special handling
        # Do it ticker by ticker to ensure proper sorting
        results = []
        for ticker in base["ticker"].unique():
            base_ticker = base[base["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            df_ticker = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)

            if len(df_ticker) == 0:
                results.append(base_ticker)
                continue

            merged_ticker = pd.merge_asof(
                base_ticker, df_ticker,
                on="date",
                direction="backward",
                suffixes=("", "_dup")
            )
            results.append(merged_ticker)

        result = pd.concat(results, ignore_index=True)

        # Remove duplicate columns
        dup_cols = [c for c in result.columns if c.endswith("_dup")]
        if dup_cols:
            result = result.drop(columns=dup_cols)

        return result
    else:
        LOGGER.info(f"  Using regular merge")
        return base.merge(df, on=["date", "ticker"], how="left")

print("=" * 60)
print("REBUILDING GOLD DATASET WITH MERGE_ASOF")
print("=" * 60)

# Load datasets
print("\n[1/6] Loading quality signals...")
quality = pd.read_parquet(SILVER / "fundamentals/quality_signals.parquet")
print(f"  {len(quality)} rows, {quality['date'].nunique()} dates")

print("\n[2/6] Loading value signals...")
value = pd.read_parquet(SILVER / "fundamentals/value_signals.parquet")
print(f"  {len(value)} rows, {value['date'].nunique()} dates")

print("\n[3/6] Loading momentum signals...")
momentum = pd.read_parquet(SILVER / "technicals/momentum_signals.parquet")
print(f"  {len(momentum)} rows, {momentum['date'].nunique()} dates")

print("\n[4/6] Loading risk signals...")
risk = pd.read_parquet(SILVER / "technicals/risk_signals.parquet")
print(f"  {len(risk)} rows, {risk['date'].nunique()} dates")

# Merge using asof
print("\n[5/6] Merging datasets...")
merged = merge_with_asof(momentum, quality, "momentum", "quality")
merged = merge_with_asof(merged, value, "momentum+quality", "value")
merged = merge_with_asof(merged, risk, "momentum+quality+value", "risk")

print(f"\n  Final merged: {len(merged)} rows")

# Compute targets
print("\n[6/6] Computing targets and splits...")
merged["date"] = pd.to_datetime(merged["date"])
merged = merged.sort_values(["ticker", "date"])

# Target: 4-quarter forward return
merged["target"] = merged.groupby("ticker")["close"].pct_change(periods=4).shift(-4)

# Drop rows without target
merged = merged.dropna(subset=["target"])
print(f"  After dropping NaN targets: {len(merged)} rows")

# Extract feature columns
exclude = {"date", "ticker", "period", "target", "close", "open", "high", "low", "volume"}
feature_cols = [c for c in merged.select_dtypes(include=[np.number]).columns if c not in exclude]
print(f"  Feature columns: {len(feature_cols)}")

# Fill NaN in features
merged[feature_cols] = merged[feature_cols].fillna(0.0)
merged[feature_cols] = merged[feature_cols].replace([np.inf, -np.inf], 0.0)

# Create splits - extended to use full history from 1970
splits = {
    "train": ("1970-01-01", "2019-12-31"),  # Extended from 1985 to 1970 (50 years)
    "val": ("2020-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2024-11-07")     # Extended to latest available
}

GOLD.mkdir(exist_ok=True)

for split_name, (start, end) in splits.items():
    mask = (merged["date"] >= start) & (merged["date"] <= end)
    split_df = merged.loc[mask].copy()

    print(f"\n  {split_name}: {len(split_df)} rows ({start} to {end})")

    if len(split_df) > 0:
        split_df.to_parquet(GOLD / f"{split_name}.parquet", index=False)

print("\n" + "=" * 60)
print("GOLD REBUILD COMPLETE")
print("=" * 60)

# Show final stats
train = pd.read_parquet(GOLD / "train.parquet")
val = pd.read_parquet(GOLD / "val.parquet")
test = pd.read_parquet(GOLD / "test.parquet")

print(f"\nFinal dataset sizes:")
print(f"  Train: {len(train)} rows (was 930)")
print(f"  Val: {len(val)} rows (was 132)")
print(f"  Test: {len(test)} rows (was 77)")
print(f"\n  Total: {len(train) + len(val) + len(test)} rows (was 1,139)")
