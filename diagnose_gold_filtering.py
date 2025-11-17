"""Diagnose why gold has so few rows"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
SILVER = BASE / 'data/silver'

datasets = [
    'fundamentals/quality_signals.parquet',
    'fundamentals/value_signals.parquet',
    'technicals/momentum_signals.parquet',
    'technicals/risk_signals.parquet'
]

print("=" * 60)
print("DIAGNOSTICO GOLD FILTERING")
print("=" * 60)

for ds_path in datasets:
    full_path = SILVER / ds_path
    if not full_path.exists():
        print(f"\n{ds_path}: FILE NOT FOUND")
        continue

    df = pd.read_parquet(full_path)

    print(f"\n{ds_path}:")
    print(f"  Total rows: {len(df)}")

    if 'ticker' in df.columns:
        print(f"  Unique tickers: {df['ticker'].nunique()}")
        print(f"  Tickers: {sorted(df['ticker'].unique())}")
    else:
        print(f"  NO TICKER COLUMN")

    if 'date' in df.columns:
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique dates: {df['date'].nunique()}")
    elif hasattr(df.index, 'name') and df.index.name == 'date':
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Unique dates: {df.index.nunique()}")

    # Check NaN ratios
    nan_ratio = df.isnull().mean().mean()
    print(f"  NaN ratio: {nan_ratio:.1%}")

# Now check what the intersection would be
print("\n" + "=" * 60)
print("INTERSECTION ANALYSIS")
print("=" * 60)

dfs = {}
for ds_path in datasets:
    full_path = SILVER / ds_path
    if full_path.exists():
        df = pd.read_parquet(full_path)
        dfs[ds_path] = df

# Try to understand the join result
if 'fundamentals/quality_signals.parquet' in dfs and 'technicals/momentum_signals.parquet' in dfs:
    quality = dfs['fundamentals/quality_signals.parquet']
    momentum = dfs['technicals/momentum_signals.parquet']

    print(f"\nQuality tickers: {set(quality['ticker'].unique() if 'ticker' in quality.columns else [])}")
    print(f"Momentum tickers: {set(momentum['ticker'].unique() if 'ticker' in momentum.columns else [])}")

    if 'ticker' in quality.columns and 'ticker' in momentum.columns:
        common_tickers = set(quality['ticker'].unique()) & set(momentum['ticker'].unique())
        print(f"\nCommon tickers: {len(common_tickers)}")
        print(f"  {sorted(common_tickers)}")
