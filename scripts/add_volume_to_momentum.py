"""Download volume from yfinance and add to momentum signals"""

import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
SILVER = BASE_DIR / "data/silver"

print("=" * 60)
print("ADDING VOLUME TO MOMENTUM SIGNALS")
print("=" * 60)

# Load existing momentum
momentum_path = SILVER / "technicals/momentum_signals.parquet"
momentum_df = pd.read_parquet(momentum_path)

print(f"\n[1/4] Loaded momentum: {len(momentum_df)} rows")
print(f"  Date range: {momentum_df['date'].min()} to {momentum_df['date'].max()}")
print(f"  Tickers: {momentum_df['ticker'].unique().tolist()}")

# Tickers
tickers = momentum_df['ticker'].unique().tolist()

# Download volume from yfinance
print(f"\n[2/4] Downloading volume from yfinance...")
volume_data = []

for ticker in tickers:
    print(f"  Downloading {ticker}...")
    try:
        # Download desde 1970 hasta hoy
        ticker_data = yf.download(
            ticker,
            start="1970-01-01",
            end=datetime.now().strftime("%Y-%m-%d"),
            progress=False
        )

        if len(ticker_data) > 0:
            ticker_data = ticker_data.reset_index()
            ticker_data['ticker'] = ticker

            # Handle column names (might have multi-level columns)
            if isinstance(ticker_data.columns, pd.MultiIndex):
                ticker_data.columns = ticker_data.columns.droplevel(1)

            ticker_data = ticker_data.rename(columns={'Date': 'date', 'Volume': 'volume'})
            ticker_data = ticker_data[['date', 'ticker', 'volume']]
            volume_data.append(ticker_data)
            print(f"    {len(ticker_data)} rows")
        else:
            print(f"    No data")
    except Exception as e:
        print(f"    Error: {e}")

# Combine
volume_df = pd.concat(volume_data, ignore_index=True)
print(f"\n  Total volume rows: {len(volume_df)}")

# Merge with momentum
print(f"\n[3/4] Merging with momentum...")
momentum_df['date'] = pd.to_datetime(momentum_df['date'])
volume_df['date'] = pd.to_datetime(volume_df['date'])

merged = momentum_df.merge(volume_df, on=['date', 'ticker'], how='left')
print(f"  Merged: {len(merged)} rows")
print(f"  Volume NaN: {merged['volume'].isna().sum()} / {len(merged)} ({merged['volume'].isna().mean():.1%})")

# Calculate volume features
print(f"\n[4/4] Calculating volume features...")

# Sort by ticker and date
merged = merged.sort_values(['ticker', 'date'])

# Volume SMA 20
merged['volume_sma_20'] = merged.groupby('ticker')['volume'].transform(
    lambda x: x.rolling(window=20, min_periods=1).mean()
)

# Volume ratio (current / SMA 20)
merged['volume_ratio'] = merged['volume'] / merged['volume_sma_20']

# Volume change
merged['volume_change'] = merged.groupby('ticker')['volume'].pct_change()

# Fill NaN with 1.0 for volume_ratio
merged['volume_ratio'] = merged['volume_ratio'].fillna(1.0)
merged['volume_change'] = merged['volume_change'].fillna(0.0)

print(f"  Added features: volume, volume_sma_20, volume_ratio, volume_change")

# Save updated momentum
output_path = SILVER / "technicals/momentum_signals.parquet"
merged.to_parquet(output_path, index=False)

print(f"\n[OK] Saved to {output_path}")
print(f"\nNew columns: {merged.columns.tolist()}")

print("\n" + "=" * 60)
print("VOLUME ADDED SUCCESSFULLY")
print("=" * 60)
print("\nNext: Re-run rebuild_gold_simple.py and train_models_corrected.py")
