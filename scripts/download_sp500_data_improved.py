"""Download SP500 price data with progress tracking"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from pathlib import Path
from datetime import datetime
import time
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
SILVER = BASE_DIR / "data/silver/technicals"
SILVER.mkdir(parents=True, exist_ok=True)

FMP_API_KEY = "your-fmp-api-key-here"

def log(msg):
    """Print with flush"""
    print(msg, flush=True)

log("=" * 60)
log("DOWNLOADING SP500 DATA (IMPROVED)")
log("=" * 60)

# Get SP500 tickers from FMP API
log("\n[1/4] Fetching SP500 ticker list...")
try:
    url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={FMP_API_KEY}"
    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        data = response.json()
        tickers = [item['symbol'] for item in data]
        log(f"  Found {len(tickers)} tickers from FMP")
        log(f"  Sample: {tickers[:10]}")
    else:
        raise Exception(f"FMP API returned {response.status_code}")

except Exception as e:
    log(f"  Error fetching from FMP: {e}")
    log("  Using fallback: 11 tickers")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'UNH', 'JNJ']

# Download price data
log(f"\n[2/4] Downloading price data for {len(tickers)} tickers...")
log(f"  This will take ~{len(tickers) * 0.12 / 60:.0f} minutes...")
log(f"  Progress updates every 10 tickers")

all_data = []
failed = []

for i, ticker in enumerate(tickers):
    if (i + 1) % 10 == 0 or (i + 1) == len(tickers):
        log(f"  Progress: {i+1}/{len(tickers)} ({(i+1)/len(tickers)*100:.1f}%) - {len(all_data)} successful, {len(failed)} failed")

    try:
        data = yf.download(
            ticker,
            start="1990-01-01",
            end=datetime.now().strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True
        )

        if len(data) > 252:  # Need at least 1 year of data
            data = data.reset_index()
            data['ticker'] = ticker

            # Handle multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            data = data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            all_data.append(data)
        else:
            failed.append((ticker, "Insufficient data"))

    except Exception as e:
        failed.append((ticker, str(e)[:50]))

    # Rate limiting
    time.sleep(0.1)

log(f"\n  Downloaded: {len(all_data)} tickers")
log(f"  Failed: {len(failed)} tickers")

if len(failed) > 0 and len(failed) < 20:
    log(f"  Failed tickers: {[t[0] for t in failed]}")

# Combine all data
log(f"\n[3/4] Combining data and calculating technical indicators...")
combined_df = pd.concat(all_data, ignore_index=True)
log(f"  Total rows: {len(combined_df):,}")
log(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")

combined_df = combined_df.sort_values(['ticker', 'date'])

def calculate_indicators(group):
    """Calculate technical indicators for a group (ticker)"""
    df = group.copy()

    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
    df['sma_200'] = df['close'].rolling(window=200, min_periods=1).mean()

    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Volume features
    df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    df['volume_change'] = df['volume'].pct_change()

    return df

log("  Calculating indicators by ticker...")
momentum_df = combined_df.groupby('ticker', group_keys=False).apply(calculate_indicators)

# Select momentum columns
momentum_cols = [
    'date', 'ticker',
    'sma_20', 'sma_50', 'sma_200',
    'ema_20', 'ema_50', 'ema_200',
    'rsi_14', 'macd', 'macd_signal',
    'volume', 'volume_sma_20', 'volume_ratio', 'volume_change'
]

momentum_signals = momentum_df[momentum_cols].copy()

# Calculate risk signals
log(f"\n[4/4] Calculating risk signals...")

def calculate_risk(group):
    """Calculate risk metrics for a group"""
    df = group.copy()

    # Returns
    df['returns_20d'] = df['close'].pct_change(periods=20)
    df['returns_60d'] = df['close'].pct_change(periods=60)
    df['returns_120d'] = df['close'].pct_change(periods=120)

    # Volatility
    df['volatility_30d'] = df['close'].pct_change().rolling(window=30).std()

    # Drawdown
    rolling_max = df['close'].rolling(window=252, min_periods=1).max()
    df['drawdown'] = (df['close'] - rolling_max) / rolling_max

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()

    return df

risk_df = combined_df.groupby('ticker', group_keys=False).apply(calculate_risk)

risk_cols = [
    'date', 'ticker', 'close',
    'returns_20d', 'returns_60d', 'returns_120d',
    'volatility_30d', 'drawdown', 'atr_14'
]

risk_signals = risk_df[risk_cols].copy()

# Save
log(f"\n[5/5] Saving to silver...")
momentum_signals.to_parquet(SILVER / "momentum_signals.parquet", index=False)
risk_signals.to_parquet(SILVER / "risk_signals.parquet", index=False)

log(f"\n  Saved momentum_signals: {len(momentum_signals):,} rows")
log(f"  Saved risk_signals: {len(risk_signals):,} rows")

# Summary
log("\n" + "=" * 60)
log("SP500 DOWNLOAD COMPLETE")
log("=" * 60)
log(f"\nTickers: {len(all_data)} (was 11)")
log(f"Total rows: {len(momentum_signals):,}")
log(f"Date range: {momentum_signals['date'].min()} to {momentum_signals['date'].max()}")
log(f"\nNext steps:")
log("  1. Download fundamentals (FMP)")
log("  2. Re-run rebuild_gold_simple.py")
log("  3. Re-train models with 40x more tickers!")
