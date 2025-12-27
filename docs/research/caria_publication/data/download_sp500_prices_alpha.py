"""
CARIA S&P 500 Price Data Downloader
====================================

Script to download historical daily price data for S&P 500 constituents
using Alpha Vantage API.

Usage:
    python download_sp500_prices_alpha.py

Requirements:
    - ALPHA_VANTAGE_KEY environment variable set
    - pandas, requests libraries

Output:
    - Individual CSV files for each ticker in data/sp500_prices_alpha/
    - failures.csv with download errors
"""

import os
import time
import json
import pandas as pd
import requests
from pathlib import Path

# ====== CONFIG ======
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "3KHQX7KNMNT7H7MZ")
if not ALPHA_VANTAGE_KEY:
    raise RuntimeError("Falta ALPHA_VANTAGE_KEY en variables de entorno")

# Input file with S&P 500 tickers
IN_TICKERS_CSV = Path(__file__).parent / "SP500_current_constituents_from_history.csv"

# Output directory
OUT_DIR = Path(__file__).parent / "sp500_prices_alpha"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "1996-01-01"   # CARIA membership starts 1996
SLEEP_S = 12                # Alpha Vantage rate limit: 5 calls/minute = 12s between calls
MAX_RETRIES = 4

BASE = "https://www.alphavantage.co"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CARIA-AlphaVantage-Downloader/1.0"})

def get_json(url, params, retries=MAX_RETRIES):
    """Get JSON data with retry logic."""
    last = None
    for k in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            print(f"  Retry {k+1}/{retries} for {url}: {str(e)[:80]}")
            time.sleep((k+1) * 1.0)
    raise RuntimeError(f"Failed after {retries} retries: {url} last_err={last}")

def fetch_eod_full(symbol: str):
    """
    Alpha Vantage TIME_SERIES_DAILY_ADJUSTED endpoint
    Returns DataFrame with historical prices.
    """
    url = f"{BASE}/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": ALPHA_VANTAGE_KEY
    }

    try:
        data = get_json(url, params)

        # Check for Alpha Vantage error
        if "Error Message" in data:
            print(f"  Alpha Vantage Error: {data['Error Message']}")
            return pd.DataFrame()

        if "Time Series (Daily)" not in data:
            print(f"  Unexpected response format for {symbol}")
            return pd.DataFrame()

        # Convert Alpha Vantage format to DataFrame
        time_series = data["Time Series (Daily)"]
        records = []

        for date_str, daily_data in time_series.items():
            record = {
                "date": date_str,
                "open": float(daily_data["1. open"]),
                "high": float(daily_data["2. high"]),
                "low": float(daily_data["3. low"]),
                "close": float(daily_data["4. close"]),
                "adjClose": float(daily_data["5. adjusted close"]),
                "volume": int(daily_data["6. volume"])
            }
            records.append(record)

        df = pd.DataFrame(records)

        if df.empty:
            return pd.DataFrame()

        # Process dates and filter
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df[df["date"] >= pd.to_datetime(START_DATE)]

        return df

    except Exception as e:
        print(f"  Error fetching {symbol}: {str(e)[:100]}")
        return pd.DataFrame()

def main():
    """Main download function."""
    print("=" * 70)
    print("CARIA S&P 500 Price Data Downloader")
    print("=" * 70)
    print(f"Input tickers: {IN_TICKERS_CSV}")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print(f"Start date: {START_DATE}")
    print(f"Rate limit delay: {SLEEP_S}s")
    print()

    # Load tickers
    if not IN_TICKERS_CSV.exists():
        raise FileNotFoundError(f"Ticker file not found: {IN_TICKERS_CSV}")

    tickers_df = pd.read_csv(IN_TICKERS_CSV)
    if "ticker" not in tickers_df.columns:
        raise ValueError("CSV must have 'ticker' column")

    tickers = tickers_df["ticker"].astype(str).str.strip().unique().tolist()
    print(f"Found {len(tickers)} unique tickers")

    # Track progress
    failures = []
    done = 0
    skipped = 0

    print("\nStarting download...")
    print("-" * 50)

    for i, ticker in enumerate(tickers, 1):
        out_file = OUT_DIR / f"{ticker}.csv"

        # Skip if already exists and has reasonable size
        if out_file.exists() and out_file.stat().st_size > 1000:
            print(f"[{i:3d}/{len(tickers):3d}] {ticker:<6} - SKIPPED (already exists)")
            skipped += 1
            continue

        print(f"[{i:3d}/{len(tickers):3d}] {ticker:<6} - DOWNLOADING...")
        try:
            df = fetch_eod_full(ticker)

            if df.empty:
                failures.append({"ticker": ticker, "reason": "empty_data"})
                print("  [EMPTY] Empty data")
            else:
                # Save only relevant columns if they exist
                cols = [c for c in ["date","open","high","low","close","adjClose","volume"] if c in df.columns]
                df[cols].to_csv(out_file, index=False)
                done += 1
                print(f"[{i:3d}/{len(tickers):3d}] {ticker:<6} - [OK] {len(df)} rows saved")
            time.sleep(SLEEP_S)

        except Exception as e:
            failures.append({"ticker": ticker, "reason": str(e)})
            print(f"  [ERROR] Error: {str(e)[:50]}")
            time.sleep(SLEEP_S)

    # Save failures log
    if failures:
        failures_df = pd.DataFrame(failures)
        failures_df.to_csv(OUT_DIR / "failures.csv", index=False)
        print(f"\nFailures saved to: {OUT_DIR / 'failures.csv'}")

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Total tickers: {len(tickers)}")
    print(f"Successfully downloaded: {done}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Failed: {len(failures)}")
    print(f"Output directory: {OUT_DIR.resolve()}")

    if done > 0:
        # Show some stats
        csv_files = list(OUT_DIR.glob("*.csv"))
        csv_files = [f for f in csv_files if f.name != "failures.csv"]

        if csv_files:
            sizes = [f.stat().st_size for f in csv_files]
            avg_size = sum(sizes) / len(sizes)
            print(f"Average file size: {avg_size/1024:.0f} KB")
    return done, len(failures)

if __name__ == "__main__":
    main()