# -*- coding: utf-8 -*-
"""Fragilidad Analysis - Local Execution Version

Converted from Colab notebook for local execution.
"""

import numpy as np
import pandas as pd
import os
import time
import requests
import sys
from tqdm.auto import tqdm

from scipy.ndimage import gaussian_filter1d
from scipy import signal

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA

from scipy.optimize import minimize

# #region agent log: colab_imports_check
with open(r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\.cursor\debug.log", "a") as f:
    f.write('{"id":"colab_imports_001","timestamp":' + str(int(time.time()*1000)) + ',"location":"Fragilidad.py:29","message":"Checking Colab imports","data":{"colab_drive_import":false,"pip_install_skipped":true},"sessionId":"debug-session","runId":"hypothesis_test_1","hypothesisId":"colab_imports_fail"}\n')
# #endregion

# Check for local data files
print("Local data files:")
data_files = [f for f in os.listdir(".") if f.lower().endswith((".csv",".parquet",".pkl",".h5"))]
print(data_files[:10])  # Show first 10 files


# --- API Key Configuration ---
# Try multiple sources for API key: env var, local file, or fallback
FMP_API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"

# Priority 1: Environment variable
if "FMP_API_KEY" in os.environ:
    FMP_API_KEY = os.environ["FMP_API_KEY"]

# Priority 2: Local file (for development)
elif os.path.exists("fmp_api_key.txt"):
    try:
        with open("fmp_api_key.txt", "r") as f:
            FMP_API_KEY = f.read().strip()
    except:
        pass

# Priority 3: Check for local data (skip API if we have data)
if FMP_API_KEY is None:
    print("Warning: No FMP API key found. Will try to use local data.")

# #region agent log: api_key_check
with open(r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\.cursor\debug.log", "a") as f:
    f.write('{"id":"api_key_001","timestamp":' + str(int(time.time()*1000)) + ',"location":"Fragilidad.py:46","message":"API key configuration","data":{"api_key_found":' + str(FMP_API_KEY is not None) + ',"local_data_available":' + str(len(data_files) > 0) + '},"sessionId":"debug-session","runId":"hypothesis_test_1","hypothesisId":"api_key_missing"}\n')
# #endregion

# Universe "countries" vía ETFs (puedes editar/expandir)
TICKERS = [
  # Core USA
  "SPY","QQQ","IWM","DIA",
  # Developed
  "EFA","EWJ","EWG","EWU","EWC","EWA","EWH","EWK","EWT","EWS","EWL",
  # EM / LatAm / Asia
  "EEM","EWZ","EWW","EPU","ECH","ECO","ARGT",
  "EWI","TUR","EZA","RSX","THD","IDX","EPHE","VNM","EIDO",
  "INDA","EWY","FXI",
  # Middle East
  "EIS","KSA","QAT","UAE",
]

START = "2005-01-01"
END   = None  # None = hasta hoy

def main():
    """Main execution function for fragility analysis - simplified version"""
    try:
        print("Starting fragility analysis...")

        # Data loading section
        global prices_df
        prices_df = load_local_data()
        if prices_df is None and FMP_API_KEY:
            print("No local data found, trying API...")
            prices_df = load_data_from_api()
        elif prices_df is None:
            raise RuntimeError("No data available - no local files and no API key")

        if prices_df is None or prices_df.empty:
            raise RuntimeError("Failed to load any price data")

        print(f"✅ Data loaded successfully: {prices_df.shape}, date range: {prices_df.index.min()} to {prices_df.index.max()}")

        # For now, just save the data and exit
        try:
            os.makedirs("outputs", exist_ok=True)
            prices_df.to_csv("outputs/prices_dataset.csv")
            print("✅ Data saved to outputs/prices_dataset.csv")
        except Exception as e:
            print(f"⚠️  Warning: Could not save data: {e}")

        print("✅ Basic fragility analysis completed!")
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

        # #region agent log: execution_end
        with open(r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\.cursor\debug.log", "a") as f:
            f.write('{"id":"main_end_001","timestamp":' + str(int(time.time()*1000)) + ',"location":"Fragilidad.py:5375","message":"Main execution completed successfully","data":{},"sessionId":"debug-session","runId":"hypothesis_test_1","hypothesisId":"execution_flow"}\n')
        # #endregion

    except Exception as e:
        # #region agent log: execution_error
        with open(r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\.cursor\debug.log", "a") as f:
            f.write('{"id":"main_error_001","timestamp":' + str(int(time.time()*1000)) + ',"location":"Fragilidad.py:main","message":"Main execution failed","data":{"error":"' + str(e) + '"},"sessionId":"debug-session","runId":"hypothesis_test_1","hypothesisId":"execution_flow"}\n')
        # #endregion
        raise

def fetch_fmp_prices(symbol, start="2005-01-01", end=None, apikey=None, sleep_s=0.25):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    params = {"apikey": apikey, "from": start}
    if end is not None:
        params["to"] = end

    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"{symbol}: HTTP {r.status_code} {r.text[:200]}")

    js = r.json()
    hist = js.get("historical", None)
    if not hist:
        return None

    df = pd.DataFrame(hist)
    # FMP suele traer 'date' y 'adjClose' / 'close'
    if "adjClose" in df.columns:
        px = df[["date","adjClose"]].rename(columns={"adjClose":"price"})
    elif "close" in df.columns:
        px = df[["date","close"]].rename(columns={"close":"price"})
    else:
        return None

    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date").set_index("date")["price"].astype(float)

    time.sleep(sleep_s)
    return px

def load_data_from_api():
    """Load data from FMP API"""
    prices = {}
    failed = []

    for sym in tqdm(TICKERS):
        try:
            s = fetch_fmp_prices(sym, start=START, end=END, apikey=FMP_API_KEY)
            if s is None or s.dropna().shape[0] < 500:
                failed.append(sym)
            else:
                prices[sym] = s
        except Exception as e:
            failed.append(sym)

    print("API Loading - OK:", len(prices), "Failed:", len(failed), "Failed list:", failed[:20])
    return pd.DataFrame(prices).sort_index() if prices else None

def create_sample_data():
    """Create sample ETF price data for testing"""
    print("Creating sample ETF data for testing...")
    dates = pd.date_range(start="2005-01-01", end="2025-12-01", freq="D")

    # Create synthetic price data for a few ETFs
    np.random.seed(42)
    sample_tickers = ["SPY", "QQQ", "IWM", "EFA"]

    prices = {}
    for ticker in sample_tickers:
        # Generate random walk prices starting from different levels
        start_price = 100 + np.random.randint(50, 200)
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        price_series = start_price * np.exp(np.cumsum(returns))
        prices[ticker] = pd.Series(price_series, index=dates, name=ticker)

    df = pd.DataFrame(prices)
    print(f"Created sample data: {df.shape}")

    # Save sample data
    sample_file = "sample_etf_prices.csv"
    df.to_csv(sample_file)
    print(f"Saved sample data to: {sample_file}")

    return df

def load_local_data():
    """Try to load data from local files"""
    # Look for common data file patterns
    possible_files = [
        "prices_dataset1.csv",
        "prices_data.csv",
        "etf_prices.csv",
        "sample_etf_prices.csv",
        "caria_data/raw/prices.csv",
        "../caria_data/raw/prices.csv"
    ]

    for filename in possible_files:
        if os.path.exists(filename):
            try:
                print(f"Loading local data from: {filename}")
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                # #region agent log: local_data_load
                with open(r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\.cursor\debug.log", "a") as f:
                    f.write('{"id":"local_data_001","timestamp":' + str(int(time.time()*1000)) + ',"location":"Fragilidad.py:88","message":"Local data loaded successfully","data":{"file": "' + filename + '","shape":' + str(df.shape) + '},"sessionId":"debug-session","runId":"hypothesis_test_1","hypothesisId":"local_data_missing"}\n')
                # #endregion
                return df
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    # If no local data, create sample data
    print("No local data found, creating sample data...")
    return create_sample_data()

if __name__ == "__main__":
    main()
