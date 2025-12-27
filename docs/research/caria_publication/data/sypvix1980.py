import os
import time
import json
import requests
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
FMP_API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"  # export FMP_API_KEY="...")

OUT_DIR = Path("./data_fmp_1990")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://financialmodelingprep.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CARIA-FMP-Downloader/1.0"})

def _get_json(url: str, params: dict, retries: int = 4, sleep_s: float = 1.0):
    last_err = None
    for k in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=45)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (k + 1))
    raise RuntimeError(f"GET failed after {retries} retries: {url}\nLast error: {last_err}")

def fetch_price_eod_full(symbol: str, outfile: Path, start: str = "1990-01-01"):
    """
    Uses FMP stable endpoint:
      https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=SYMBOL
    Docs: Stock/Index EOD full. (Index example shows ^GSPC) :contentReference[oaicite:2]{index=2}
    """
    url = f"{BASE}/stable/historical-price-eod/full"
    data = _get_json(url, params={"symbol": symbol, "apikey": FMP_API_KEY})

    # FMP response shape can vary; handle common cases robustly.
    # Often: {"symbol": "...", "historical": [ {date, open, high, low, close, volume, ...}, ... ]}
    if isinstance(data, dict) and "historical" in data and isinstance(data["historical"], list):
        hist = pd.DataFrame(data["historical"])
    elif isinstance(data, list):
        hist = pd.DataFrame(data)
    else:
        raise ValueError(f"Unexpected response for {symbol}: {type(data)} keys={getattr(data, 'keys', lambda:[])()}")

    if hist.empty:
        raise ValueError(f"No data returned for {symbol}")

    if "date" not in hist.columns:
        raise ValueError(f"Missing 'date' column for {symbol}. Columns: {hist.columns.tolist()}")

    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date")
    hist = hist[hist["date"] >= pd.to_datetime(start)]
    hist.to_csv(outfile, index=False)
    print(f"✅ Saved {symbol}: {len(hist)} rows -> {outfile}")

def fetch_sp500_constituents_current(outfile: Path):
    """
    FMP stable S&P 500 constituents:
      https://financialmodelingprep.com/stable/sp500-constituent :contentReference[oaicite:3]{index=3}
    """
    url = f"{BASE}/stable/sp500-constituent"
    data = _get_json(url, params={"apikey": FMP_API_KEY})
    df = pd.DataFrame(data if isinstance(data, list) else data.get("data", data))
    df.to_csv(outfile, index=False)
    print(f"✅ Saved S&P500 current constituents: {len(df)} -> {outfile}")

def fetch_sp500_constituents_history(outfile: Path):
    """
    FMP stable historical changes (add/remove) to S&P 500:
      https://financialmodelingprep.com/stable/historical-sp500-constituent :contentReference[oaicite:4]{index=4}
    """
    url = f"{BASE}/stable/historical-sp500-constituent"
    data = _get_json(url, params={"apikey": FMP_API_KEY})
    df = pd.DataFrame(data if isinstance(data, list) else data.get("data", data))
    df.to_csv(outfile, index=False)
    print(f"✅ Saved S&P500 historical constituent changes: {len(df)} -> {outfile}")

if __name__ == "__main__":
    # --- Prices ---
    # SPY exists from 1993; if quieres “desde 1990”, usa ^GSPC como mercado.
    fetch_price_eod_full("^VIX",  OUT_DIR / "VIX_EOD_full.csv",  start="1990-01-01")
    fetch_price_eod_full("SPY",   OUT_DIR / "SPY_EOD_full.csv",  start="1990-01-01")
    fetch_price_eod_full("^GSPC", OUT_DIR / "GSPC_EOD_full.csv", start="1990-01-01")

    # --- Constituents ---
    fetch_sp500_constituents_current(OUT_DIR / "SP500_constituents_current.csv")
    fetch_sp500_constituents_history(OUT_DIR / "SP500_constituents_changes.csv")

    print("\nDone.")
