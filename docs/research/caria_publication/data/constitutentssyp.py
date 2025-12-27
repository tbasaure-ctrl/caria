import os, time, json
import pandas as pd
import requests
from pathlib import Path

# ====== CONFIG ======
FMP_API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"  # set en env var
if not FMP_API_KEY:
    raise RuntimeError("Falta FMP_API_KEY en variables de entorno")

IN_TICKERS_CSV = r"/mnt/data/SP500_current_constituents_from_history.csv"  # ajusta ruta
OUT_DIR = Path("./sp500_prices_fmp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "1996-01-01"   # tu membership parte 1996
SLEEP_S = 0.25              # sube esto si te rate-limitea
MAX_RETRIES = 4

BASE = "https://financialmodelingprep.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CARIA-FMP-Downloader/1.0"})

def get_json(url, params, retries=MAX_RETRIES):
    last = None
    for k in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep((k+1) * 1.0)
    raise RuntimeError(f"Failed: {url} last_err={last}")

def fetch_eod_full(symbol: str):
    """
    FMP stable endpoint: /stable/historical-price-eod/full?symbol=...
    """
    url = f"{BASE}/stable/historical-price-eod/full"
    data = get_json(url, {"symbol": symbol, "apikey": FMP_API_KEY})
    if isinstance(data, dict) and "historical" in data:
        df = pd.DataFrame(data["historical"])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return pd.DataFrame()

    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df[df["date"] >= pd.to_datetime(START_DATE)]
    return df

tickers = pd.read_csv(IN_TICKERS_CSV)["ticker"].astype(str).str.strip().unique().tolist()

failures = []
done = 0

for t in tickers:
    out = OUT_DIR / f"{t}.csv"
    if out.exists() and out.stat().st_size > 1000:
        done += 1
        continue

    try:
        dfp = fetch_eod_full(t)
        if dfp.empty:
            failures.append({"ticker": t, "reason": "empty"})
        else:
            # guarda solo columnas típicas si existen
            cols = [c for c in ["date","open","high","low","close","adjClose","volume"] if c in dfp.columns]
            dfp[cols].to_csv(out, index=False)
            done += 1
        time.sleep(SLEEP_S)

    except Exception as e:
        failures.append({"ticker": t, "reason": str(e)})
        time.sleep(SLEEP_S)

# logs
pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
print(f"Done. tickers_ok={done} failures={len(failures)}")
print(f"Saved to: {OUT_DIR.resolve()}")
