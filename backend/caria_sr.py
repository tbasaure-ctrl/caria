
import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
from typing import Optional, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALPHA_VANTAGE_KEY = "3KHQX7KNMNT7H7MZ"

class AlphaVantageClient:
    def __init__(self, key: str = ALPHA_VANTAGE_KEY):
        self.key = key

    def download_daily(self, ticker: str, outputsize: str = "full") -> Optional[pd.Series]:
        """Download daily adjusted close prices from Alpha Vantage."""
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize={outputsize}&apikey={self.key}"
        try:
            r = requests.get(url, timeout=10)
            data = r.json()
            if "Time Series (Daily)" not in data:
                logger.warning(f"Alpha Vantage: No daily data for {ticker}. Response: {list(data.keys())}")
                return None
            
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            # '5. adjusted close' is standard for TIME_SERIES_DAILY_ADJUSTED
            series = df["5. adjusted close"].astype(float)
            series.index = pd.to_datetime(series.index)
            series = series.sort_index()
            return series
        except Exception as e:
            logger.error(f"Alpha Vantage error for {ticker}: {e}")
            return None

def download_data_with_fallback(ticker: str, start: str = "2007-01-01") -> pd.Series:
    """Try Alpha Vantage first, then Yahoo Finance."""
    av = AlphaVantageClient()
    series = av.download_daily(ticker)
    
    if series is not None and not series.empty:
        # Filter by start date
        series = series[series.index >= start]
        if not series.empty:
             logger.info(f"Loaded {ticker} from Alpha Vantage")
             return series

    logger.info(f"Fallback to Yahoo Finance for {ticker}")
    # yfinance download
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
         return df["Close"].dropna()
    elif isinstance(df, pd.Series) and not df.empty:
         return df.dropna()
         
    return pd.Series(dtype=float)

def download_credit_series(start="2007-01-01"):
    """Download HYG and calculate credit volatility."""
    # HYG often simpler via Yahoo for long history consistency, but let's try fallback logic
    hyg = download_data_with_fallback("HYG", start=start)
    
    if hyg.empty:
        return None
        
    ret_hyg = hyg.pct_change()
    if isinstance(ret_hyg, pd.DataFrame): # Handle potential multi-col if yfinance returns weird shape
        ret_hyg = ret_hyg.iloc[:, 0]
        
    vol_credit = (ret_hyg.rolling(42).std() * np.sqrt(252)).rename("vol_credit")
    return vol_credit

def compute_sr_for_ticker(ticker: str, vol_credit: pd.Series, start="2007-01-01") -> Optional[Tuple[pd.DataFrame, Dict[str, float]]]:
    """Compute CARIA-SR metrics for a given ticker."""
    px = download_data_with_fallback(ticker, start=start)
    if px.empty:
        return None
        
    px = px.squeeze() # Ensure Series
    ret = px.pct_change().dropna()
    if ret.empty:
        return None

    px.name = "price"
    ret.name = "ret"

    # Align credit vol
    vol_credit_aligned = vol_credit.reindex(ret.index).ffill().dropna()
    px_aligned = px.reindex(ret.index)

    df = pd.DataFrame({
        "price": px_aligned,
        "ret": ret,
        "vol_credit": vol_credit_aligned
    }).dropna()

    if len(df) < 300: # Need enough history
        return None

    # ROC + mom_norm + sync
    roc = df["ret"].rolling(21).sum()
    # Handle potential zeros/nans in expanding window if needed, but rolling(252) handles it
    mom_norm = (roc - roc.rolling(252).mean()) / roc.rolling(252).std()
    
    # Sync: Correlation between momentum and credit volatility
    # If mom is high (good returns) and vol_credit is low (calm), sync is usually negative? 
    # Logic from user: ((mom_norm.corr(vol_credit) + 1) / 2) -> 0..1
    # Note: user used rolling(21).corr
    sync = ((mom_norm.rolling(21).corr(df["vol_credit"]) + 1) / 2).rename("sync")

    # E4 (Volatility Mix)
    vol5  = df["ret"].rolling(5).std() * np.sqrt(252)
    vol21 = df["ret"].rolling(21).std() * np.sqrt(252)
    vol63 = df["ret"].rolling(63).std() * np.sqrt(252)
    e4 = (0.20*vol5 + 0.30*vol21 + 0.25*vol63 + 0.25*df["vol_credit"]).rename("E4")

    # SR (Systemic Risk Score)
    # Rank percentile of E4 * adjusted sync
    sr = ((e4.rank(pct=True) * (1 + sync))).rank(pct=True).rename("SR")

    df = pd.concat([df, sync, e4, sr], axis=1).dropna()

    # Regime Definition
    q_sync = df["sync"].quantile(0.80)
    q_e4   = df["E4"].quantile(0.80)
    df["regime"] = ((df["sync"] > q_sync) & (df["E4"] > q_e4)).astype(int)

    # Future Return (Target for validation)
    future10 = df["ret"].rolling(10).sum().shift(-10).rename("future_ret_10d")
    df = pd.concat([df, future10], axis=1) # Don't dropna yet to keep recent rows for current status

    # Metrics Aggregation (on rows where we have future returns)
    valid_df = df.dropna(subset=["future_ret_10d", "regime", "SR"])
    
    auc = 0.5
    m0 = 0.0
    m1 = 0.0
    
    if len(valid_df) > 50 and len(valid_df["regime"].unique()) > 1:
        try:
            auc = roc_auc_score(valid_df["regime"], valid_df["SR"])
            m0 = valid_df[valid_df["regime"] == 0]["future_ret_10d"].mean()
            m1 = valid_df[valid_df["regime"] == 1]["future_ret_10d"].mean()
        except ValueError:
            pass # Use defaults if AUC fails

    stats = {
        "auc": float(auc) if not pd.isna(auc) else 0.5, 
        "mean_normal": float(m0) if not pd.isna(m0) else 0.0, 
        "mean_fragile": float(m1) if not pd.isna(m1) else 0.0
    }

    return df, stats
