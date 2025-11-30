import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

LOGGER = logging.getLogger(__name__)

class FMPDataService:
    def __init__(self):
        # Securely load the key you provided
        self.api_key = os.getenv("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.pillars = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "SPY", "QQQ"]

    def fetch_market_pulse(self, lookback_days=90):
        """
        Fetches 'Real-World' adjusted close prices for the Market Pillars.
        """
        if not self.api_key:
            LOGGER.error("FMP_API_KEY not found. The MRI cannot run without fuel.")
            raise ValueError("FMP_API_KEY not found. The MRI cannot run without fuel.")

        LOGGER.info(f"📡 CONNECTING TO FMP: Ingesting last {lookback_days} days of institutional data...")
        
        # We use the Batch Endpoint for efficiency (Professional Standard)
        # Note: Free tier might need individual loops; assuming Tier 1+ for 'Real' usage
        # Fallback to individual fetch if batch fails
        prices_dict = {}
        
        for ticker in self.pillars:
            try:
                # Calculate start date based on lookback
                start_date = (datetime.now() - timedelta(days=lookback_days + 30)).strftime('%Y-%m-%d')
                
                url = f"{self.base_url}/historical-price-full/{ticker}?from={start_date}&apikey={self.api_key}"
                response = requests.get(url, timeout=10)
                data = response.json()
                
                if "historical" in data:
                    # Create a series for this asset
                    df = pd.DataFrame(data["historical"])
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                    df.sort_index(inplace=True)
                    # Use adjusted close for total return accuracy
                    prices_dict[ticker] = df["adjClose"]
                else:
                    LOGGER.warning(f"⚠️ Warning: No data for {ticker}")
            except Exception as e:
                LOGGER.error(f"❌ Error fetching {ticker}: {e}")

        if not prices_dict:
            raise RuntimeError("Failed to fetch any market data from FMP")

        # Align all time series to the same calendar
        market_df = pd.DataFrame(prices_dict).fillna(method="ffill").dropna()
        
        if market_df.empty:
             raise RuntimeError("Market data is empty after alignment")

        # Calculate Log Returns (The standard for TDA)
        returns_df = np.log(market_df / market_df.shift(1)).dropna()
        
        return returns_df
