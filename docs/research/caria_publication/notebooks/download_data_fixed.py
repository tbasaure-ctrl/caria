"""
Fixed Data Download Script for Caria Publication
=================================================

Run this script to download data with proper error handling.
Copy the resulting data_daily dictionary to the notebook.

Usage:
    python download_data_fixed.py
    
Or copy this code into a notebook cell.
"""

import time
import pandas as pd

try:
    import yfinance as yf
    print(f"yfinance version: {yf.__version__}")
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.run(["pip", "install", "yfinance"])
    import yfinance as yf

# ==============================================================================
# ASSETS WITH CORRECT INCEPTION DATES
# ==============================================================================

ASSETS_WITH_DATES = {
    # Equity ETFs
    'SPY': '1993-01-29',   # S&P 500 - oldest ETF
    'QQQ': '1999-03-10',   # NASDAQ 100
    'IWM': '2000-05-26',   # Russell 2000
    
    # Bond ETFs
    'TLT': '2002-07-30',   # 20+ Year Treasury
    'IEF': '2002-07-30',   # 7-10 Year Treasury
    'HYG': '2007-04-11',   # High Yield Corporate
    
    # Commodity ETFs
    'GLD': '2004-11-18',   # Gold
    'USO': '2006-04-10',   # Oil
    'DBC': '2006-02-06',   # Commodities Index
    
    # Crypto
    'BTC-USD': '2014-09-17',
    'ETH-USD': '2017-11-09',
    
    # International
    'EFA': '2001-08-27',   # EAFE (Developed Markets ex-US)
    'EEM': '2003-04-14',   # Emerging Markets
    'FXI': '2004-10-08',   # China Large-Cap
}

# ==============================================================================
# DOWNLOAD FUNCTION WITH RETRY LOGIC
# ==============================================================================

def download_with_retry(ticker, start_date, end_date='2025-12-31', max_retries=3):
    """Download data with retry logic and proper error handling."""
    
    for attempt in range(max_retries):
        try:
            # Download with explicit parameters
            df = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                progress=False,
                auto_adjust=True,  # Use adjusted prices
                actions=False,      # Don't download dividends/splits
                timeout=15
            )
            
            if df is not None and len(df) > 100:
                # Handle both single and multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    return df['Close'][ticker] if 'Close' in df.columns.get_level_values(0) else df.iloc[:, 0]
                elif 'Close' in df.columns:
                    return df['Close']
                elif 'Adj Close' in df.columns:
                    return df['Adj Close']
                else:
                    return df.iloc[:, 0]
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} for {ticker}: {str(e)[:50]}")
                time.sleep(2 * (attempt + 1))  # Exponential backoff
            else:
                print(f"    Failed after {max_retries} attempts: {str(e)[:50]}")
    
    return None


# ==============================================================================
# MAIN DOWNLOAD
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Downloading data for Caria Publication")
    print("=" * 60)
    print()
    
    data_daily = {}
    
    for ticker, inception in ASSETS_WITH_DATES.items():
        # Use inception date or 2000, whichever is later
        start_date = max(inception, '2000-01-01')
        
        print(f"Downloading {ticker} (from {start_date})...", end=" ")
        
        prices = download_with_retry(ticker, start_date)
        
        if prices is not None and len(prices) > 100:
            data_daily[ticker] = prices
            print(f"✓ {len(prices)} days")
        else:
            print(f"✗ Failed or insufficient data")
    
    print()
    print("=" * 60)
    print(f"Successfully downloaded: {len(data_daily)}/{len(ASSETS_WITH_DATES)} assets")
    print("=" * 60)
    
    if len(data_daily) > 0:
        print("\nAvailable tickers:", list(data_daily.keys()))
        print("\nDate ranges:")
        for ticker, prices in data_daily.items():
            print(f"  {ticker}: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # ==============================================================================
    # ALTERNATIVE: Use OpenBB if yfinance fails
    # ==============================================================================
    
    if len(data_daily) == 0:
        print("\n" + "=" * 60)
        print("yfinance failed. Trying OpenBB as alternative...")
        print("=" * 60)
        
        try:
            from openbb import obb
            
            for ticker in ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'BTC-USD']:
                try:
                    # Clean ticker for OpenBB
                    clean_ticker = ticker.replace('-USD', '')
                    
                    if 'BTC' in ticker or 'ETH' in ticker:
                        # Crypto
                        result = obb.crypto.price.historical(clean_ticker, provider='yfinance')
                    else:
                        # Equity
                        result = obb.equity.price.historical(clean_ticker, start_date='2000-01-01')
                    
                    if result is not None:
                        df = result.to_df()
                        if len(df) > 100:
                            data_daily[ticker] = df['close']
                            print(f"  ✓ {ticker}: {len(df)} days via OpenBB")
                            
                except Exception as e:
                    print(f"  ✗ {ticker}: {e}")
                    
        except ImportError:
            print("OpenBB not available. Please install: pip install openbb")
    
    # Save to pickle for later use
    if len(data_daily) > 0:
        import pickle
        with open('../data/prices_daily.pkl', 'wb') as f:
            pickle.dump(data_daily, f)
        print("\nData saved to: ../data/prices_daily.pkl")
