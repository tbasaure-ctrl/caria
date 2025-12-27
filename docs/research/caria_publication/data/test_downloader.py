"""
Test script for FMP API connection and data download
"""
import pandas as pd
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append('.')

from download_sp500_prices_fmp import fetch_eod_full, FMP_API_KEY

def test_api_connection():
    """Test FMP API connection with a few tickers."""
    print("=" * 50)
    print("Testing FMP API Connection")
    print("=" * 50)
    print(f"API Key configured: {'Yes' if FMP_API_KEY else 'No'}")
    print()

    # Test with a few well-known tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']

    results = []

    for ticker in test_tickers:
        print(f"Testing {ticker}...", end=" ")
        try:
            df = fetch_eod_full(ticker)
            if not df.empty:
                date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
                results.append({
                    'ticker': ticker,
                    'status': 'success',
                    'rows': len(df),
                    'date_range': date_range
                })
                print(f"[OK] {len(df)} rows, {date_range}")
            else:
                results.append({
                    'ticker': ticker,
                    'status': 'empty',
                    'rows': 0,
                    'date_range': None
                })
                print("[EMPTY] Empty data")
        except Exception as e:
            results.append({
                'ticker': ticker,
                'status': 'error',
                'rows': 0,
                'date_range': str(e)[:100]
            })
            print(f"[ERROR] Error: {str(e)[:50]}")

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"Successful downloads: {success_count}/{len(test_tickers)}")

    if success_count > 0:
        print("\nSample data structure for AAPL:")
        aapl_result = next((r for r in results if r['ticker'] == 'AAPL' and r['status'] == 'success'), None)
        if aapl_result:
            df = fetch_eod_full('AAPL')
            print(f"Columns: {df.columns.tolist()}")
            print(f"Sample rows:")
            print(df.head(3))

    return success_count > 0

if __name__ == "__main__":
    success = test_api_connection()
    if success:
        print("\n[SUCCESS] API connection test passed!")
    else:
        print("\n[FAILED] API connection test failed!")
        sys.exit(1)










