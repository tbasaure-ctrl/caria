"""Download SP500 fundamentals using FMP API + yfinance backup"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime
import time
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parents[1]
SILVER = BASE_DIR / "data/silver/fundamentals"
SILVER.mkdir(parents=True, exist_ok=True)

FMP_API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"

def log(msg):
    """Print with flush"""
    print(msg, flush=True)

log("=" * 60)
log("DOWNLOADING SP500 FUNDAMENTALS")
log("=" * 60)

# Get SP500 tickers from FMP API (same as price download)
log("\n[1/5] Fetching SP500 ticker list...")
try:
    url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={FMP_API_KEY}"
    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        data = response.json()
        tickers = [item['symbol'] for item in data]
        log(f"  Found {len(tickers)} tickers from FMP")
    else:
        raise Exception(f"FMP API returned {response.status_code}")

except Exception as e:
    log(f"  Error: {e}")
    log("  Using fallback tickers")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'UNH', 'JNJ']

# Download key metrics from FMP
log(f"\n[2/5] Downloading key metrics from FMP for {len(tickers)} tickers...")
log(f"  This will take ~{len(tickers) * 0.12 / 60:.0f} minutes...")
log("  Progress updates every 50 tickers")

all_quality = []
all_value = []
failed = []

for i, ticker in enumerate(tickers):
    if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
        log(f"  Progress: {i+1}/{len(tickers)} ({(i+1)/len(tickers)*100:.1f}%)")

    try:
        # FMP API: Key metrics (quarterly)
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=quarter&apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if len(data) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(data)
                df['ticker'] = ticker
                df['date'] = pd.to_datetime(df['date'])

                # Quality features
                quality_cols = [
                    'date', 'ticker',
                    'roic', 'roiic', 'roe', 'roa',
                    'grossProfitMargin', 'netProfitMargin',
                    'freeCashFlowYield', 'freeCashFlowPerShare',
                    'capexPerShare', 'researchAndDevelopmentToRevenue'
                ]

                # Rename some columns to match our schema
                df = df.rename(columns={
                    'roe': 'returnOnEquity',
                    'roa': 'returnOnAssets',
                    'capexPerShare': 'capitalExpenditures',
                    'researchAndDevelopmentToRevenue': 'r_and_d'
                })

                quality_data = df[['date', 'ticker'] + [c for c in quality_cols[2:] if c in df.columns or c.replace('_', '') in df.columns]].copy()
                all_quality.append(quality_data)

                # Value features from same API
                value_cols = [
                    'date', 'ticker',
                    'priceToBookRatio', 'priceToSalesRatio', 'enterpriseValue',
                    'marketCap', 'revenuePerShare', 'netIncomePerShare'
                ]

                value_data = df[['date', 'ticker'] + [c for c in value_cols[2:] if c in df.columns]].copy()
                all_value.append(value_data)

        else:
            failed.append((ticker, f"HTTP {response.status_code}"))

    except Exception as e:
        failed.append((ticker, str(e)))

    # Rate limiting
    time.sleep(0.12)  # FMP allows ~250 requests/min

log(f"\n  Downloaded from FMP: {len(all_quality)} tickers")
log(f"  Failed: {len(failed)} tickers")

# Download financial ratios from FMP (annual for growth rates)
log(f"\n[3/5] Downloading financial ratios from FMP...")

for i, ticker in enumerate(tickers):
    if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
        log(f"  Progress: {i+1}/{len(tickers)} ({(i+1)/len(tickers)*100:.1f}%)")

    try:
        # FMP API: Financial ratios (annual)
        url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?period=annual&apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if len(data) > 0:
                df = pd.DataFrame(data)
                df['ticker'] = ticker
                df['date'] = pd.to_datetime(df['date'])

                # Extract growth rates
                growth_cols = [
                    'date', 'ticker',
                    'revenueGrowth', 'netIncomeGrowth', 'operatingIncomeGrowth'
                ]

                growth_data = df[['date', 'ticker'] + [c for c in growth_cols[2:] if c in df.columns]].copy()

                # Merge with existing value data for this ticker
                if len(all_value) > 0:
                    for j, val_df in enumerate(all_value):
                        if val_df['ticker'].iloc[0] == ticker:
                            # Merge on ticker and date
                            all_value[j] = val_df.merge(growth_data, on=['date', 'ticker'], how='left')
                            break

    except Exception as e:
        pass  # Already counted in failed

    time.sleep(0.12)

# Fallback to yfinance for failed tickers
log(f"\n[4/5] Downloading from yfinance for {len(failed)} failed tickers...")

for ticker, error in failed:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get quarterly financials
        quarterly = stock.quarterly_financials
        balance = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow

        if len(quarterly.columns) > 0:
            # Create date index
            dates = quarterly.columns

            for date in dates:
                # Quality features from financials
                quality_row = {
                    'date': pd.to_datetime(date),
                    'ticker': ticker,
                    'returnOnEquity': info.get('returnOnEquity', np.nan),
                    'returnOnAssets': info.get('returnOnAssets', np.nan),
                    'grossProfitMargin': info.get('grossMargins', np.nan),
                    'netProfitMargin': info.get('profitMargins', np.nan),
                    'freeCashFlowPerShare': info.get('freeCashflow', np.nan) / info.get('sharesOutstanding', 1) if info.get('freeCashflow') else np.nan
                }

                all_quality.append(pd.DataFrame([quality_row]))

                # Value features
                value_row = {
                    'date': pd.to_datetime(date),
                    'ticker': ticker,
                    'priceToBookRatio': info.get('priceToBook', np.nan),
                    'priceToSalesRatio': info.get('priceToSalesTrailing12Months', np.nan),
                    'enterpriseValue': info.get('enterpriseValue', np.nan),
                    'marketCap': info.get('marketCap', np.nan),
                    'revenueGrowth': info.get('revenueGrowth', np.nan)
                }

                all_value.append(pd.DataFrame([value_row]))

    except Exception as e:
        log(f"  {ticker}: yfinance also failed ({str(e)[:50]})")

# Combine all data
log(f"\n[5/5] Combining and saving data...")

quality_df = pd.concat(all_quality, ignore_index=True)
value_df = pd.concat(all_value, ignore_index=True)

log(f"  Quality signals: {len(quality_df)} rows")
log(f"  Value signals: {len(value_df)} rows")

# Sort by ticker and date
quality_df = quality_df.sort_values(['ticker', 'date'])
value_df = value_df.sort_values(['ticker', 'date'])

# Add net_debt calculation if balance sheet available
if 'totalDebt' in value_df.columns and 'cashAndCashEquivalents' in value_df.columns:
    value_df['net_debt'] = value_df['totalDebt'] - value_df['cashAndCashEquivalents']

# Save
quality_path = SILVER / "quality_signals.parquet"
value_path = SILVER / "value_signals.parquet"

quality_df.to_parquet(quality_path, index=False)
value_df.to_parquet(value_path, index=False)

log(f"\n  Saved quality_signals: {quality_path}")
log(f"  Saved value_signals: {value_path}")

# Summary
log("\n" + "=" * 60)
log("SP500 FUNDAMENTALS DOWNLOAD COMPLETE")
log("=" * 60)
log(f"\nTickers: {quality_df['ticker'].nunique()} quality, {value_df['ticker'].nunique()} value")
log(f"Date range quality: {quality_df['date'].min()} to {quality_df['date'].max()}")
log(f"Date range value: {value_df['date'].min()} to {value_df['date'].max()}")
log(f"\nNext steps:")
log("  1. Re-run rebuild_gold_simple.py")
log("  2. Re-train models with 40x more tickers!")
