#!/usr/bin/env python3
"""
Generate Missing Data Files for Production
===========================================
This script helps generate the missing quality_signals.parquet and value_signals.parquet files.

Usage:
    1. Set your FMP_API_KEY environment variable OR edit the API key below
    2. Run: python generate_missing_data.py
    3. Wait for the download to complete (may take 30+ minutes for full S&P 500)
    
Alternative (Quick Start with Mock Data):
    python generate_missing_data.py --mock
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "silver" / "fundamentals"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Check for existing raw fundamentals in parent notebooks directory
# Current file is in: notebooks/caria/generate_missing_data.py
# We need to go to: notebooks/data/raw/fundamentals/fmp
NOTEBOOKS_DIR = BASE_DIR.parent  # Go up one level from caria to notebooks
RAW_FMP = NOTEBOOKS_DIR / "data" / "raw" / "fundamentals" / "fmp"

# Get API key from environment or set it here
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

def log(msg):
    """Print with timestamp"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def generate_mock_data():
    """Generate mock data for testing/development"""
    log("=" * 60)
    log("GENERATING MOCK DATA FOR TESTING")
    log("=" * 60)
    
    # Use a subset of popular tickers
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 
        'V', 'UNH', 'JNJ', 'WMT', 'JPM', 'MA', 'XOM', 'PG', 'HD', 'CVX', 
        'LLY', 'ABBV', 'MRK', 'KO', 'AVGO', 'PEP', 'COST'
    ]
    
    # Generate quarterly data for last 2 years
    dates = pd.date_range(end=datetime.now(), periods=8, freq='Q')
    
    quality_data = []
    value_data = []
    
    for ticker in tickers:
        for date in dates:
            # Quality signals (profitability metrics)
            quality_row = {
                'date': date,
                'ticker': ticker,
                'roic': np.random.uniform(0.05, 0.25),
                'roiic': np.random.uniform(0.08, 0.30),
                'returnOnEquity': np.random.uniform(0.10, 0.35),
                'returnOnAssets': np.random.uniform(0.05, 0.20),
                'grossProfitMargin': np.random.uniform(0.20, 0.70),
                'netProfitMargin': np.random.uniform(0.05, 0.35),
                'freeCashFlowPerShare': np.random.uniform(1.0, 15.0),
                'freeCashFlowYield': np.random.uniform(0.02, 0.08),
                'capitalExpenditures': np.random.uniform(0.5, 5.0),
                'r_and_d': np.random.uniform(0.02, 0.20),
            }
            quality_data.append(quality_row)
            
            # Value signals (valuation + growth metrics)
            value_row = {
                'date': date,
                'ticker': ticker,
                'priceToBookRatio': np.random.uniform(2.0, 15.0),
                'priceToSalesRatio': np.random.uniform(1.5, 10.0),
                'enterpriseValue': np.random.uniform(1e11, 3e12),
                'marketCap': np.random.uniform(1e11, 3e12),
                'revenueGrowth': np.random.uniform(-0.05, 0.30),
                'netIncomeGrowth': np.random.uniform(-0.10, 0.40),
                'operatingIncomeGrowth': np.random.uniform(-0.08, 0.35),
                'totalDebt': np.random.uniform(1e10, 1e11),
                'cashAndCashEquivalents': np.random.uniform(1e10, 2e11),
            }
            value_row['net_debt'] = value_row['totalDebt'] - value_row['cashAndCashEquivalents']
            value_data.append(value_row)
    
    # Create DataFrames
    quality_df = pd.DataFrame(quality_data)
    value_df = pd.DataFrame(value_data)
    
    # Save
    quality_path = DATA_DIR / "quality_signals.parquet"
    value_path = DATA_DIR / "value_signals.parquet"
    
    quality_df.to_parquet(quality_path, index=False)
    value_df.to_parquet(value_path, index=False)
    
    log(f"\n✅ MOCK DATA GENERATED SUCCESSFULLY")
    log(f"   Quality signals: {len(quality_df):,} rows ({quality_df['ticker'].nunique()} tickers)")
    log(f"   Value signals: {len(value_df):,} rows ({value_df['ticker'].nunique()} tickers)")
    log(f"   Date range: {quality_df['date'].min()} to {quality_df['date'].max()}")
    log(f"\n📁 Files saved to:")
    log(f"   {quality_path}")
    log(f"   {value_path}")
    log("\n⚠️  NOTE: This is MOCK DATA for testing. For production, use real data.")
    
def convert_existing_data():
    """Convert existing FMP fundamentals to silver format"""
    log("=" * 60)
    log("CONVERTING EXISTING FMP FUNDAMENTALS")
    log("=" * 60)
    
    if not RAW_FMP.exists():
        log(f"\n❌ ERROR: Raw FMP data not found at {RAW_FMP}")
        log("\n   Checked location: C:\\key\\wise_adviser_cursor_context\\notebooks\\data\\raw\\fundamentals\\fmp")
        log("\n   Falling back to download mode...")
        return False
    
    # Find merged fundamentals files
    merged_files = list(RAW_FMP.glob("*_fundamentals_merged.parquet"))
    
    if not merged_files:
        log(f"\n❌ No fundamentals files found in {RAW_FMP}")
        return False
    
    log(f"\n✅ Found {len(merged_files)} tickers with fundamentals data")
    log(f"   Processing all files...\n")
    
    all_quality = []
    all_value = []
    
    for i, file in enumerate(merged_files):
        if (i + 1) % 20 == 0 or (i + 1) == len(merged_files):
            log(f"   Processing: {i+1}/{len(merged_files)} ({(i+1)/len(merged_files)*100:.1f}%)")
        
        ticker = file.stem.replace("_fundamentals_merged", "")
        
        try:
            df = pd.read_parquet(file)
            
            # Extract quality features (profitability, NO overlaps with value)
            quality_cols = [
                'date',
                'roic', 'roiic', 'returnOnEquity', 'returnOnAssets',
                'grossProfitMargin', 'netProfitMargin',
                'freeCashFlowPerShare', 'freeCashFlowYield',
                'capitalExpenditureCoverageRatio', 'researchAndDdevelopementToRevenue'
            ]
            
            # Map column names from FMP format
            rename_map = {
                'returnOnInvestedCapital': 'roic',
                'roe': 'returnOnEquity',
                'roa': 'returnOnAssets',
                'researchAndDdevelopementToRevenue': 'r_and_d',
                'capitalExpenditureCoverageRatio': 'capitalExpenditures'
            }
            
            df = df.rename(columns=rename_map)
            
            quality_data = df[['date'] + [c for c in quality_cols[1:] if c in df.columns]].copy()
            quality_data['ticker'] = ticker
            quality_data = quality_data.loc[:, ~quality_data.columns.duplicated()]
            all_quality.append(quality_data)
            
            # Extract value features (valuation + growth)
            value_cols = [
                'date',
                'priceToBookRatio', 'priceToSalesRatio', 'enterpriseValue', 'marketCap',
                'revenueGrowth', 'netIncomeGrowth', 'operatingIncomeGrowth',
                'totalDebt', 'cashAndCashEquivalents'
            ]
            
            value_data = df[['date'] + [c for c in value_cols[1:] if c in df.columns]].copy()
            value_data['ticker'] = ticker
            value_data = value_data.loc[:, ~value_data.columns.duplicated()]
            
            # Calculate net_debt if available
            if 'totalDebt' in value_data.columns and 'cashAndCashEquivalents' in value_data.columns:
                value_data['net_debt'] = value_data['totalDebt'] - value_data['cashAndCashEquivalents']
            
            all_value.append(value_data)
            
        except Exception as e:
            log(f"   ⚠️  Error processing {ticker}: {str(e)[:50]}")
    
    # Combine all data
    log(f"\n   Combining data from {len(all_quality)} tickers...")
    
    quality_df = pd.concat(all_quality, ignore_index=True)
    value_df = pd.concat(all_value, ignore_index=True)
    
    # Sort by ticker and date
    quality_df = quality_df.sort_values(['ticker', 'date'])
    value_df = value_df.sort_values(['ticker', 'date'])
    
    # Save
    quality_path = DATA_DIR / "quality_signals.parquet"
    value_path = DATA_DIR / "value_signals.parquet"
    
    quality_df.to_parquet(quality_path, index=False)
    value_df.to_parquet(value_path, index=False)
    
    log(f"\n✅ CONVERSION COMPLETED SUCCESSFULLY")
    log(f"   Quality signals: {len(quality_df):,} rows ({quality_df['ticker'].nunique()} tickers)")
    log(f"   Value signals: {len(value_df):,} rows ({value_df['ticker'].nunique()} tickers)")
    log(f"   Date range: {quality_df['date'].min()} to {quality_df['date'].max()}")
    log(f"\n📁 Files saved to:")
    log(f"   {quality_path}")
    log(f"   {value_path}")
    log(f"\n   Sample tickers: {', '.join(sorted(quality_df['ticker'].unique())[:10])}")
    
    return True
    
def download_real_data():
    """Download real fundamentals data from FMP"""
    if not FMP_API_KEY:
        log("❌ ERROR: FMP_API_KEY not set!")
        log("\nPlease set the FMP_API_KEY environment variable:")
        log("   Windows: $env:FMP_API_KEY='your-api-key-here'")
        log("   Linux/Mac: export FMP_API_KEY='your-api-key-here'")
        log("\nOr edit this script and add your API key at the top.")
        log("\nAlternatively, use --mock flag to generate test data:")
        log("   python generate_missing_data.py --mock")
        sys.exit(1)
    
    log("=" * 60)
    log("DOWNLOADING REAL FUNDAMENTALS DATA")
    log("=" * 60)
    log(f"\n⏱️  This may take 30-60 minutes for full S&P 500...")
    log("   You can cancel anytime with Ctrl+C\n")
    
    # Import the actual download script
    script_path = BASE_DIR / "caria_data" / "scripts" / "download_sp500_fundamentals.py"
    
    if not script_path.exists():
        log(f"❌ ERROR: Download script not found at {script_path}")
        sys.exit(1)
    
    # Update the script's API key
    import subprocess
    env = os.environ.copy()
    env['FMP_API_KEY'] = FMP_API_KEY
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        cwd=str(BASE_DIR)
    )
    
    if result.returncode == 0:
        log("\n✅ REAL DATA DOWNLOAD COMPLETED SUCCESSFULLY")
    else:
        log("\n❌ ERROR during download. Check the output above.")
        sys.exit(1)

def verify_files():
    """Verify all required files exist"""
    log("\n" + "=" * 60)
    log("VERIFYING REQUIRED FILES")
    log("=" * 60)
    
    required_files = {
        "Quality Signals": DATA_DIR / "quality_signals.parquet",
        "Value Signals": DATA_DIR / "value_signals.parquet",
        "Historical Events": BASE_DIR / "caria_data" / "raw" / "wisdom" / "historical_events_wisdom.jsonl",
        "Regime Model": BASE_DIR / "caria_data" / "models" / "regime_hmm_model.pkl",
    }
    
    all_present = True
    for name, path in required_files.items():
        if path.exists():
            size = path.stat().st_size
            log(f"✅ {name}: {path.relative_to(BASE_DIR)} ({size:,} bytes)")
        else:
            log(f"❌ {name}: MISSING - {path.relative_to(BASE_DIR)}")
            all_present = False
    
    log("\n" + "=" * 60)
    if all_present:
        log("✅ ALL REQUIRED FILES PRESENT")
        log("\nNext steps:")
        log("   1. Add files to git: git add data/silver/fundamentals/*.parquet caria_data/raw/wisdom/*.jsonl")
        log("   2. Commit: git commit -m 'Add production data files'")
        log("   3. Push to main: git push origin main")
    else:
        log("⚠️  SOME FILES ARE MISSING")
        log("\nRun this script with --mock or without flags to generate them.")
    log("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Generate missing production data files")
    parser.add_argument("--mock", action="store_true", help="Generate mock data for testing")
    parser.add_argument("--verify", action="store_true", help="Only verify files exist")
    parser.add_argument("--download", action="store_true", help="Force download new data instead of converting existing")
    args = parser.parse_args()
    
    if args.verify:
        verify_files()
    elif args.mock:
        generate_mock_data()
        verify_files()
    elif args.download:
        download_real_data()
        verify_files()
    else:
        # Default: Try to convert existing data first
        success = convert_existing_data()
        if not success:
            log("\n⚠️  Existing data not found. Attempting download...")
            download_real_data()
        verify_files()

if __name__ == "__main__":
    main()
