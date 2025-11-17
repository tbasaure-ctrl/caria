"""Convert FMP raw fundamentals to silver quality and value signals"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_FMP = BASE_DIR / "data/raw/fundamentals/fmp"
SILVER = BASE_DIR / "data/silver/fundamentals"
SILVER.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(msg, flush=True)

log("=" * 60)
log("CONVERTING FMP FUNDAMENTALS TO SILVER")
log("=" * 60)

# Get all merged fundamentals
merged_files = list(RAW_FMP.glob("*_fundamentals_merged.parquet"))
log(f"\nFound {len(merged_files)} tickers with merged fundamentals")

all_quality = []
all_value = []

for i, file in enumerate(merged_files):
    if (i + 1) % 20 == 0 or (i + 1) == len(merged_files):
        log(f"  Processing: {i+1}/{len(merged_files)} ({(i+1)/len(merged_files)*100:.1f}%)")

    ticker = file.stem.replace("_fundamentals_merged", "")

    try:
        df = pd.read_parquet(file)

        # Extract quality features (ONLY profitability, NO overlaps with value)
        quality_cols = [
            'date',
            'roic', 'roiic', 'returnOnEquity', 'returnOnAssets',
            'grossProfitMargin', 'netProfitMargin',
            'freeCashFlowPerShare',
            'capitalExpenditureCoverageRatio', 'researchAndDdevelopementToRevenue'
        ]

        # Map column names
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
        # Remove duplicate columns if any
        quality_data = quality_data.loc[:, ~quality_data.columns.duplicated()]
        all_quality.append(quality_data)

        # Extract value features (valuation ratios + growth, NO profitability overlaps)
        value_cols = [
            'date',
            'priceToBookRatio', 'priceToSalesRatio', 'enterpriseValue', 'marketCap',
            'freeCashFlowYield',  # Keep this for valuation
            'revenueGrowth', 'netIncomeGrowth', 'operatingIncomeGrowth',
            'totalDebt', 'cashAndCashEquivalents'
        ]

        value_data = df[['date'] + [c for c in value_cols[1:] if c in df.columns]].copy()
        value_data['ticker'] = ticker
        # Remove duplicate columns if any
        value_data = value_data.loc[:, ~value_data.columns.duplicated()]

        # Calculate net_debt if available
        if 'totalDebt' in value_data.columns and 'cashAndCashEquivalents' in value_data.columns:
            value_data['net_debt'] = value_data['totalDebt'] - value_data['cashAndCashEquivalents']

        all_value.append(value_data)

    except Exception as e:
        log(f"  Error processing {ticker}: {str(e)[:50]}")

# Combine all data
log(f"\nCombining data from {len(all_quality)} tickers...")

quality_df = pd.concat(all_quality, ignore_index=True)
value_df = pd.concat(all_value, ignore_index=True)

# Sort by ticker and date
quality_df = quality_df.sort_values(['ticker', 'date'])
value_df = value_df.sort_values(['ticker', 'date'])

# Save
quality_path = SILVER / "quality_signals.parquet"
value_path = SILVER / "value_signals.parquet"

quality_df.to_parquet(quality_path, index=False)
value_df.to_parquet(value_path, index=False)

log(f"\nSaved quality_signals: {len(quality_df):,} rows, {quality_df['ticker'].nunique()} tickers")
log(f"Saved value_signals: {len(value_df):,} rows, {value_df['ticker'].nunique()} tickers")
log(f"\nDate range quality: {quality_df['date'].min()} to {quality_df['date'].max()}")
log(f"Date range value: {value_df['date'].min()} to {value_df['date'].max()}")

# Sample
log(f"\nSample tickers: {sorted(quality_df['ticker'].unique())[:20]}")

log("\n" + "=" * 60)
log("CONVERSION COMPLETE")
log("=" * 60)
