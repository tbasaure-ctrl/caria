"""Check data availability in silver/bronze"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

# Check silver
silver = BASE / 'data/silver'
if silver.exists():
    fundamentals = list((silver / 'fundamentals').glob('*.parquet'))
    technicals = list((silver / 'technicals').glob('*.parquet'))

    print(f"SILVER:")
    print(f"  Fundamentals: {len(fundamentals)} files")
    print(f"  Technicals: {len(technicals)} files")

    if fundamentals:
        sample = pd.read_parquet(fundamentals[0])
        print(f"  Sample fundamental: {fundamentals[0].stem}")
        print(f"    Shape: {sample.shape}")
        print(f"    Date range: {sample.index.min()} to {sample.index.max()}")

    if technicals:
        sample_tech = pd.read_parquet(technicals[0])
        print(f"  Sample technical: {technicals[0].stem}")
        print(f"    Shape: {sample_tech.shape}")

# Check bronze
bronze = BASE / 'data/bronze'
if bronze.exists():
    bronze_files = list(bronze.glob('**/*.parquet'))
    print(f"\nBRONZE:")
    print(f"  Total files: {len(bronze_files)}")

# Check if there's a ticker universe file
universe = BASE / 'data/tickers.txt'
if universe.exists():
    tickers = universe.read_text().strip().split('\n')
    print(f"\nUNIVERSE:")
    print(f"  Tickers defined: {len(tickers)}")
    print(f"  First 20: {tickers[:20]}")
