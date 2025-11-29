import pandas as pd
import os

def inspect_fred(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(f"Columns in {file_path}:")
        print(df.columns.tolist())
        
        # Check for specific liquidity cols
        target_cols = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'T10Y2Y', 'DGS10', 'DGS2']
        found = [c for c in target_cols if c in df.columns]
        missing = [c for c in target_cols if c not in df.columns]
        
        print(f"Found columns: {found}")
        print(f"Missing columns: {missing}")
        
    except Exception as e:
        print(f"Error: {e}")

inspect_fred(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/silver/macro/fred_us.parquet")
