import pandas as pd
import os

def inspect_etfs(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        df = pd.read_parquet(file_path)
        print(f"--- Inspecting {os.path.basename(file_path)} ---")
        print("Columns:", df.columns.tolist())
        if 'ticker' in df.columns:
            print("Unique Tickers:", df['ticker'].unique())
        print("Head:")
        print(df.head())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

inspect_etfs(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/silver/market/sector_etfs_daily.parquet")
