import pandas as pd
import os

def inspect_prices(file_path):
    try:
        df = pd.read_parquet(file_path)
        print("Columns:", df.columns.tolist())
    except Exception as e:
        print(f"Error: {e}")

inspect_prices(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/silver/market/stock_prices_daily.parquet")
