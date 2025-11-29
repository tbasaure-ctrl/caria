import pandas as pd
import os

path = r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/data/silver/fundamentals/quality_signals.parquet"
try:
    df = pd.read_parquet(path)
    print(f"Columns in {path}:")
    print(df.columns.tolist())
except Exception as e:
    print(f"Error: {e}")
