import pandas as pd
import os

def inspect_value(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(f"Columns in {file_path}:")
        print(df.columns.tolist())
        print("Head:")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")

inspect_value(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/data/silver/fundamentals/value_signals.parquet")
