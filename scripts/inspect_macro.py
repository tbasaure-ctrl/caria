import pandas as pd
import os

def inspect_parquet(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        df = pd.read_parquet(file_path)
        print(f"--- Inspecting {os.path.basename(file_path)} ---")
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print("Head:")
        print(df.head())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

inspect_parquet(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/silver/macro/fred_us.parquet")
