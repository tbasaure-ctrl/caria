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
        print("Date range:", df['date'].min(), "to", df['date'].max()) if 'date' in df.columns else None
        print("\n")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Check quality signals
inspect_parquet(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/data/silver/fundamentals/quality_signals.parquet")

# Check for any macro data in likely places
# (User didn't provide a path, so I'll just check the quality one for now to see if we have partial real data)
