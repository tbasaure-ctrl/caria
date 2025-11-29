import pandas as pd
df = pd.read_parquet(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/silver/macro/fred_us.parquet")
print(df.columns.tolist())
