import requests
import pandas as pd
import os
import time

# Configuration
API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"
BASE_URL = "https://financialmodelingprep.com/api/v3"
OUTPUT_DIR = "coodination_data"

SECTORS = {
    'FixedIncome': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP', 'BND', 'AGG', 'MUB', 'MBB'],
    'Commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBB', 'DBC', 'PALL', 'PPLT']
}

def fetch_data(symbol):
    url = f"{BASE_URL}/historical-price-full/{symbol}?from=1990-01-01&apikey={API_KEY}"
    print(f"Fetching {symbol}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
            # Keep necessary columns
            cols = ['date', 'adjClose'] if 'adjClose' in df.columns else ['date', 'close']
            df = df[cols].rename(columns={cols[1]: symbol})
            
            filename = f"{symbol}.csv"
            output_path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(output_path, index=False)
            print(f"Saved {symbol} ({len(df)} records)")
        else:
            print(f"No historical data found for {symbol}")
            
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for sector, symbols in SECTORS.items():
        print(f"\n--- Fetching {sector} Constituents ---")
        for sym in symbols:
            fetch_data(sym)
            time.sleep(0.1) # Rate limit politeness
