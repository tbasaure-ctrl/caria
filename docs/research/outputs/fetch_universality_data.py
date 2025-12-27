import requests
import pandas as pd
import os

# Configuration
API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"
BASE_URL = "https://financialmodelingprep.com/api/v3"
OUTPUT_DIR = "coodination_data"

ASSETS = {
    'GLD': 'Gold',
    'USO': 'Oil',
    'TLT': 'Treasuries_20Y',
    'EURUSD': 'Euro_USD'
}

def fetch_data(symbol, name):
    # FMP Endpoint handles both Stocks/ETFs and Forex with the same structure usually
    # But strictly: Forex might ideally use /historical-price-full/EURUSD
    url = f"{BASE_URL}/historical-price-full/{symbol}?from=2000-01-01&apikey={API_KEY}"
    print(f"Fetching {symbol} ({name})...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
            df = df[['date', 'close', 'adjClose', 'volume']] if 'adjClose' in df.columns else df[['date', 'close', 'volume']]
            
            filename = f"{name}.csv"
            output_path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(output_path, index=False)
            print(f"Saved {symbol} to {output_path} ({len(df)} records)")
        else:
            print(f"No historical data found for {symbol}")
            
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for symbol, name in ASSETS.items():
        fetch_data(symbol, name)
