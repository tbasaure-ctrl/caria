import requests
import pandas as pd
import os

# Configuration
API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"
BASE_URL = "https://financialmodelingprep.com/api/v3"
OUTPUT_DIR = "coodination_data"

def fetch_data(symbol, filename):
    url = f"{BASE_URL}/historical-price-full/{symbol}?from=2010-01-01&apikey={API_KEY}"
    print(f"Fetching {symbol}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
            # Save relevant columns
            df = df[['date', 'close', 'volume']]
            
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
        
    fetch_data("BTCUSD", "BTC_USD.csv")
