import requests
import pandas as pd
import os

API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"
BASE_URL = "https://financialmodelingprep.com/api/v3"

def fetch_data(symbol, name):
    print(f"Fetching {name} ({symbol})...")
    url = f"{BASE_URL}/historical-price-full/{symbol}?from=1900-01-01&apikey={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "historical" in data:
            df = pd.DataFrame(data["historical"])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            start_date = df['date'].iloc[0]
            end_date = df['date'].iloc[-1]
            print(f"  -> Success. Range: {start_date.date()} to {end_date.date()} ({len(df)} records)")
            return df
        else:
            print(f"  -> No 'historical' key in response for {symbol}")
            return None
    except Exception as e:
        print(f"  -> Error fetching {symbol}: {e}")
        return None

# Symbols to check
assets = {
    "^VIX": "CBOE Volatility Index",
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",
    "^RUT": "Russell 2000"
}

results = {}
output_dir = "C:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/outputs/coodination_data"
os.makedirs(output_dir, exist_ok=True)

for sym, name in assets.items():
    df = fetch_data(sym, name)
    if df is not None:
        filename = os.path.join(output_dir, f"{name.replace(' ', '_')}.csv")
        df.to_csv(filename, index=False)
        results[sym] = df

print("\nData acquisition complete.")
