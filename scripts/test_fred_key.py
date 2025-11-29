from fredapi import Fred
import pandas as pd

API_KEY = "4b90ca15ff28cfec137179c22fd8246d"

def test_fred():
    try:
        fred = Fred(api_key=API_KEY)
        print("Connected to FRED.")
        
        series = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'T10Y2Y']
        for s in series:
            try:
                data = fred.get_series(s, observation_start='2020-01-01')
                print(f"Fetched {s}: {len(data)} records. Last val: {data.iloc[-1]}")
            except Exception as e:
                print(f"Error fetching {s}: {e}")
                
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_fred()
