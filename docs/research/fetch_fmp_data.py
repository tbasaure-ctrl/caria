
import os
import requests
import pandas as pd
import numpy as np

# API Key
API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"
BASE_URL = "https://financialmodelingprep.com/api/v3"

# Global Macro Universe (Proxies)
# Ideally we would use 35 years of data, but ETFs have shorter history.
# We will use Indices where possible, or long-running ETFs.
ASSETS = {
    # US Equities
    'SPY': 'SP500', 'QQQ': 'Nasdaq', 'IWM': 'Russell2000',
    'XLE': 'Energy', 'XLF': 'Financials', 'XLV': 'Health', 'XLK': 'Tech', 
    'XLU': 'Utilities', 'XLI': 'Industrial', 'XLP': 'Staples', 'XLB': 'Materials', 'XLY': 'Discretionary',
    
    # Global Equities
    'EFA': 'EAFE', 'EEM': 'Emerging', 'EWJ': 'Japan', 'EWZ': 'Brazil', 'FXI': 'China', 
    'EWG': 'Germany', 'EWU': 'UK', 'EWC': 'Canada', 'EWY': 'SouthKorea', 'EWA': 'Australia',
    
    # Credit & Rates (Using ETFs for consistency, note history limits)
    'LQD': 'InvGrade', 'HYG': 'HighYield', 'TLT': 'LongTreasury', 'IEF': 'MidTreasury', 
    'SHY': 'ShortTreasury', 'AGG': 'AggBond', 'MUB': 'Muni', 'TIP': 'TIPS',
    
    # Commodities
    'GLD': 'Gold', 'SLV': 'Silver', 'USO': 'Oil', 'DBA': 'Agriculture', 'UNG': 'NatGas',
    
    # Currencies (FMP Tickers often different, trying standard pairs)
    # Note: FMP Forex tickers are typically 'EURUSD'.
    # We will try to fetch some if possible, but might fail if tickers vary.
    # Sticking to ETFs for FX proxy might be safer: UUP (Dollar), FXE (Euro), FXY (Yen).
    'UUP': 'USD', 'FXE': 'Euro', 'FXY': 'Yen'
}

OUTPUT_DIR = "c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/media"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def fetch_price_history(symbol, start_date="1990-01-01"):
    """Fetches daily adjusted close prices from FMP."""
    url = f"{BASE_URL}/historical-price-full/{symbol}?from={start_date}&apikey={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        
        if "historical" not in data:
            print(f"Warning: No data for {symbol}")
            return None
            
        df = pd.DataFrame(data["historical"])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df['adjClose']
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def main():
    print("Fetching Global Macro Data from FMP...")
    combined_df = pd.DataFrame()
    
    for symbol, name in ASSETS.items():
        print(f"Fetching {symbol} ({name})...")
        series = fetch_price_history(symbol)
        if series is not None:
            combined_df[symbol] = series
            
    # Clean Data
    # Drop rows where everything is NaN (holidays)
    combined_df.dropna(how='all', inplace=True)
    
    # Fill gaps (forward fill limit 5 days)
    combined_df.fillna(method='ffill', limit=5, inplace=True)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'Global_Macro_Prices_FMP.csv')
    combined_df.to_csv(output_path)
    print(f"Saved {combined_df.shape[1]} assets to {output_path}")
    print(f"Date Range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    # Quick Stat check
    print("\nMissing Data Check (First Valid Index):")
    print(combined_df.apply(lambda x: x.first_valid_index()))

if __name__ == "__main__":
    main()
