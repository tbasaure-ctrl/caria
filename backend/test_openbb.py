import sys
import os
from pathlib import Path
from openbb import obb

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from api.services.openbb_client import OpenBBClient

import os

def test():
    # Set FMP Key if available
    fmp_key = os.getenv("FMP_API_KEY")
    if fmp_key:
        print("Setting FMP Key...")
        obb.user.credentials.fmp_api_key = fmp_key

    client = OpenBBClient()
    print("Fetching AAPL data via get_ticker_data...")
    
    data = client.get_ticker_data("AAPL")
    print("Keys:", data.keys())
    print("Current Price:", data.get("current_price"))
    
    print("\nDebugging Quote...")
    try:
        quote = obb.equity.price.quote(symbol="AAPL", provider="fmp")
        if hasattr(quote, 'to_df'):
            df = quote.to_df()
            print("Quote Columns:", df.columns.tolist())
            print("Quote Row:", df.iloc[0].to_dict())
    except Exception as e:
        print("Quote Debug Error:", e)

if __name__ == "__main__":
    test()
