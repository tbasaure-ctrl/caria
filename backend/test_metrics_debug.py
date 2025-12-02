
import os
import sys
from openbb import obb

# Mock environment variables if needed, or rely on system env
# os.environ["FMP_API_KEY"] = "..." # Assuming it's already set in the environment

def test_metrics(ticker="AAPL"):
    print(f"Testing metrics for {ticker}...")
    try:
        # Try fetching ratios
        print("Fetching ratios...")
        ratios = obb.equity.fundamental.ratios(symbol=ticker, provider="fmp", limit=1)
        if hasattr(ratios, 'to_df'):
            print("Ratios DF:")
            print(ratios.to_df().iloc[0])
        else:
            print("Ratios result:", ratios)

        # Try fetching metrics
        print("\nFetching metrics...")
        metrics = obb.equity.fundamental.metrics(symbol=ticker, provider="fmp", limit=1)
        if hasattr(metrics, 'to_df'):
            print("Metrics DF:")
            print(metrics.to_df().iloc[0])
        else:
            print("Metrics result:", metrics)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_metrics()
