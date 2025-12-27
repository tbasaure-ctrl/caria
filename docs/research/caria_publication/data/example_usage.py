"""
Example: How to load and use the downloaded S&P 500 data
========================================================

This script shows how to load the data downloaded by download_sp500_prices_fmp.py
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_sp500_data(data_dir="./sp500_prices_alpha"):
    """Load all S&P 500 stock data into a dictionary of DataFrames."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all CSV files (excluding failures.csv)
    csv_files = list(data_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != "failures.csv"]

    print(f"Found {len(csv_files)} stock data files")

    data = {}
    for csv_file in csv_files:
        ticker = csv_file.stem  # Remove .csv extension
        try:
            df = pd.read_csv(csv_file)
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                data[ticker] = df
        except Exception as e:
            print(f"Error loading {ticker}: {e}")

    return data

def analyze_data_summary(data):
    """Print summary statistics of the loaded data."""
    print("\n" + "="*60)
    print("S&P 500 DATA SUMMARY")
    print("="*60)

    if not data:
        print("No data loaded!")
        return

    # Basic stats
    total_stocks = len(data)
    dates_per_stock = [len(df) for df in data.values()]
    avg_dates = sum(dates_per_stock) / len(dates_per_stock)

    print(f"Total stocks loaded: {total_stocks}")
    print(f"Average data points per stock: {avg_dates:.0f}")

    # Date range
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    print(f"Date range: {all_dates[0].date()} to {all_dates[-1].date()}")
    print(f"Total trading days: {len(all_dates)}")

    # Show sample stocks
    sample_stocks = list(data.keys())[:5]
    print(f"\nSample stocks: {', '.join(sample_stocks)}")

    # Show columns available
    if data:
        sample_df = next(iter(data.values()))
        print(f"Available columns: {list(sample_df.columns)}")

def plot_price_example(data, ticker="AAPL"):
    """Plot price data for a sample ticker."""
    if ticker not in data:
        print(f"Ticker {ticker} not found in data")
        return

    df = data[ticker]

    # Plot closing prices (last 2 years for visibility)
    plt.figure(figsize=(12, 6))
    recent_data = df[df.index >= pd.Timestamp('2023-01-01')]

    if 'close' in df.columns:
        plt.plot(recent_data.index, recent_data['close'], label=f'{ticker} Close Price')
        plt.title(f'{ticker} Stock Price (2023-2025)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No 'close' column found for {ticker}")

if __name__ == "__main__":
    print("Loading S&P 500 data...")

    try:
        # Load data
        sp500_data = load_sp500_data()

        # Show summary
        analyze_data_summary(sp500_data)

        # Plot example
        if sp500_data:
            plot_price_example(sp500_data, "AAPL")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo download the data, run:")
        print("python download_sp500_prices_alpha.py")
    except Exception as e:
        print(f"Unexpected error: {e}")










