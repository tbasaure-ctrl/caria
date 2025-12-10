# Data Sources and Download Instructions

## Overview

This directory contains instructions for obtaining the data required to replicate
the results in:

> **"Entropic Resonance and Volatility Compression as Precursors to Systemic Failure"**
> Basaure, T. (2025)

## Data Sources

### 1. Yahoo Finance (via yfinance)

Primary source for price data. Automatically downloaded by the replication notebook.

**Assets:**
- Equity: SPY, QQQ, IWM
- Bonds: TLT, IEF, HYG
- Commodities: GLD, USO, DBC
- Crypto: BTC-USD, ETH-USD
- International: EFA, EEM, FXI

**Time Period:** 1990-01-01 to 2025-12-31 (where available)

**Frequency:** Daily (close prices)

### 2. Manual Download (Optional)

For users without internet access or for archival purposes:

```python
import yfinance as yf
import pandas as pd

# Download all assets
tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'IEF', 'HYG', 
           'GLD', 'USO', 'DBC', 'BTC-USD', 'ETH-USD',
           'EFA', 'EEM', 'FXI']

data = yf.download(tickers, start='1990-01-01', end='2025-12-31')

# Save to CSV
data['Adj Close'].to_csv('data/prices_daily.csv')
```

## Data Dictionary

| Column | Description | Type |
|--------|-------------|------|
| Date | Trading date | datetime |
| Adj Close | Adjusted closing price | float |
| Volume | Trading volume | int |

## Crisis Events (Ground Truth)

Crisis labels are generated algorithmically using the composite method:

1. **Tail Events (EVT)**: Returns in bottom 5th percentile for 3+ consecutive days
2. **Drawdowns**: Max drawdown > 10% in 20-day window
3. **Jump Detection**: Barndorff-Nielsen & Shephard test (p < 0.01)
4. **Percentile**: Return < p10 OR Volatility > p90

A day is labeled as "crisis" if at least 2 methods agree.

## File Structure

After running the replication notebook:

```
data/
├── README.md              # This file
├── prices_daily.csv       # Downloaded price data (optional save)
├── features/
│   └── computed_features.parquet  # Pre-computed features (optional)
└── labels/
    └── crisis_labels.parquet      # Generated crisis labels (optional)
```

## Data Availability Statement

All data used in this study is publicly available through Yahoo Finance API.
No proprietary or restricted data was used.

For replication:
1. Run the Jupyter notebook `notebooks/full_replication.ipynb`
2. Data will be automatically downloaded on first run
3. Results will be saved to `results/` directory

## Contact

For data-related questions, please open an issue on the project repository.
