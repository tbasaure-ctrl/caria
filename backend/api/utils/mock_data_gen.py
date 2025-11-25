import pandas as pd
import numpy as np
from pathlib import Path
import logging

LOGGER = logging.getLogger("caria.utils.mock_data")

def ensure_mock_data_exists():
    """
    Checks if required parquet files exist. If not, generates mock data.
    This ensures the application works on Railway/Production even if the data pipeline hasn't run.
    """
    try:
        # Define paths relative to this file (backend/api/utils/mock_data_gen.py)
        # We need to reach /workspace/data/silver/fundamentals
        # From here: ../../../data/silver/fundamentals
        # Actually, it depends on where the app is running.
        # In app.py: BASE_DIR = Path(__file__).resolve().parents[2] (backend/)
        # Data usually at root level or caria_data.
        
        # Let's try to find the data directory relative to the backend root
        backend_root = Path(__file__).resolve().parents[3] # workspace/
        base_path = backend_root / "data" / "silver" / "fundamentals"
        
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=True)
            
        quality_path = base_path / "quality_signals.parquet"
        value_path = base_path / "value_signals.parquet"
        
        if quality_path.exists() and value_path.exists():
            LOGGER.info("Data layer files exist. Skipping mock generation.")
            return

        LOGGER.warning(f"Data layer missing at {base_path}. Generating mock data for stability...")
        
        # Define mock tickers
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK.B", "LLY", "V", "SPY", "QQQ"]
        
        # 1. Generate Quality Signals
        quality_data = {
            "ticker": tickers,
            "date": [pd.Timestamp.now()] * len(tickers),
            "profitability": np.random.uniform(40, 99, len(tickers)), 
            "momentum": np.random.uniform(30, 95, len(tickers)),      
            "value": np.random.uniform(20, 90, len(tickers)),         
            "growth": np.random.uniform(10, 90, len(tickers)),        
            "sector": ["Technology"] * len(tickers)
        }
        df_quality = pd.DataFrame(quality_data)
        df_quality.to_parquet(quality_path)
        
        # 2. Generate Value Signals
        value_data = {
            "ticker": tickers,
            "date": [pd.Timestamp.now()] * len(tickers),
            "revenue": np.random.uniform(1e10, 3e11, len(tickers)),
            "freeCashFlowPerShare": np.random.uniform(2, 15, len(tickers)),
            "sharesOutstanding": np.random.uniform(1e9, 1.5e10, len(tickers)),
            "freeCashFlow": np.random.uniform(1e9, 1e11, len(tickers)),
            "totalDebt": np.random.uniform(1e9, 5e10, len(tickers)),
            "cashAndCashEquivalents": np.random.uniform(1e9, 5e10, len(tickers)),
            "ebitda": np.random.uniform(5e9, 1e11, len(tickers)),
            "price": np.random.uniform(100, 500, len(tickers)),
            "sector": ["Technology"] * len(tickers),
            "industry": ["Consumer Electronics"] * len(tickers)
        }
        df_value = pd.DataFrame(value_data)
        df_value.to_parquet(value_path)
        
        LOGGER.info("Mock data generation complete.")
        
    except Exception as e:
        LOGGER.error(f"Failed to generate mock data: {e}")
