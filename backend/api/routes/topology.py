from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import pandas as pd
import numpy as np
from caria.services.topology_service import TopologyService
# Mock data generator for demo purposes if real data isn't fully piped yet
from caria.services.market_data_service import MarketDataService 

router = APIRouter(prefix="/api/topology", tags=["topology"])

# Singleton instances (in a real app, use dependency injection)
topology_service = TopologyService()
# market_data_service = MarketDataService() # Assuming this exists or we mock

@router.get("/scan")
async def get_topology_scan():
    """
    Performs a real-time Topological MRI scan of the market.
    """
    try:
        # Use FMP Client to fetch real data
        from caria.ingestion.clients.fmp_client import FMPClient
        import os
        from datetime import datetime, timedelta

        fmp_client = FMPClient()
        
        # Basket of "Mag 7" + Indices + Volatility for Topological Analysis
        # These are the "Pillars" of the market manifold
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'SPY', 'QQQ']
        
        # Fetch last 60 days of data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d') # Buffer for trading days
        
        returns_dict = {}
        
        for ticker in tickers:
            try:
                history = fmp_client.get_price_history(ticker, start_date=start_date, end_date=end_date)
                if history:
                    # Convert to DataFrame
                    df = pd.DataFrame(history)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()
                    
                    # Calculate daily returns
                    # Use 'adjClose' if available, else 'close'
                    price_col = 'adjClose' if 'adjClose' in df.columns else 'close'
                    returns_dict[ticker] = df[price_col].pct_change().dropna()
            except Exception as e:
                print(f"Failed to fetch data for {ticker}: {e}")
                continue

        if not returns_dict:
             raise HTTPException(status_code=500, detail="Failed to fetch market data from FMP")

        # Align data (inner join to ensure same dates)
        returns_df = pd.DataFrame(returns_dict).dropna()
        
        if returns_df.empty:
             raise HTTPException(status_code=500, detail="Insufficient overlapping market data")

        # Run Scan
        result = topology_service.scan_market_topology(returns_df)
        
        return result
        
    except Exception as e:
        # Fallback to synthetic if FMP fails (e.g. no key)
        print(f"Real data scan failed: {e}. Falling back to synthetic.")
        # ... (keep synthetic fallback or just raise error? User asked for real data)
        # Let's raise the error to be transparent about the key requirement
        raise HTTPException(status_code=500, detail=f"Topological Scan Failed: {str(e)}")
