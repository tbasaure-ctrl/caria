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
        # TODO: Replace with real S&P 500 returns fetch
        # For now, generate a synthetic "Healthy" or "Critical" market for demo
        # based on random seed or time to show functionality
        
        # Synthetic Data Generation for Demo (replace with real DB fetch)
        # Generate 50 assets, 60 days
        np.random.seed(42)
        
        # Simulate a "Healthy" market (random walk, low correlation)
        # or "Critical" (high correlation factor model)
        
        # Let's simulate a mix to get some interesting topology
        n_assets = 30
        n_days = 60
        
        # Market factor
        mkt = np.random.normal(0, 1, n_days)
        
        returns_dict = {}
        for i in range(n_assets):
            # Asset specific noise
            noise = np.random.normal(0, 1, n_days)
            # Correlation to market (randomize to create structure)
            beta = np.random.uniform(0, 1.5) 
            
            # Create an "Alien" (Asset 0)
            if i == 0:
                ret = np.random.normal(0, 2, n_days) # Pure noise, high vol, no beta
            else:
                ret = beta * mkt + noise
                
            returns_dict[f"TICKER_{i}"] = ret
            
        returns_df = pd.DataFrame(returns_dict)
        
        # Run Scan
        result = topology_service.scan_market_topology(returns_df)
        
        # Decorate with "Alien" names for flavor if it's our synthetic data
        if result.get("aliens"):
            result["aliens"][0]["ticker"] = "NVDA-X (Alien)" # Example
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
