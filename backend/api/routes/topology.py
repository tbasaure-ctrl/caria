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
        from caria.services.fmp_service import FMPDataService
        from caria.services.topology_engine import TopologicalMRI
        
        # 1. Ingest Real Data
        data_service = FMPDataService()
        market_data = data_service.fetch_market_pulse(lookback_days=60)
        
        if market_data.empty:
             raise HTTPException(status_code=500, detail="No market data available for scan")

        # 2. Run the Topological Scan
        mri = TopologicalMRI()
        result = mri.scan(market_data)
        
        return result
        
    except Exception as e:
        print(f"Topological Scan Failed: {e}")
        raise HTTPException(status_code=500, detail=f"Topological Scan Failed: {str(e)}")
