from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import pandas as pd
import numpy as np
import sys
import os
# Add caria-lib to path to allow importing shared services
# Path relative to this file: ../../../../caria-lib
current_dir = os.path.dirname(os.path.abspath(__file__))
caria_lib_path = os.path.abspath(os.path.join(current_dir, "../../../../caria-lib"))
if caria_lib_path not in sys.path:
    sys.path.append(caria_lib_path)

try:
    from caria.services.topology_service import TopologyService
except ImportError as e:
    print(f"Failed to import TopologyService: {e}")
    TopologyService = None

# Mock data generator for demo purposes if real data isn't fully piped yet
from caria.services.market_data_service import MarketDataService 

router = APIRouter(prefix="/api/topology", tags=["topology"])

# Singleton instances (in a real app, use dependency injection)
if TopologyService:
    topology_service = TopologyService()
else:
    topology_service = None
# market_data_service = MarketDataService() # Assuming this exists or we mock

@router.get("/scan")
async def get_topology_scan():
    """
    Performs a real-time Topological MRI scan of the market.
    """
    if not topology_service:
        return {
            "status": "OFFLINE", 
            "diagnosis": "Service Unavailable", 
            "description": "Topology engine not loaded (missing dependencies?)",
            "status_color": "gray",
            "metrics": {"betti_1_loops": 0, "total_persistence": 0, "complexity_score": 0},
            "aliens": []
        }

    try:
        from caria.services.fmp_service import FMPDataService
        
        # 1. Ingest Real Data
        data_service = FMPDataService()
        market_data = data_service.fetch_market_pulse(lookback_days=60)
        
        if market_data.empty:
             # Return a "Waiting for Data" state instead of 500
             return {
                "status": "WAITING",
                "diagnosis": "Insufficient Data",
                "description": "Waiting for market data feed...",
                "status_color": "yellow",
                "metrics": {"betti_1_loops": 0, "total_persistence": 0, "complexity_score": 0},
                "aliens": []
             }

        # 2. Run the Topological Scan
        result = topology_service.scan_market_topology(market_data)
        
        return result
        
    except Exception as e:
        print(f"Topological Scan Failed: {e}")
        # Return error state instead of 500 to keep widget alive
        return {
            "status": "ERROR",
            "diagnosis": "System Error",
            "description": str(e),
            "status_color": "red",
            "metrics": {"betti_1_loops": 0, "total_persistence": 0, "complexity_score": 0},
            "aliens": []
        }
