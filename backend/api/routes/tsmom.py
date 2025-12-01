from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from api.services.tsmom_service import tsmom_service

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

@router.get("/tsmom/{symbol}")
async def get_tsmom_signal(symbol: str) -> Dict[str, Any]:
    """
    Get Time Series Momentum (TSMOM) signal for a symbol.
    Returns trend direction, strength, and volatility context.
    """
    result = tsmom_service.calculate_market_regime(symbol)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result.get("error_detail", "Unknown error"))
    return result

