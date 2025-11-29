from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from caria.services.liquidity_service import LiquidityService

router = APIRouter(prefix="/liquidity", tags=["Liquidity"])
LOGGER = logging.getLogger(__name__)

def get_liquidity_service():
    return LiquidityService()

@router.get("/status")
async def get_liquidity_status(
    service: LiquidityService = Depends(get_liquidity_service)
) -> Dict[str, Any]:
    """
    Get the current Hydraulic Score and Liquidity State.
    """
    try:
        return service.get_current_status()
    except Exception as e:
        LOGGER.error(f"Error fetching liquidity status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy")
async def get_hydraulic_strategy(
    service: LiquidityService = Depends(get_liquidity_service)
) -> Dict[str, Any]:
    """
    Get the active strategy mode based on the current liquidity state.
    """
    try:
        return service.get_strategy_mode()
    except Exception as e:
        LOGGER.error(f"Error fetching hydraulic strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))
