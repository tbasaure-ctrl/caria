from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

from ..services.simple_valuation import SimpleValuationService

router = APIRouter()
LOGGER = logging.getLogger("caria.api.valuation")

class ValuationRequest(BaseModel):
    current_price: float

@router.post("/{ticker}")
async def get_valuation(ticker: str, request: ValuationRequest):
    """
    Get valuation using the robust SimpleValuationService.
    """
    try:
        service = SimpleValuationService()
        result = service.get_valuation(ticker, request.current_price)
        return result
    except Exception as e:
        LOGGER.error(f"Valuation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
