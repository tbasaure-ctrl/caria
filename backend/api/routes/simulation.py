from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from ..services.simulation_service import SimulationService

router = APIRouter(prefix="/api/simulation", tags=["simulation"])

class PortfolioItem(BaseModel):
    ticker: str
    quantity: float
    weight: Optional[float] = 0.0

class CrisisRequest(BaseModel):
    portfolio: List[PortfolioItem]
    crisis_id: str

class MacroRequest(BaseModel):
    portfolio: List[PortfolioItem]
    params: Dict[str, float]

def get_simulation_service():
    return SimulationService()

@router.post("/crisis")
async def simulate_crisis(
    request: CrisisRequest,
    service: SimulationService = Depends(get_simulation_service)
):
    try:
        # Convert Pydantic models to dicts for service
        portfolio_dicts = [item.dict() for item in request.portfolio]
        result = service.simulate_crisis(portfolio_dicts, request.crisis_id)
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/macro")
async def simulate_macro(
    request: MacroRequest,
    service: SimulationService = Depends(get_simulation_service)
):
    try:
        portfolio_dicts = [item.dict() for item in request.portfolio]
        return service.simulate_macro(portfolio_dicts, request.params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
