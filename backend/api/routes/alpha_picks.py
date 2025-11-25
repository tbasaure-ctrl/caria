from fastapi import APIRouter, Depends, Request, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

from api.dependencies import get_current_user
from api.services.alpha_service import AlphaService
from api.routes.factors import _guard_factor_service # Reuse the guard if possible, or reimplement

router = APIRouter(prefix="/api/alpha-picks", tags=["alpha-picks"])

class AlphaPickResponse(BaseModel):
    ticker: str
    company_name: str
    sector: str
    cas_score: float
    scores: Dict[str, float]
    explanation: str

@router.get("/", response_model=List[AlphaPickResponse])
def get_weekly_picks(
    request: Request,
    current_user = Depends(get_current_user)
):
    """
    Generate top 3 weekly alpha stock picks.
    """
    # Get FactorService from app state (it's initialized in main/app setup)
    factor_service = getattr(request.app.state, "factor_service", None)
    if not factor_service:
        raise HTTPException(status_code=503, detail="Factor service unavailable")
        
    alpha_service = AlphaService(factor_service)
    
    try:
        picks = alpha_service.compute_alpha_picks()
        return picks
    except Exception as e:
        import logging
        logger = logging.getLogger("caria.api.alpha_picks")
        logger.exception("Error generating alpha picks")
        raise HTTPException(status_code=500, detail=str(e))
