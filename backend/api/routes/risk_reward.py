"""
Risk-Reward Analysis API Routes
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator
import logging

from ..services.risk_reward_service import RiskRewardService
from ..dependencies import get_current_user

router = APIRouter(prefix="/api/risk-reward", tags=["risk-reward"])
LOGGER = logging.getLogger("caria.api.risk_reward")


class RiskRewardRequest(BaseModel):
    """Request model for risk-reward analysis."""
    ticker: str = Field(..., description="Stock ticker symbol")
    horizon_months: int = Field(24, ge=12, le=36, description="Time horizon in months (12, 24, or 36)")
    probabilities: dict[str, float] | None = Field(
        None,
        description="Optional probabilities for bear/base/bull scenarios (must sum to 1.0)"
    )
    
    @validator('probabilities')
    def validate_probabilities(cls, v):
        if v is not None:
            total = sum(v.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Probabilities must sum to 1.0, got {total}")
            # Ensure all keys are valid
            valid_keys = {'bear', 'base', 'bull'}
            if not all(k in valid_keys for k in v.keys()):
                raise ValueError(f"Probabilities must only contain keys: {valid_keys}")
        return v


class RiskRewardResponse(BaseModel):
    """Response model for risk-reward analysis."""
    ticker: str
    horizon_months: int
    current_price: float
    scenarios: dict[str, dict[str, float]]
    metrics: dict[str, float]
    explanations: dict[str, str]
    volatility_metrics: dict[str, float]


@router.post("/analyze", response_model=RiskRewardResponse)
async def analyze_risk_reward(
    request: RiskRewardRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Analyze risk-reward for a given stock.
    
    Generates Bear/Base/Bull scenarios, calculates Risk-Reward Ratio (RRR) and Expected Value (EV),
    and provides educational explanations with analogies.
    
    Args:
        request: RiskRewardRequest with ticker, horizon, and optional probabilities
        current_user: Authenticated user (from dependency)
    
    Returns:
        RiskRewardResponse with scenarios, metrics, and explanations
    """
    try:
        service = RiskRewardService()
        
        result = service.analyze(
            ticker=request.ticker,
            horizon_months=request.horizon_months,
            probabilities=request.probabilities,
        )
        
        return RiskRewardResponse(**result)
        
    except ValueError as e:
        LOGGER.warning(f"Validation error for {request.ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        LOGGER.error(f"Runtime error for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        LOGGER.error(f"Unexpected error analyzing {request.ticker}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze risk-reward for {request.ticker}: {str(e)}"
        )

