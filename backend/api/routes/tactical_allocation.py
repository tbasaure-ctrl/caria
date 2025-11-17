"""
Tactical Asset Allocation endpoints per audit document (2.2).
Macro-conditional portfolio allocation based on regime signals (Tabla 4).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_current_user
from api.services.tactical_allocation import get_tactical_allocation_service
from caria.models.auth import UserInDB

router = APIRouter(prefix="/api/portfolio/tactical", tags=["Tactical Allocation"])


@router.get("/allocation")
async def get_tactical_allocation(
    regime: Optional[str] = Query(None, description="Regime override (expansion, slowdown, recession, stress)"),
    vix: Optional[float] = Query(None, description="VIX override"),
    detailed: bool = Query(False, description="Include sub-asset class breakdown"),
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Get tactical asset allocation based on current regime signals.
    
    Per audit document (2.2): Returns allocation percentages based on Tabla 4:
    - Alto Riesgo (Stress/Recession + VIX > 25): 30% stocks / 70% bonds
    - Riesgo Moderado (Slowdown): 50% stocks / 50% bonds
    - Bajo Riesgo (Expansion + VIX < 20): 70% stocks / 30% bonds
    
    If regime not provided, fetches current regime from model.
    """
    try:
        service = get_tactical_allocation_service()
        
        # Get regime from model if not provided
        if not regime:
            # Fetch current regime from regime service
            from caria.services.regime_service import RegimeService
            import os
            import psycopg2
            
            db_conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                user=os.getenv("POSTGRES_USER", "caria_user"),
                password=os.getenv("POSTGRES_PASSWORD"),
                database=os.getenv("POSTGRES_DB", "caria"),
            )
            
            try:
                regime_service = RegimeService(db_conn)
                current_regime = regime_service.get_current_regime()
                regime = current_regime.get("regime", "slowdown") if current_regime else "slowdown"
            finally:
                db_conn.close()
        
        # Validate regime
        valid_regimes = ["expansion", "slowdown", "recession", "stress"]
        if regime.lower() not in valid_regimes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid regime. Must be one of: {', '.join(valid_regimes)}"
            )
        
        # Get allocation
        if detailed:
            result = service.get_detailed_allocation(regime.lower(), vix)
        else:
            result = service.get_allocation(regime.lower(), vix, include_etfs=True)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating allocation: {str(e)}") from e


@router.get("/risk-level")
async def get_risk_level(
    regime: str = Query(..., description="Current regime"),
    vix: Optional[float] = Query(None, description="VIX level (fetched if not provided)"),
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Get risk level determination based on regime and VIX.
    
    Returns risk level classification: "high_risk", "moderate_risk", "low_risk", "extreme_stress"
    """
    try:
        service = get_tactical_allocation_service()
        risk_level = service.determine_risk_level(regime, vix)
        
        from api.services.tactical_allocation import ALLOCATION_RULES
        
        return {
            "regime": regime,
            "vix": vix if vix else service.get_current_vix(),
            "risk_level": risk_level,
            "description": ALLOCATION_RULES.get(risk_level, {}).get("description", "Risk level determined"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error determining risk level: {str(e)}") from e

