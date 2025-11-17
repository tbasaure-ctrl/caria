"""Rutas para detección de régimen macroeconómico."""

from __future__ import annotations

import sys
from pathlib import Path

# Configurar paths para encontrar caria
CURRENT_FILE = Path(__file__).resolve()
CARIA_DATA_SRC = CURRENT_FILE.parent.parent.parent.parent / "caria_data" / "src"
if CARIA_DATA_SRC.exists() and str(CARIA_DATA_SRC) not in sys.path:
    sys.path.insert(0, str(CARIA_DATA_SRC))

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.dependencies import check_rate_limit, get_optional_current_user
from caria.models.auth import UserInDB
from caria.services.regime_service import RegimeService

LOGGER = logging.getLogger("caria.api.regime")

router = APIRouter(prefix="/api/regime", tags=["regime"])


class RegimeResponse(BaseModel):
    """Respuesta con probabilidades de régimen."""
    regime: str
    probabilities: dict[str, float]
    confidence: float
    features_used: dict[str, float] | None = None  # Opcional si no hay datos disponibles


def _guard_regime_service(request: Request) -> RegimeService:
    """Verifica que el servicio de régimen esté disponible."""
    regime_service = getattr(request.app.state, "regime_service", None)
    if regime_service is None or not regime_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio de régimen no disponible. Entrena el modelo HMM primero.",
        )
    return regime_service


@router.get("/current", response_model=RegimeResponse)
def get_current_regime(
    request: Request,
    current_user: UserInDB | None = Depends(get_optional_current_user),
    _: None = Depends(check_rate_limit),
) -> RegimeResponse:
    """Obtiene el régimen macroeconómico actual detectado por HMM.
    
    Retorna probabilidades de estar en cada régimen:
    - expansion: Crecimiento económico fuerte
    - slowdown: Desaceleración económica
    - recession: Contracción económica
    - stress: Crisis/volatilidad extrema
    """
    regime_service = getattr(request.app.state, "regime_service", None)
    
    # Si el servicio no está disponible, retornar régimen por defecto
    if regime_service is None or not regime_service.is_available():
        LOGGER.warning("Regime service not available, returning default regime")
        return RegimeResponse(
            regime="slowdown",
            confidence=0.5,
            probabilities={
                "expansion": 0.2,
                "slowdown": 0.5,
                "recession": 0.2,
                "stress": 0.1,
            },
            features_used={}  # Features vacías cuando no hay datos
        )
    
    result = regime_service.get_regime_probabilities()
    if result is None:
        # Si no se puede detectar, retornar régimen por defecto en lugar de error
        LOGGER.warning("Could not detect regime, returning default")
        return RegimeResponse(
            regime="slowdown",
            confidence=0.5,
            probabilities={
                "expansion": 0.2,
                "slowdown": 0.5,
                "recession": 0.2,
                "stress": 0.1,
            },
            features_used={}  # Features vacías cuando no hay datos
        )
    
    # Asegurar que features_used esté presente
    if "features_used" not in result or result["features_used"] is None:
        result["features_used"] = {}
    
    return RegimeResponse(**result)

