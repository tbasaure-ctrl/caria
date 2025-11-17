"""Rutas para screening de factores cuantitativos."""

from __future__ import annotations

import sys
from pathlib import Path

# Configurar paths para encontrar caria
CURRENT_FILE = Path(__file__).resolve()
CARIA_DATA_SRC = CURRENT_FILE.parent.parent.parent.parent / "caria_data" / "src"
if CARIA_DATA_SRC.exists() and str(CARIA_DATA_SRC) not in sys.path:
    sys.path.insert(0, str(CARIA_DATA_SRC))

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.validators import ISODate, TopN
from caria.models.auth import UserInDB
from caria.services.factor_service import FactorService

router = APIRouter(prefix="/api/factors", tags=["factors"])


class FactorScreenRequest(BaseModel):
    """Request para screening de factores."""
    top_n: TopN = Field(50, description="Número de empresas a retornar (1-500)")
    regime: str | None = Field(None, description="Régimen macro (opcional, se detecta automáticamente)")
    date: ISODate | None = Field(None, description="Fecha específica para screening en formato ISO (YYYY-MM-DD)")
    page: int = Field(1, ge=1, description="Número de página (empezando en 1)")
    page_size: int = Field(50, ge=1, le=100, description="Tamaño de página (1-100)")


class CompanyFactorScore(BaseModel):
    """Score de factores para una empresa."""
    ticker: str
    date: str
    composite_score: float
    rank: int
    factor_scores: dict[str, float]
    regime: str


class FactorScreenResponse(BaseModel):
    """Respuesta de screening de factores."""
    companies: list[CompanyFactorScore]
    regime_used: str
    pagination: dict[str, int | bool]


def _guard_factor_service(request: Request) -> FactorService:
    """Verifica que el servicio de factores esté disponible."""
    factor_service = getattr(request.app.state, "factor_service", None)
    if factor_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio de factores no disponible.",
        )
    return factor_service


@router.post("/screen", response_model=FactorScreenResponse)
def screen_companies(
    request: Request,
    payload: FactorScreenRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> FactorScreenResponse:
    """Screena empresas usando factores cuantitativos.
    
    Retorna top N empresas rankeadas por composite score de factores,
    ajustado según régimen macro actual.
    """
    factor_service = _guard_factor_service(request)
    
    try:
        # Obtener todas las empresas (sin paginación del servicio)
        all_companies = factor_service.screen_companies(
            top_n=payload.top_n * payload.page_size,  # Obtener más para paginar
            regime=payload.regime,
            date=payload.date,
        )
        
        # Si no hay empresas, retornar lista vacía en lugar de error
        if not all_companies:
            import logging
            logger = logging.getLogger("caria.api.factors")
            logger.warning("No companies found in factor screening, returning empty list")
            return FactorScreenResponse(
                companies=[],
                regime_used=payload.regime or "unknown",
                pagination={
                    "total": 0,
                    "page": payload.page,
                    "page_size": payload.page_size,
                    "has_next": False,
                    "has_prev": False,
                },
            )
        
        # Aplicar paginación
        total = len(all_companies)
        start_idx = (payload.page - 1) * payload.page_size
        end_idx = start_idx + payload.page_size
        paginated_companies = all_companies[start_idx:end_idx]
        
        regime_used = all_companies[0]["regime"] if all_companies else payload.regime or "unknown"
        
        return FactorScreenResponse(
            companies=[CompanyFactorScore(**c) for c in paginated_companies],
            regime_used=regime_used,
            pagination={
                "total": total,
                "page": payload.page,
                "page_size": payload.page_size,
                "has_next": end_idx < total,
                "has_prev": payload.page > 1,
            },
        )
    except Exception as exc:
        import logging
        logger = logging.getLogger("caria.api.factors")
        logger.exception("Error en factor screening: %s", exc)
        # Retornar lista vacía en lugar de error 500
        return FactorScreenResponse(
            companies=[],
            regime_used=payload.regime or "unknown",
            pagination={
                "total": 0,
                "page": payload.page,
                "page_size": payload.page_size,
                "has_next": False,
                "has_prev": False,
            },
        )

