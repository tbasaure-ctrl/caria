"""Rutas para precios en tiempo real."""

from __future__ import annotations

import sys
from pathlib import Path

# Configurar paths para encontrar caria
CURRENT_FILE = Path(__file__).resolve()
CARIA_DATA_SRC = CURRENT_FILE.parent.parent.parent.parent / "caria_data" / "src"
if CARIA_DATA_SRC.exists() and str(CARIA_DATA_SRC) not in sys.path:
    sys.path.insert(0, str(CARIA_DATA_SRC))

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from caria.models.auth import UserInDB
from api.services.openbb_client import OpenBBClient

router = APIRouter(prefix="/api/prices", tags=["prices"])


class RealtimePriceRequest(BaseModel):
    """Request para precios en tiempo real."""
    tickers: list[str] = Field(..., min_items=1, max_items=50, description="Lista de tickers (máx 50)")


class RealtimePriceResponse(BaseModel):
    """Respuesta con precios en tiempo real."""
    prices: dict[str, dict[str, Any]]  # ticker -> datos de precio


def _get_openbb_client() -> OpenBBClient:
    """Obtiene cliente OpenBB."""
    return OpenBBClient()


@router.post("/realtime", response_model=RealtimePriceResponse)
def get_realtime_prices(
    request: RealtimePriceRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> RealtimePriceResponse:
    """Obtiene precios en tiempo real para múltiples tickers.
    
    Usa FMP API para obtener datos actualizados de precios, cambios y porcentajes.
    """
    try:
        client = _get_openbb_client()
        prices = client.get_current_prices(request.tickers)
        
        return RealtimePriceResponse(prices=prices)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo precios: {str(exc)}",
        ) from exc


@router.get("/realtime/{ticker}")
def get_realtime_price_single(
    ticker: str,
    current_user: UserInDB = Depends(get_current_user),
) -> dict[str, Any]:
    """Obtiene precio en tiempo real para un solo ticker."""
    try:
        client = _get_openbb_client()
        # Use batch method for single ticker to get full dict structure
        prices = client.get_current_prices([ticker.upper()])
        price_data = prices.get(ticker.upper())
        
        if price_data is None or price_data.get("price") == 0:
             # Try fallback if 0
             pass
        
        if price_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Precio no encontrado para {ticker}",
            )
        
        return price_data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo precio para {ticker}: {str(exc)}",
        ) from exc

