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
from api.services.openbb_client import openbb_client


def _value(entry: Any, field: str) -> Any:
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get(field)
    if hasattr(entry, "model_dump"):
        data = entry.model_dump()
        if isinstance(data, dict):
            return data.get(field)
    return getattr(entry, field, None)

router = APIRouter(prefix="/api/prices", tags=["prices"])


class RealtimePriceRequest(BaseModel):
    """Request para precios en tiempo real."""
    tickers: list[str] = Field(..., min_items=1, max_items=50, description="Lista de tickers (máx 50)")


class RealtimePriceResponse(BaseModel):
    """Respuesta con precios en tiempo real."""
    prices: dict[str, dict[str, Any]]  # ticker -> datos de precio


def _build_quote(symbol: str) -> dict[str, Any]:
    history = openbb_client.get_price_history(symbol, start_date="2023-01-01")
    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Precio no encontrado para {symbol}")

    latest = history[-1]
    prev = history[-2] if len(history) > 1 else None
    close = _value(latest, "close") or _value(latest, "adj_close")
    prev_close = _value(prev, "close") if prev is not None else None

    change = None
    change_percent = None
    if close is not None and prev_close is not None:
        change = close - prev_close
        if prev_close != 0:
            change_percent = (change / prev_close) * 100

    return {
        "symbol": symbol.upper(),
        "price": close,
        "previousClose": prev_close,
        "change": change,
        "changesPercentage": change_percent,
        "date": _value(latest, "date"),
    }


@router.post("/realtime", response_model=RealtimePriceResponse)
def get_realtime_prices(
    request: RealtimePriceRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> RealtimePriceResponse:
    """Obtiene precios en tiempo real para múltiples tickers.
    
    Usa FMP API para obtener datos actualizados de precios, cambios y porcentajes.
    """
    try:
        prices: dict[str, dict[str, Any]] = {}
        for ticker in request.tickers:
            prices[ticker.upper()] = _build_quote(ticker)
        return RealtimePriceResponse(prices=prices)
    except HTTPException:
        raise
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
        return _build_quote(ticker)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo precio para {ticker}: {str(exc)}",
        ) from exc

