"""Validadores Pydantic customizados para inputs de API."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, Field, field_validator


def validate_ticker(value: str | None) -> str | None:
    """Valida formato de ticker (3-5 caracteres, alfanuméricos, uppercase)."""
    if value is None:
        return None
    
    if not value:
        raise ValueError("Ticker no puede estar vacío")
    
    # Limpiar espacios y convertir a uppercase
    ticker = value.strip().upper()
    
    # Validar longitud (algunos tickers pueden tener hasta 6 caracteres, ej: NVIDIA)
    if len(ticker) < 1 or len(ticker) > 6:
        raise ValueError("Ticker debe tener entre 1 y 6 caracteres")
    
    # Validar formato (solo letras y números)
    if not re.match(r"^[A-Z0-9]+$", ticker):
        raise ValueError("Ticker solo puede contener letras y números")
    
    return ticker


def validate_date(value: str) -> str:
    """Valida formato de fecha ISO (YYYY-MM-DD)."""
    if not value:
        raise ValueError("Fecha no puede estar vacía")
    
    try:
        # Intentar parsear fecha ISO
        date_obj = datetime.fromisoformat(value.replace("Z", "+00:00"))
        
        # Validar rango razonable (no más de 100 años en el futuro, no antes de 1900)
        if date_obj.year < 1900:
            raise ValueError("Fecha no puede ser anterior a 1900")
        
        if date_obj.year > datetime.now().year + 100:
            raise ValueError("Fecha no puede ser más de 100 años en el futuro")
        
        return value
    except ValueError as exc:
        if "Invalid isoformat" in str(exc) or "does not match format" in str(exc):
            raise ValueError("Fecha debe estar en formato ISO (YYYY-MM-DD)") from exc
        raise


def validate_top_n(value: int) -> int:
    """Valida que top_n esté en rango válido (1-500)."""
    if value < 1:
        raise ValueError("top_n debe ser al menos 1")
    if value > 500:
        raise ValueError("top_n no puede ser mayor a 500")
    return value


# Type aliases con validación
Ticker = Annotated[str, AfterValidator(validate_ticker), Field(description="Stock ticker symbol (3-5 chars, uppercase)")]
ISODate = Annotated[str, AfterValidator(validate_date), Field(description="Date in ISO format (YYYY-MM-DD)")]
TopN = Annotated[int, AfterValidator(validate_top_n), Field(ge=1, le=500, description="Number of results to return")]

