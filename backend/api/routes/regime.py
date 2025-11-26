"""Rutas para detección de régimen macroeconómico."""

from __future__ import annotations

import math
import statistics
import sys
from datetime import datetime, timedelta
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
from api.services.openbb_client import OpenBBClient

# Instantiate OpenBBClient for fallback
openbb_client = OpenBBClient()

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


def _quick_regime_from_prices(symbol: str = "SPY") -> RegimeResponse | None:
    """Fallback regime classification using price momentum & volatility."""
    try:
        start_date = (datetime.utcnow() - timedelta(days=420)).date().isoformat()
        history = openbb_client.get_price_history(symbol, start_date=start_date)

        # Convert OBBject to list of dicts if needed
        if history and hasattr(history, 'to_df'):
            import pandas as pd
            df = history.to_df()
            if not df.empty and 'close' in df.columns:
                closes = df['close'].dropna().tolist()
            else:
                LOGGER.warning(f"No close price data in history for {symbol}")
                return None
        elif isinstance(history, (list, tuple)):
            # Handle list/tuple of entries
            closes = []
            for entry in history:
                if isinstance(entry, dict):
                    close = entry.get("close") or entry.get("adj_close")
                    if close is not None:
                        closes.append(float(close))
                elif hasattr(entry, 'close'):
                    closes.append(float(entry.close))
        else:
            LOGGER.warning(f"Unexpected history format for {symbol}: {type(history)}")
            return None

        closes = [float(price) for price in closes if price is not None and price > 0]

        if len(closes) < 60:
            LOGGER.warning(f"Insufficient data points for {symbol}: {len(closes)}")
            return None

        def pct_change(days: int) -> float:
            if len(closes) <= days:
                return 0.0
            return (closes[-1] / closes[-days] - 1) * 100

        ma50 = statistics.fmean(closes[-50:]) if len(closes) >= 50 else closes[-1]
        ma200 = statistics.fmean(closes[-200:]) if len(closes) >= 200 else statistics.fmean(closes)
        r1m = pct_change(21)
        r3m = pct_change(63)
        r6m = pct_change(126)
        r12m = pct_change(252)

        vol = 0.0
        if len(closes) >= 63:
            returns = [
                (closes[i] / closes[i - 1]) - 1
                for i in range(len(closes) - 62, len(closes))
            ]
            if returns:
                vol = statistics.pstdev(returns) * math.sqrt(252) * 100

        score = 0
        if ma50 > ma200:
            score += 1
        if r3m > 0:
            score += 1
        if r6m > 0:
            score += 1
        if r12m > 0:
            score += 1

        stress_trigger = r1m <= -7 or vol >= 28
        if stress_trigger:
            regime = "stress"
        elif score >= 3:
            regime = "expansion"
        elif score == 2:
            regime = "slowdown"
        else:
            regime = "recession"

        confidence = min(0.95, max(0.2, 0.35 + (score / 4) * 0.4))
        if stress_trigger:
            confidence = min(0.9, confidence + 0.1)

        base_probs = {
            "expansion": 0.2,
            "slowdown": 0.25,
            "recession": 0.3,
            "stress": 0.25,
        }
        dominant = min(0.75, 0.45 + confidence / 2)
        base_probs[regime] = dominant
        total_other = sum(v for k, v in base_probs.items() if k != regime)
        remainder = 1 - dominant
        if total_other > 0:
            for key in base_probs:
                if key == regime:
                    continue
                base_probs[key] = remainder * (base_probs[key] / total_other)

        features_used = {
            "ma50": round(ma50, 2),
            "ma200": round(ma200, 2),
            "return_1m_pct": round(r1m, 2),
            "return_3m_pct": round(r3m, 2),
            "return_6m_pct": round(r6m, 2),
            "return_12m_pct": round(r12m, 2),
            "volatility_pct": round(vol, 2),
        }

        return RegimeResponse(
            regime=regime,
            probabilities={key: float(value) for key, value in base_probs.items()},
            confidence=float(confidence),
            features_used=features_used,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Fallback regime computation failed: %s", exc)
        return None


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
        LOGGER.warning("Regime service not available, using OpenBB fallback")
        fallback = _quick_regime_from_prices()
        if fallback:
            return fallback
        return RegimeResponse(
            regime="slowdown",
            confidence=0.5,
            probabilities={
                "expansion": 0.2,
                "slowdown": 0.5,
                "recession": 0.2,
                "stress": 0.1,
            },
            features_used={},
        )
    
    result = regime_service.get_regime_probabilities()
    if result is None:
        LOGGER.warning("Could not detect regime with trained model, using fallback heuristics")
        fallback = _quick_regime_from_prices()
        if fallback:
            return fallback
        return RegimeResponse(
            regime="slowdown",
            confidence=0.5,
            probabilities={
                "expansion": 0.2,
                "slowdown": 0.5,
                "recession": 0.2,
                "stress": 0.1,
            },
            features_used={},
        )
    
    # Asegurar que features_used esté presente y solo contenga valores numéricos
    if "features_used" not in result or result["features_used"] is None:
        result["features_used"] = {}
    else:
        # Filtrar solo valores numéricos (excluir strings como 'symbol')
        result["features_used"] = {
            k: float(v) for k, v in result["features_used"].items()
            if isinstance(v, (int, float)) and not (isinstance(v, bool))
        }
    
    return RegimeResponse(**result)

