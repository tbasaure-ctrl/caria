"""Rutas de valuación: quick valuation DCF + explicación transparente."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field

from api.validators import Ticker

router = APIRouter(prefix="/api/valuation", tags=["valuation"])

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


# ---------- Modelos Pydantic que deben calzar con el frontend ----------


class DcfAssumptions(BaseModel):
    fcf_yield_start: float = Field(
        0.05,
        description="FCF yield inicial (FCF / Market Cap) usado como ancla si no hay datos completos.",
    )
    high_growth_rate: float = Field(
        0.12,
        description="Crecimiento anual de FCF en etapa de alto crecimiento.",
    )
    high_growth_years: int = Field(
        5, description="Número de años de alto crecimiento."
    )
    fade_years: int = Field(
        5,
        description="Años en que el crecimiento decae linealmente hasta la tasa terminal.",
    )
    terminal_growth_rate: float = Field(
        0.03,
        description="Crecimiento a perpetuidad (normalmente 2–3% para economías maduras).",
    )
    discount_rate: float = Field(
        0.10,
        description="Tasa de descuento requerida (cost of equity / WACC aproximado).",
    )
    horizon_years: int = Field(
        10,
        description="Horizonte total explícito del DCF. Normalmente = high_growth_years + fade_years.",
    )
    shares_outstanding: Optional[float] = Field(
        None, description="Número de acciones; si no se entrega, se intenta obtener desde FMP."
    )
    net_debt: Optional[float] = Field(
        None,
        description="Deuda neta (deuda total - caja). Si es None, se estima desde FMP cuando es posible.",
    )


class QuickValuationRequest(BaseModel):
    method: str = Field(
        "dcf",
        description="Método principal de valuación: por ahora 'dcf' (se puede extender a 'scorecard', 'multiples', etc.).",
    )
    assumptions: Optional[DcfAssumptions] = Field(
        None,
        description="Suposiciones del DCF; si no se entregan, se usan defaults razonables.",
    )


class QuickValuationResponse(BaseModel):
    ticker: str
    current_price: Optional[float]
    fair_value: Optional[float]
    upside_pct: Optional[float]
    method: str
    assumptions_used: Dict[str, Any]
    explanation: str
    raw_inputs: Dict[str, Any]


# ---------- Helpers FMP ----------


def _get_fmp_api_key() -> str:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="FMP_API_KEY no configurada en el backend.",
        )
    return api_key


def _fmp_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    api_key = _get_fmp_api_key()
    url = f"{FMP_BASE_URL}{path}"
    params = params or {}
    params["apikey"] = api_key
    resp = requests.get(url, params=params, timeout=20)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Error al llamar a FMP ({resp.status_code})",
        )
    data = resp.json()
    return data


def _fetch_basic_fundamentals(ticker: str) -> Dict[str, Any]:
    """Obtiene precio, market cap, FCF y deuda/caja desde FMP."""
    # Precio y market cap
    quote_data = _fmp_get(f"/quote/{ticker.upper()}")
    if not quote_data:
        raise HTTPException(
            status_code=404,
            detail=f"Precio no encontrado para {ticker.upper()}",
        )
    quote = quote_data[0]
    price = float(quote.get("price") or 0.0)
    market_cap = float(quote.get("marketCap") or 0.0)

    # Perfil (para acciones en circulación)
    profile_data = _fmp_get(f"/profile/{ticker.upper()}")
    shares_out = None
    if profile_data:
        profile = profile_data[0]
        shares_out = profile.get("sharesOutstanding")

    # Cash flow (FCF)
    cf_data = _fmp_get(f"/cash-flow-statement/{ticker.upper()}", params={"limit": 1})
    fcf = None
    if cf_data:
        cf = cf_data[0]
        # Algunos tickers traen freeCashFlow directo; si no, lo aproximamos
        if "freeCashFlow" in cf and cf["freeCashFlow"] is not None:
            fcf = float(cf["freeCashFlow"])
        else:
            op_cf = float(cf.get("operatingCashFlow") or 0.0)
            capex = float(cf.get("capitalExpenditure") or 0.0)
            fcf = op_cf + capex  # capex normalmente es negativo en FMP

    # Balance (deuda neta)
    bs_data = _fmp_get(
        f"/balance-sheet-statement/{ticker.upper()}",
        params={"limit": 1},
    )
    net_debt = None
    if bs_data:
        bs = bs_data[0]
        total_debt = float(bs.get("totalDebt") or 0.0)
        cash = float(
            bs.get("cashAndShortTermInvestments")
            or bs.get("cashAndCashEquivalents")
            or 0.0
        )
        net_debt = total_debt - cash

    return {
        "price": price,
        "market_cap": market_cap,
        "shares_outstanding": shares_out,
        "fcf": fcf,
        "net_debt": net_debt,
    }


# ---------- Lógica DCF muy transparente ----------


def _run_simple_dcf(
    fcf_current: float,
    assumptions: DcfAssumptions,
) -> Dict[str, Any]:
    """DCF de dos etapas + valor terminal con explicación de cada paso."""
    if assumptions.horizon_years < assumptions.high_growth_years + assumptions.fade_years:
        # Aseguramos que horizon ≥ high_growth + fade
        horizon = assumptions.high_growth_years + assumptions.fade_years
    else:
        horizon = assumptions.horizon_years

    discount_rate = assumptions.discount_rate
    g_high = assumptions.high_growth_rate
    g_terminal = assumptions.terminal_growth_rate

    cash_flows = []
    current_fcf = fcf_current

    # 1) Etapa de alto crecimiento
    for year in range(1, assumptions.high_growth_years + 1):
        current_fcf *= (1 + g_high)
        cash_flows.append((year, current_fcf))

    # 2) Etapa de "fade": la tasa baja linealmente hasta g_terminal
    fade_years = max(assumptions.fade_years, 0)
    if fade_years > 0:
        step = (g_high - g_terminal) / fade_years
        g_current = g_high
        for i in range(1, fade_years + 1):
            g_current -= step
            current_fcf *= (1 + g_current)
            year = assumptions.high_growth_years + i
            cash_flows.append((year, current_fcf))

    # 3) Si horizon_years > high_growth + fade, crecemos al ritmo terminal
    last_year = assumptions.high_growth_years + fade_years
    while last_year < horizon:
        current_fcf *= (1 + g_terminal)
        last_year += 1
        cash_flows.append((last_year, current_fcf))

    # 4) Descontar flujos
    pv_flows = 0.0
    for year, cf in cash_flows:
        pv = cf / ((1 + discount_rate) ** year)
        pv_flows += pv

    # 5) Valor terminal a partir del último FCF
    fcf_terminal = cash_flows[-1][1] * (1 + g_terminal)
    if discount_rate <= g_terminal:
        # Evitar división por 0 o valores raros
        terminal_value = 0.0
    else:
        terminal_value = fcf_terminal * (1 + 0.0) / (discount_rate - g_terminal)

    pv_terminal = terminal_value / ((1 + discount_rate) ** horizon)

    enterprise_value = pv_flows + pv_terminal

    return {
        "cash_flows": cash_flows,
        "pv_flows": pv_flows,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "horizon_years": horizon,
    }


# ---------- Endpoint principal: quick valuation ----------


@router.post(
    "/{ticker}",
    response_model=QuickValuationResponse,
)
def quick_valuation(
    ticker: Ticker = Path(..., description="Ticker de la empresa, ej: AAPL, NVDA, AMZN"),
    payload: QuickValuationRequest | None = None,
) -> QuickValuationResponse:
    """
    Valuación rápida tipo DCF:
    - Usa FMP para precio, FCF, deuda neta y acciones.
    - Aplica un DCF simple de dos etapas + valor terminal.
    - Devuelve fair value por acción, upside y explicación del método/suposiciones.
    """

    method = (payload.method if payload else "dcf").lower()
    if method != "dcf":
        raise HTTPException(
            status_code=400,
            detail=f"Método '{method}' no soportado todavía. Usa 'dcf'.",
        )

    # 1) Obtener fundamentales desde FMP
    fundamentals = _fetch_basic_fundamentals(str(ticker))
    price = fundamentals["price"]
    market_cap = fundamentals["market_cap"]
    fcf = fundamentals["fcf"]
    shares_outstanding = fundamentals["shares_outstanding"]
    net_debt_from_data = fundamentals["net_debt"]

    if price <= 0 or market_cap <= 0:
        raise HTTPException(
            status_code=502,
            detail=f"Datos de precio/market cap no válidos para {ticker}.",
        )

    if fcf is None:
        # Sin FCF no tiene sentido forzar un DCF
        raise HTTPException(
            status_code=502,
            detail=f"No se encontró Free Cash Flow reciente para {ticker}.",
        )

    # 2) Construir supuestos usados (mezcla de defaults y datos)
    default_assumptions = DcfAssumptions()
    user_assumptions = payload.assumptions if payload and payload.assumptions else None

    assumptions = DcfAssumptions(
        fcf_yield_start=user_assumptions.fcf_yield_start
        if user_assumptions
        else default_assumptions.fcf_yield_start,
        high_growth_rate=user_assumptions.high_growth_rate
        if user_assumptions
        else default_assumptions.high_growth_rate,
        high_growth_years=user_assumptions.high_growth_years
        if user_assumptions
        else default_assumptions.high_growth_years,
        fade_years=user_assumptions.fade_years
        if user_assumptions
        else default_assumptions.fade_years,
        terminal_growth_rate=user_assumptions.terminal_growth_rate
        if user_assumptions
        else default_assumptions.terminal_growth_rate,
        discount_rate=user_assumptions.discount_rate
        if user_assumptions
        else default_assumptions.discount_rate,
        horizon_years=user_assumptions.horizon_years
        if user_assumptions
        else default_assumptions.horizon_years,
        shares_outstanding=user_assumptions.shares_outstanding
        if user_assumptions and user_assumptions.shares_outstanding is not None
        else (shares_outstanding if shares_outstanding is not None else None),
        net_debt=user_assumptions.net_debt
        if user_assumptions and user_assumptions.net_debt is not None
        else net_debt_from_data,
    )

    # 3) Correr DCF
    dcf_result = _run_simple_dcf(fcf_current=fcf, assumptions=assumptions)
    enterprise_value = dcf_result["enterprise_value"]

    # 4) Pasar a equity y fair value por acción
    net_debt = assumptions.net_debt or 0.0
    equity_value = enterprise_value - net_debt

    if not assumptions.shares_outstanding or assumptions.shares_outstanding <= 0:
        fair_value = None
        upside_pct = None
    else:
        fair_value = equity_value / assumptions.shares_outstanding
        if fair_value <= 0:
            upside_pct = None
        else:
            upside_pct = (fair_value / price - 1.0) * 100.0

    # 5) Explicación (texto para el usuario)
    explanation = (
        f"Metodología: DCF de dos etapas + valor terminal.\n\n"
        f"- Punto de partida: último Free Cash Flow anual reportado en FMP.\n"
        f"- Etapa de alto crecimiento: {assumptions.high_growth_years} años al "
        f"{assumptions.high_growth_rate*100:.1f}% anual.\n"
        f"- Etapa de 'fade': {assumptions.fade_years} años donde el crecimiento decae "
        f"linealmente hasta {assumptions.terminal_growth_rate*100:.1f}%.\n"
        f"- Horizonte explícito: {dcf_result['horizon_years']} años.\n"
        f"- Tasa de descuento: {assumptions.discount_rate*100:.1f}%.\n"
        f"- Valor terminal: FCF del último año creciendo al {assumptions.terminal_growth_rate*100:.1f}% "
        f"y descontado al {assumptions.discount_rate*100:.1f}%.\n"
        f"- Enterprise value: suma de FCF descontados + valor terminal.\n"
        f"- Equity value: enterprise value - deuda neta (deuda total - caja).\n"
        f"- Fair value por acción: equity value / acciones en circulación.\n\n"
        f"Esta es una aproximación educativa y simplificada; no reemplaza un análisis profundo de negocio, "
        f"estructura de capital ni de riesgos específicos."
    )

    assumptions_used = assumptions.dict()

    raw_inputs = {
        "fundamentals": fundamentals,
        "dcf_result": dcf_result,
    }

    return QuickValuationResponse(
        ticker=str(ticker).upper(),
        current_price=price,
        fair_value=fair_value,
        upside_pct=upside_pct,
        method="dcf",
        assumptions_used=assumptions_used,
        explanation=explanation,
        raw_inputs=raw_inputs,
    )
