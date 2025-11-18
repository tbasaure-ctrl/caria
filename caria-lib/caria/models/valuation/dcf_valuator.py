"""Modelo DCF (Discounted Cash Flow) para empresas consolidadas.

Este modelo ajusta dinámicamente el WACC según el régimen macro detectado
por el Sistema I (HMM), y puede usar proyecciones de FCF basadas en NLP
del Sistema II (RAG) para earnings calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("caria.models.valuation.dcf")


@dataclass
class DCFValuation:
    """Resultado de valuación DCF."""
    ticker: str
    fair_value_per_share: float
    current_price: float
    upside_downside: float  # Porcentaje de upside/downside
    wacc: float
    terminal_value: float
    pv_fcf: float  # Valor presente de FCF proyectados
    explanation: str  # Explicación simple de por qué es caro/barato


class DCFValuator:
    """Valuador DCF con ajuste dinámico de WACC según régimen macro."""
    
    # WACC base por régimen macro
    REGIME_WACC_ADJUSTMENTS = {
        "expansion": -0.005,  # WACC más bajo en expansión
        "slowdown": 0.0,       # WACC neutral
        "recession": 0.015,    # WACC más alto en recesión
        "stress": 0.025,       # WACC mucho más alto en estrés
    }
    
    def __init__(
        self,
        base_wacc: float = 0.10,
        terminal_growth: float = 0.03,
        projection_years: int = 5,
    ) -> None:
        """Inicializa el valuador DCF.
        
        Args:
            base_wacc: WACC base (10% por defecto)
            terminal_growth: Tasa de crecimiento terminal (3% por defecto)
            projection_years: Años de proyección (5 por defecto)
        """
        self.base_wacc = base_wacc
        self.terminal_growth = terminal_growth
        self.projection_years = projection_years
    
    def _adjust_wacc(self, regime: str | None) -> float:
        """Ajusta WACC según régimen macro."""
        adjustment = self.REGIME_WACC_ADJUSTMENTS.get(regime or "slowdown", 0.0)
        return self.base_wacc + adjustment
    
    def _project_fcf(
        self,
        current_fcf: float,
        growth_rate: float,
        years: int,
    ) -> list[float]:
        """Proyecta FCF futuro con tasa de crecimiento constante."""
        projections = []
        fcf = current_fcf
        for _ in range(years):
            fcf *= (1 + growth_rate)
            projections.append(fcf)
        return projections
    
    def _calculate_terminal_value(
        self,
        final_fcf: float,
        wacc: float,
        growth_rate: float,
    ) -> float:
        """Calcula valor terminal usando modelo de crecimiento perpetuo."""
        if wacc <= growth_rate:
            # Ajustar si WACC es muy bajo
            wacc = growth_rate + 0.01
        return final_fcf * (1 + growth_rate) / (wacc - growth_rate)
    
    def value(
        self,
        ticker: str,
        current_fcf: float,
        shares_outstanding: float,
        current_price: float,
        total_debt: float = 0.0,
        cash_and_equivalents: float = 0.0,
        regime: str | None = None,
        fcf_growth_rate: float | None = None,
        nlp_projection: float | None = None,
    ) -> DCFValuation:
        """Valúa una empresa usando DCF.

        Args:
            ticker: Símbolo de la empresa
            current_fcf: FCF actual (millones USD)
            shares_outstanding: Acciones en circulación (millones)
            current_price: Precio actual por acción
            total_debt: Deuda total (millones USD)
            cash_and_equivalents: Efectivo y equivalentes (millones USD)
            regime: Régimen macro (para ajustar WACC)
            fcf_growth_rate: Tasa de crecimiento de FCF (si None, estima desde histórico)
            nlp_projection: Proyección de crecimiento desde NLP (Sistema II)

        Returns:
            DCFValuation con valor justo y explicación

        Raises:
            ValueError: Si FCF es negativo o shares_outstanding es cero
        """
        # Validaciones
        if shares_outstanding <= 0:
            raise ValueError(f"shares_outstanding debe ser positivo, recibido: {shares_outstanding}")

        # Manejar FCF negativo
        if current_fcf <= 0:
            LOGGER.warning(
                "FCF negativo para %s (%.2fM). DCF no es apropiado para empresas con FCF negativo. "
                "Considera usar valuación por múltiplos o Scorecard Method.",
                ticker, current_fcf
            )
            # Retornar valuación indicando que DCF no es aplicable
            return DCFValuation(
                ticker=ticker,
                fair_value_per_share=0.0,
                current_price=current_price,
                upside_downside=-100.0,
                wacc=self._adjust_wacc(regime),
                terminal_value=0.0,
                pv_fcf=0.0,
                explanation=f"{ticker} tiene FCF negativo (${current_fcf:.2f}M). "
                           f"DCF no es apropiado. Recomienda usar valuación por múltiplos o Scorecard Method para empresas pre-revenue."
            )
        # Ajustar WACC según régimen
        wacc = self._adjust_wacc(regime)
        
        # Determinar tasa de crecimiento
        if nlp_projection is not None:
            # Usar proyección de NLP si está disponible
            growth_rate = nlp_projection
            LOGGER.info("Usando proyección NLP para %s: %.2f%%", ticker, growth_rate * 100)
        elif fcf_growth_rate is not None:
            growth_rate = fcf_growth_rate
        else:
            # Estimación conservadora por defecto
            growth_rate = 0.05  # 5% anual
            LOGGER.warning("Usando crecimiento por defecto para %s: %.2f%%", ticker, growth_rate * 100)
        
        # Proyectar FCF
        fcf_projections = self._project_fcf(current_fcf, growth_rate, self.projection_years)
        
        # Calcular valor presente de FCF proyectados
        pv_fcf = sum([
            fcf / ((1 + wacc) ** (i + 1))
            for i, fcf in enumerate(fcf_projections)
        ])
        
        # Calcular valor terminal
        final_fcf = fcf_projections[-1]
        terminal_value = self._calculate_terminal_value(final_fcf, wacc, self.terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** self.projection_years)

        # Valor de la empresa (Enterprise Value)
        enterprise_value = pv_fcf + pv_terminal

        # CORREGIDO: Calcular Equity Value = Enterprise Value - Net Debt
        net_debt = total_debt - cash_and_equivalents
        equity_value = enterprise_value - net_debt

        # Valor por acción = Equity Value / Shares Outstanding
        fair_value_per_share = equity_value / shares_outstanding

        # Log información del cálculo
        LOGGER.info(
            "%s DCF: EV=%.2fM, Debt=%.2fM, Cash=%.2fM, NetDebt=%.2fM, EquityValue=%.2fM",
            ticker, enterprise_value, total_debt, cash_and_equivalents, net_debt, equity_value
        )

        # Calcular upside/downside
        upside_downside = ((fair_value_per_share - current_price) / current_price) * 100
        
        # Generar explicación simple
        explanation = self._generate_explanation(
            ticker,
            fair_value_per_share,
            current_price,
            upside_downside,
            wacc,
            regime,
        )
        
        return DCFValuation(
            ticker=ticker,
            fair_value_per_share=fair_value_per_share,
            current_price=current_price,
            upside_downside=upside_downside,
            wacc=wacc,
            terminal_value=pv_terminal,
            pv_fcf=pv_fcf,
            explanation=explanation,
        )
    
    def _generate_explanation(
        self,
        ticker: str,
        fair_value: float,
        current_price: float,
        upside_downside: float,
        wacc: float,
        regime: str | None,
    ) -> str:
        """Genera explicación simple de por qué es caro/barato."""
        if upside_downside > 20:
            valuation_status = "significativamente infravalorada"
        elif upside_downside > 5:
            valuation_status = "ligeramente infravalorada"
        elif upside_downside < -20:
            valuation_status = "significativamente sobrevalorada"
        elif upside_downside < -5:
            valuation_status = "ligeramente sobrevalorada"
        else:
            valuation_status = "razonablemente valorada"
        
        regime_context = ""
        if regime:
            regime_context = f" En el contexto de régimen macro '{regime}', "
        
        explanation = (
            f"{ticker} está {valuation_status} según nuestro modelo DCF. "
            f"{regime_context}"
            f"El precio actual es ${current_price:.2f} vs valor justo estimado de ${fair_value:.2f} "
            f"(diferencia: {upside_downside:+.1f}%). "
            f"El WACC utilizado es {wacc*100:.1f}%, ajustado según el régimen macro actual."
        )
        
        return explanation

