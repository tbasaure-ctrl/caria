"""Valuación por múltiplos comparables para empresas con ingresos."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("caria.models.valuation.multiples")


@dataclass
class MultiplesValuation:
    """Resultado de valuación por múltiplos."""
    ticker: str
    fair_value_per_share: float
    current_price: float
    upside_downside: float
    method_used: str  # "P/S", "EV/Revenue", "P/E", etc.
    comparable_multiple: float  # Múltiplo del sector/comparable
    company_metric: float  # Métrica de la empresa (revenue, earnings, etc.)
    explanation: str


class MultiplesValuator:
    """Valuador por múltiplos comparables.

    Apropiado para:
    - Empresas con ingresos pero FCF negativo
    - Empresas en crecimiento sin earnings positivos
    - Empresas en industrias donde múltiplos son estándar (SaaS, retail, etc.)
    """

    # Múltiplos promedio por sector (EV/Revenue)
    # Fuente: Datos históricos de industria, actualizar periódicamente
    SECTOR_MULTIPLES = {
        "software": 8.0,  # SaaS companies típicamente 5-15x revenue
        "technology": 4.0,
        "healthcare": 3.0,
        "consumer": 2.0,
        "retail": 1.5,
        "industrial": 1.8,
        "financial": 2.5,
        "energy": 1.2,
        "utilities": 2.0,
        "telecom": 1.5,
        "default": 2.5,  # Múltiplo conservador por defecto
    }

    def __init__(self) -> None:
        """Inicializa el valuador de múltiplos."""
        pass

    def _get_sector_multiple(self, sector: str | None) -> float:
        """Obtiene múltiplo promedio del sector."""
        if not sector:
            return self.SECTOR_MULTIPLES["default"]

        sector_lower = sector.lower()
        for key, multiple in self.SECTOR_MULTIPLES.items():
            if key in sector_lower:
                return multiple

        return self.SECTOR_MULTIPLES["default"]

    def value_by_revenue_multiple(
        self,
        ticker: str,
        annual_revenue: float,
        shares_outstanding: float,
        current_price: float,
        total_debt: float = 0.0,
        cash_and_equivalents: float = 0.0,
        sector: str | None = None,
        custom_multiple: float | None = None,
    ) -> MultiplesValuation:
        """Valúa empresa usando múltiplo EV/Revenue.

        Args:
            ticker: Símbolo de la empresa
            annual_revenue: Ingresos anuales (millones USD)
            shares_outstanding: Acciones en circulación (millones)
            current_price: Precio actual por acción
            total_debt: Deuda total (millones USD)
            cash_and_equivalents: Efectivo (millones USD)
            sector: Sector de la empresa (para seleccionar múltiplo apropiado)
            custom_multiple: Múltiplo personalizado (sobrescribe sector)

        Returns:
            MultiplesValuation con valuación estimada
        """
        # Validaciones
        if annual_revenue <= 0:
            raise ValueError(f"Revenue debe ser positivo para valuación por múltiplos, recibido: {annual_revenue}")

        if shares_outstanding <= 0:
            raise ValueError(f"shares_outstanding debe ser positivo, recibido: {shares_outstanding}")

        # Determinar múltiplo a usar
        if custom_multiple is not None:
            ev_revenue_multiple = custom_multiple
            LOGGER.info("Usando múltiplo personalizado para %s: %.2fx", ticker, ev_revenue_multiple)
        else:
            ev_revenue_multiple = self._get_sector_multiple(sector)
            LOGGER.info("Usando múltiplo de sector '%s' para %s: %.2fx", sector or "default", ticker, ev_revenue_multiple)

        # Calcular Enterprise Value
        enterprise_value = annual_revenue * ev_revenue_multiple

        # Calcular Equity Value
        net_debt = total_debt - cash_and_equivalents
        equity_value = enterprise_value - net_debt

        # Valor por acción
        fair_value_per_share = equity_value / shares_outstanding

        # Upside/Downside
        upside_downside = ((fair_value_per_share - current_price) / current_price) * 100

        # Explicación
        explanation = self._generate_explanation(
            ticker=ticker,
            fair_value=fair_value_per_share,
            current_price=current_price,
            upside_downside=upside_downside,
            method="EV/Revenue",
            multiple=ev_revenue_multiple,
            metric_value=annual_revenue,
            metric_name="revenue anual",
            sector=sector,
        )

        return MultiplesValuation(
            ticker=ticker,
            fair_value_per_share=fair_value_per_share,
            current_price=current_price,
            upside_downside=upside_downside,
            method_used="EV/Revenue",
            comparable_multiple=ev_revenue_multiple,
            company_metric=annual_revenue,
            explanation=explanation,
        )

    def value_by_ps_ratio(
        self,
        ticker: str,
        annual_revenue: float,
        shares_outstanding: float,
        current_price: float,
        sector: str | None = None,
        custom_multiple: float | None = None,
    ) -> MultiplesValuation:
        """Valúa empresa usando ratio P/S (Price-to-Sales).

        Simplificación de EV/Revenue que no considera deuda.
        Útil para comparación rápida.

        Args:
            ticker: Símbolo
            annual_revenue: Ingresos anuales (millones USD)
            shares_outstanding: Acciones en circulación (millones)
            current_price: Precio actual
            sector: Sector para seleccionar múltiplo
            custom_multiple: Múltiplo P/S personalizado

        Returns:
            MultiplesValuation
        """
        # Validaciones
        if annual_revenue <= 0:
            raise ValueError(f"Revenue debe ser positivo, recibido: {annual_revenue}")

        if shares_outstanding <= 0:
            raise ValueError(f"shares_outstanding debe ser positivo, recibido: {shares_outstanding}")

        # P/S típicamente es ~0.8x de EV/Revenue (aproximado)
        if custom_multiple is not None:
            ps_multiple = custom_multiple
        else:
            ev_multiple = self._get_sector_multiple(sector)
            ps_multiple = ev_multiple * 0.8  # Ajuste heurístico

        # Revenue por acción
        revenue_per_share = annual_revenue / shares_outstanding

        # Valor por acción
        fair_value_per_share = revenue_per_share * ps_multiple

        # Upside/Downside
        upside_downside = ((fair_value_per_share - current_price) / current_price) * 100

        # Explicación
        explanation = self._generate_explanation(
            ticker=ticker,
            fair_value=fair_value_per_share,
            current_price=current_price,
            upside_downside=upside_downside,
            method="P/S",
            multiple=ps_multiple,
            metric_value=annual_revenue,
            metric_name="revenue anual",
            sector=sector,
        )

        return MultiplesValuation(
            ticker=ticker,
            fair_value_per_share=fair_value_per_share,
            current_price=current_price,
            upside_downside=upside_downside,
            method_used="P/S",
            comparable_multiple=ps_multiple,
            company_metric=annual_revenue,
            explanation=explanation,
        )

    def _generate_explanation(
        self,
        ticker: str,
        fair_value: float,
        current_price: float,
        upside_downside: float,
        method: str,
        multiple: float,
        metric_value: float,
        metric_name: str,
        sector: str | None,
    ) -> str:
        """Genera explicación de valuación."""
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

        sector_text = f" del sector {sector}" if sector else ""

        explanation = (
            f"{ticker} está {valuation_status} según valuación por {method}. "
            f"Usando un múltiplo {method} de {multiple:.2f}x{sector_text} "
            f"y {metric_name} de ${metric_value:.2f}M, "
            f"el valor justo estimado es ${fair_value:.2f} por acción "
            f"vs precio actual de ${current_price:.2f} "
            f"(diferencia: {upside_downside:+.1f}%)."
        )

        return explanation


class ComparableCompaniesAnalysis:
    """Análisis de empresas comparables para obtener múltiplos dinámicos.

    Este método es más sofisticado que usar múltiplos de sector fijos.
    Busca empresas similares y calcula múltiplos medianos.
    """

    def __init__(self, comparables_df: pd.DataFrame) -> None:
        """Inicializa con DataFrame de empresas comparables.

        Args:
            comparables_df: DataFrame con columnas:
                - ticker
                - sector
                - revenue
                - market_cap
                - enterprise_value
                - total_debt
                - cash
        """
        self.comparables_df = comparables_df

    def get_peer_multiple(
        self,
        ticker: str,
        sector: str,
        metric: str = "ev_revenue",
    ) -> float:
        """Obtiene múltiplo mediano de peers en el mismo sector.

        Args:
            ticker: Ticker de la empresa a valuar (se excluye de peers)
            sector: Sector de la empresa
            metric: Métrica ("ev_revenue", "p_s", "p_e", etc.)

        Returns:
            Múltiplo mediano de peers
        """
        # Filtrar peers del mismo sector
        peers = self.comparables_df[
            (self.comparables_df["sector"] == sector) &
            (self.comparables_df["ticker"] != ticker)
        ].copy()

        if len(peers) == 0:
            LOGGER.warning("No se encontraron peers para %s en sector %s", ticker, sector)
            return MultiplesValuator.SECTOR_MULTIPLES.get(sector.lower(), 2.5)

        # Calcular múltiplo según métrica
        if metric == "ev_revenue":
            peers["multiple"] = peers["enterprise_value"] / peers["revenue"]
        elif metric == "p_s":
            peers["multiple"] = peers["market_cap"] / peers["revenue"]
        else:
            raise ValueError(f"Métrica no soportada: {metric}")

        # Filtrar outliers (fuera de percentiles 10-90)
        p10 = peers["multiple"].quantile(0.10)
        p90 = peers["multiple"].quantile(0.90)
        peers_filtered = peers[(peers["multiple"] >= p10) & (peers["multiple"] <= p90)]

        # Múltiplo mediano
        median_multiple = peers_filtered["multiple"].median()

        LOGGER.info(
            "Múltiplo %s para %s (sector %s): %.2fx (de %d peers)",
            metric, ticker, sector, median_multiple, len(peers_filtered)
        )

        return float(median_multiple)
