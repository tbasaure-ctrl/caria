"""
Módulo de valuación para análisis cualitativo.

NO retorna scores numéricos al usuario.
SÍ retorna contexto cualitativo: múltiplos vs histórico, DCF ranges, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


class ValuationAnalyzer:
    """Analiza valuación usando DCF y múltiplos, retorna insights cualitativos."""

    def __init__(self):
        # Cargar datos históricos para contexto
        base_dir = Path(__file__).resolve().parents[2]
        try:
            self.hist_data = pd.read_parquet(base_dir / "data/gold/train.parquet")
        except:
            self.hist_data = None

    def analyze(
        self,
        ticker: str,
        current_price: float,
        fcf_per_share: Optional[float] = None,
        revenue: Optional[float] = None,
        earnings: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analiza valuación y retorna insights cualitativos.

        Returns:
            {
                "multiples_context": "P/E 28x vs histórico 18x (percentil 85)...",
                "dcf_range": "DCF sugiere valor intrínseco $150-180...",
                "relative_valuation": "Premium de 40% vs sector...",
                "key_drivers": ["Services growth 15%", "Margins expanding"],
                "risks": ["Competencia", "Multiples contraídos en recesión"]
            }
        """

        # Por ahora, retornar placeholder con estructura correcta
        # TODO: Implementar cálculos reales

        result = {
            "multiples_context": self._get_multiples_context(ticker, current_price),
            "dcf_range": self._get_dcf_range(ticker, fcf_per_share),
            "relative_valuation": self._get_relative_valuation(ticker),
            "key_drivers": self._get_key_drivers(ticker),
            "risks": self._get_valuation_risks(ticker)
        }

        return result

    def _get_multiples_context(self, ticker: str, current_price: float) -> str:
        """
        Analiza múltiplos actuales vs históricos.

        Retorna contexto cualitativo, NO scores.
        """
        if self.hist_data is None:
            return f"{ticker}: Múltiplos no disponibles (datos históricos faltantes)"

        ticker_data = self.hist_data[self.hist_data['ticker'] == ticker]

        if len(ticker_data) == 0:
            return f"{ticker}: Sin datos históricos para comparar múltiplos"

        # Analizar P/B ratio si está disponible
        if 'priceToBookRatio' in ticker_data.columns:
            current_pb = ticker_data['priceToBookRatio'].iloc[-1]
            hist_pb_median = ticker_data['priceToBookRatio'].median()
            percentile = (ticker_data['priceToBookRatio'] <= current_pb).mean() * 100

            if pd.notna(current_pb) and pd.notna(hist_pb_median):
                return (
                    f"{ticker} P/B: {current_pb:.1f}x vs histórico {hist_pb_median:.1f}x "
                    f"(percentil {percentile:.0f}). "
                    f"{'Premium justificado si ROE > coste capital' if current_pb > hist_pb_median else 'Descuento vs histórico'}"
                )

        return f"{ticker}: Análisis de múltiplos en progreso"

    def _get_dcf_range(self, ticker: str, fcf_per_share: Optional[float]) -> str:
        """
        Calcula rango de valor intrínseco con DCF simplificado.

        Escenarios:
        - Base: FCF crece 5% anual
        - Bull: FCF crece 10% anual
        - Bear: FCF crece 0% (flat)

        Discount rate: 10% (conservador)
        """
        # Si no se provee FCF, intentar extraerlo de los datos históricos
        if fcf_per_share is None or pd.isna(fcf_per_share):
            if self.hist_data is not None:
                ticker_data = self.hist_data[self.hist_data['ticker'] == ticker]
                if len(ticker_data) > 0 and 'freeCashFlowPerShare' in ticker_data.columns:
                    fcf_per_share = ticker_data['freeCashFlowPerShare'].iloc[-1]

        # Si aún no hay FCF disponible
        if fcf_per_share is None or pd.isna(fcf_per_share):
            return f"{ticker}: DCF requiere FCF actual (dato no disponible)"

        # Parámetros DCF
        discount_rate = 0.10
        terminal_growth = 0.03
        years = 10

        # Escenarios de growth
        scenarios = {
            'bear': 0.00,
            'base': 0.05,
            'bull': 0.10
        }

        dcf_values = {}

        for scenario, growth_rate in scenarios.items():
            # Proyectar FCF
            fcf_projection = [fcf_per_share * (1 + growth_rate) ** i for i in range(1, years + 1)]

            # Descontar flujos
            pv_fcf = sum([fcf / (1 + discount_rate) ** i for i, fcf in enumerate(fcf_projection, 1)])

            # Terminal value
            terminal_fcf = fcf_projection[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / (1 + discount_rate) ** years

            dcf_values[scenario] = pv_fcf + pv_terminal

        return (
            f"{ticker} DCF (10 años, 10% discount rate): "
            f"Bear ${dcf_values['bear']:.0f} | "
            f"Base ${dcf_values['base']:.0f} | "
            f"Bull ${dcf_values['bull']:.0f}. "
            f"Asume growth sostenible y márgenes estables."
        )

    def _get_relative_valuation(self, ticker: str) -> str:
        """Compara valuación vs sector/mercado."""
        # TODO: Implementar comparación vs sector
        return f"{ticker}: Comparación vs sector (próximamente con datos de industria)"

    def _get_key_drivers(self, ticker: str) -> list[str]:
        """Identifica drivers clave de valor."""
        # TODO: Analizar datos históricos para identificar drivers
        return [
            "Análisis de drivers en progreso",
            "Revisar: growth de revenue, expansión de márgenes, ROIC"
        ]

    def _get_valuation_risks(self, ticker: str) -> list[str]:
        """Identifica riesgos de valuación."""
        return [
            "Múltiplos contraídos en recesión (históricamente -30 a -50%)",
            "Crecimiento más lento que esperado",
            "Compresión de márgenes por competencia"
        ]


def calculate_wacc(
    risk_free_rate: float = 0.04,
    equity_risk_premium: float = 0.06,
    beta: float = 1.0,
    debt_to_equity: float = 0.0,
    tax_rate: float = 0.21
) -> float:
    """
    Calcula WACC (Weighted Average Cost of Capital).

    WACC = E/V * Re + D/V * Rd * (1-Tc)

    Donde:
    - Re = Cost of equity (CAPM)
    - Rd = Cost of debt
    - E/V = % equity
    - D/V = % debt
    - Tc = Tax rate
    """
    # Cost of equity (CAPM)
    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    # Cost of debt (simplified: risk_free + spread)
    cost_of_debt = risk_free_rate + 0.02  # 2% spread

    # Weights
    total_value = 1 + debt_to_equity
    weight_equity = 1 / total_value
    weight_debt = debt_to_equity / total_value

    wacc = (
        weight_equity * cost_of_equity +
        weight_debt * cost_of_debt * (1 - tax_rate)
    )

    return wacc


def calculate_intrinsic_value_dcf(
    current_fcf: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth: float = 0.03,
    years: int = 10
) -> float:
    """
    Calcula valor intrínseco con DCF de 2 etapas.

    Etapa 1: Growth rate específico (years)
    Etapa 2: Terminal growth perpetuo (terminal_growth)
    """
    # Stage 1: High growth
    fcf_projection = [current_fcf * (1 + growth_rate) ** i for i in range(1, years + 1)]
    pv_fcf = sum([fcf / (1 + discount_rate) ** i for i, fcf in enumerate(fcf_projection, 1)])

    # Stage 2: Terminal value
    terminal_fcf = fcf_projection[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** years

    intrinsic_value = pv_fcf + pv_terminal

    return intrinsic_value
