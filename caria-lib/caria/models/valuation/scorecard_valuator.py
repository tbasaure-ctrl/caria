"""Modelo Scorecard/Berkus para valuación de empresas pre-revenue/early-stage.

Este módulo implementa dos metodologías para startups sin ingresos:
1. Scorecard Method: Evalúa factores cualitativos ponderados
2. Berkus Method: Asigna valor por cada factor de riesgo mitigado

Apropiado para empresas que:
- No tienen ingresos (pre-revenue)
- Están en etapa temprana (pre-seed, seed)
- DCF y múltiplos no son aplicables
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger("caria.models.valuation.scorecard")


@dataclass
class ScorecardFactors:
    """Factores evaluados en Scorecard Method."""
    team_quality: float  # 0-10: Experiencia y track record del equipo
    technology: float  # 0-10: Innovación y ventaja tecnológica
    market_opportunity: float  # 0-10: Tamaño y crecimiento del mercado
    product_progress: float  # 0-10: Estado del producto (concepto → MVP → producto)
    traction: float  # 0-10: Early traction (usuarios, pilotos, LOIs)
    fundraising: float  # 0-10: Capacidad de levantar capital
    go_to_market: float  # 0-10: Estrategia de GTM y sales

    def total_score(self) -> float:
        """Calcula score total (0-70)."""
        return (
            self.team_quality +
            self.technology +
            self.market_opportunity +
            self.product_progress +
            self.traction +
            self.fundraising +
            self.go_to_market
        )

    def normalized_score(self) -> float:
        """Score normalizado (0-1)."""
        return self.total_score() / 70.0


@dataclass
class ScorecardValuation:
    """Resultado de valuación Scorecard."""
    ticker: str
    estimated_value: float  # Valuación en millones USD
    base_value: float  # Valor base antes de ajustes
    scorecard_scores: ScorecardFactors | dict[str, float]  # Scores por factor
    confidence: float  # 0-1: Confianza en la estimación
    stage: str  # "pre-seed", "seed", etc.
    explanation: str


class ScorecardValuator:
    """Valuador Scorecard para empresas pre-revenue.

    Referencias:
    - Scorecard Method: Compara con valuaciones promedio de industria/etapa
    - Ajusta por factores cualitativos (equipo, tecnología, mercado, etc.)
    """

    # Valuaciones promedio por etapa (millones USD)
    STAGE_VALUATIONS = {
        "pre-seed": {"min": 1.0, "median": 3.0, "max": 8.0},
        "seed": {"min": 3.0, "median": 8.0, "max": 20.0},
        "series-a": {"min": 10.0, "median": 25.0, "max": 60.0},
        "series-b": {"min": 25.0, "median": 60.0, "max": 150.0},
    }

    # Multiplicadores por sector
    SECTOR_MULTIPLIERS = {
        "ai": 1.5,
        "biotech": 1.4,
        "fintech": 1.3,
        "saas": 1.2,
        "ecommerce": 1.0,
        "consumer": 0.9,
        "default": 1.0,
    }

    def __init__(self) -> None:
        """Inicializa el valuador Scorecard."""
        pass
    
    def _score_factor(self, value: float, max_value: float = 1.0) -> float:
        """Normaliza un score de factor a [0, 1]."""
        return min(max(value / max_value, 0.0), 1.0)
    
    def value(
        self,
        ticker: str,
        factors: ScorecardFactors,
        stage: str,
        sector: str | None = None,
        recent_funding_valuation: float | None = None,
    ) -> ScorecardValuation:
        """Valúa empresa pre-revenue usando Scorecard Method.

        Args:
            ticker: Símbolo/nombre de la empresa
            factors: Factores evaluados (scores 0-10)
            stage: Etapa ("pre-seed", "seed", "series-a", "series-b")
            sector: Sector/industria (opcional)
            recent_funding_valuation: Valuación en último round (opcional, en millones USD)

        Returns:
            ScorecardValuation con estimación
        """
        # Validar factors
        total_score = factors.total_score()
        if total_score < 0 or total_score > 70:
            raise ValueError(f"Score total debe estar en [0, 70], recibido: {total_score}")

        # Obtener valuación base de la etapa
        stage_lower = stage.lower().replace(" ", "-")
        base_valuation = self.STAGE_VALUATIONS.get(stage_lower, self.STAGE_VALUATIONS["seed"])

        # Score normalizado (0-1)
        normalized_score = factors.normalized_score()

        # Valuación interpolada entre min y max según score
        if normalized_score < 0.5:
            t = normalized_score / 0.5
            stage_valuation = base_valuation["min"] + t * (base_valuation["median"] - base_valuation["min"])
        else:
            t = (normalized_score - 0.5) / 0.5
            stage_valuation = base_valuation["median"] + t * (base_valuation["max"] - base_valuation["median"])

        # Aplicar multiplicador de sector
        sector_multiplier = self._get_sector_multiplier(sector)
        estimated_value = stage_valuation * sector_multiplier

        # Si hay valuación reciente, promediar
        if recent_funding_valuation is not None and recent_funding_valuation > 0:
            LOGGER.info(
                "Averaging scorecard (%.2fM) with recent funding (%.2fM)",
                estimated_value, recent_funding_valuation
            )
            estimated_value = 0.6 * recent_funding_valuation + 0.4 * estimated_value

        # Calcular confianza
        confidence = self._calculate_confidence(normalized_score, stage, recent_funding_valuation is not None)

        # Explicación
        explanation = self._generate_explanation(ticker, factors, estimated_value, stage, sector, confidence)

        return ScorecardValuation(
            ticker=ticker,
            estimated_value=estimated_value,
            base_value=base_valuation["median"],
            scorecard_scores=factors,
            confidence=confidence,
            stage=stage,
            explanation=explanation,
        )

    def _get_sector_multiplier(self, sector: str | None) -> float:
        """Obtiene multiplicador de sector."""
        if not sector:
            return 1.0
        sector_lower = sector.lower()
        for key, mult in self.SECTOR_MULTIPLIERS.items():
            if key in sector_lower:
                return mult
        return 1.0

    def _calculate_confidence(self, normalized_score: float, stage: str, has_funding: bool) -> float:
        """Calcula confianza (0-1)."""
        confidence = 0.5
        if has_funding:
            confidence += 0.2
        stage_boost = {"pre-seed": 0.0, "seed": 0.05, "series-a": 0.1, "series-b": 0.15}
        confidence += stage_boost.get(stage.lower().replace(" ", "-"), 0.0)
        if normalized_score < 0.2 or normalized_score > 0.9:
            confidence -= 0.1
        return max(0.0, min(1.0, confidence))
    
    def _generate_explanation(
        self,
        ticker: str,
        factors: ScorecardFactors,
        estimated_value: float,
        stage: str,
        sector: str | None,
        confidence: float,
    ) -> str:
        """Genera explicación de la valuación."""
        total_score = factors.total_score()
        normalized_score = factors.normalized_score()

        # Top strengths
        factor_dict = {
            "Equipo": factors.team_quality,
            "Tecnología": factors.technology,
            "Mercado": factors.market_opportunity,
            "Producto": factors.product_progress,
            "Tracción": factors.traction,
            "Fundraising": factors.fundraising,
            "Go-to-Market": factors.go_to_market,
        }
        sorted_factors = sorted(factor_dict.items(), key=lambda x: x[1], reverse=True)
        top_strengths = ", ".join([f"{name} ({score:.1f}/10)" for name, score in sorted_factors[:3]])

        sector_text = f" del sector {sector}" if sector else ""

        explanation = (
            f"{ticker} (empresa pre-revenue en etapa {stage}{sector_text}) tiene una valuación estimada de "
            f"${estimated_value:.1f}M según Scorecard Method. "
            f"Score total: {total_score:.1f}/70 ({normalized_score*100:.0f}%). "
            f"Principales fortalezas: {top_strengths}. "
            f"Confianza: {confidence*100:.0f}%. "
            f"NOTA: Valuación altamente subjetiva, usar como referencia."
        )

        return explanation

