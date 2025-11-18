"""Clasificador de etapa de empresa para determinar método de valuación."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

LOGGER = logging.getLogger("caria.models.valuation.classifier")


@dataclass
class CompanyStage:
    """Etapa de la empresa."""
    ticker: str
    stage: str  # "consolidated" o "pre_revenue"
    revenue: float
    age_years: float | None = None
    confidence: float = 1.0


class CompanyClassifier:
    """Clasifica empresas en consolidadas vs pre-revenue."""
    
    def classify(
        self,
        ticker: str,
        revenue: float,
        age_years: float | None = None,
    ) -> CompanyStage:
        """Clasifica una empresa según su etapa.
        
        Criterios:
        - Pre-revenue: revenue == 0 OR age < 3 años
        - Consolidated: revenue > 0 AND age >= 3 años
        
        Args:
            ticker: Símbolo de la empresa
            revenue: Ingresos anuales (millones USD)
            age_years: Edad de la empresa en años (opcional)
            
        Returns:
            CompanyStage con clasificación
        """
        # Si no hay ingresos, es pre-revenue
        if revenue == 0 or pd.isna(revenue):
            return CompanyStage(
                ticker=ticker,
                stage="pre_revenue",
                revenue=0.0,
                age_years=age_years,
                confidence=1.0,
            )
        
        # Si es muy joven (< 3 años), probablemente pre-revenue o early-stage
        if age_years is not None and age_years < 3:
            return CompanyStage(
                ticker=ticker,
                stage="pre_revenue",
                revenue=revenue,
                age_years=age_years,
                confidence=0.8,
            )
        
        # Empresa consolidada
        return CompanyStage(
            ticker=ticker,
            stage="consolidated",
            revenue=revenue,
            age_years=age_years,
            confidence=1.0,
        )

