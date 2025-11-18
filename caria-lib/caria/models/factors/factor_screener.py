"""Motor de Factores Cuantitativos para screening de acciones.

Este módulo implementa factor investing basado en factores canónicos:
- Valor (Value): FCF Yield, P/B, P/E
- Rentabilidad (Profitability): ROIC, ROE
- Crecimiento (Growth): Revenue Growth, EPS Growth
- Solvencia (Solvency): Debt-to-Equity, Current Ratio
- Momentum: 12-month return, Price vs SMA

El screening es cross-sectional (no time-series), rankeando empresas por factores
en cada punto en el tiempo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("caria.models.factors")


@dataclass
class FactorScores:
    """Scores de factores para una empresa."""
    ticker: str
    date: pd.Timestamp
    value_score: float
    profitability_score: float
    growth_score: float
    solvency_score: float
    momentum_score: float
    composite_score: float
    rank: int


@dataclass
class FactorWeights:
    """Pesos de factores para scoring compuesto."""
    value: float = 0.25
    profitability: float = 0.25
    growth: float = 0.20
    solvency: float = 0.15
    momentum: float = 0.15
    
    def normalize(self) -> None:
        """Normaliza pesos para que sumen 1.0."""
        total = sum([
            self.value,
            self.profitability,
            self.growth,
            self.solvency,
            self.momentum,
        ])
        if total > 0:
            self.value /= total
            self.profitability /= total
            self.growth /= total
            self.solvency /= total
            self.momentum /= total


class FactorScreener:
    """Screener de factores cuantitativos para identificar empresas de calidad."""
    
    def __init__(
        self,
        weights: FactorWeights | None = None,
        regime_weights: dict[str, FactorWeights] | None = None,
    ) -> None:
        """Inicializa el screener de factores.
        
        Args:
            weights: Pesos de factores por defecto
            regime_weights: Pesos ajustados por régimen macro (opcional)
        """
        self.weights = weights or FactorWeights()
        self.weights.normalize()
        self.regime_weights = regime_weights or {}
    
    def _calculate_value_score(self, df: pd.DataFrame) -> pd.Series:
        """Calcula score de valor basado en múltiplos y FCF yield."""
        score = pd.Series(0.0, index=df.index)
        
        # FCF Yield (mayor es mejor)
        if "freeCashFlowPerShare" in df.columns and "close" in df.columns:
            fcf_yield = df["freeCashFlowPerShare"] / (df["close"] + 1e-6)
            score += self._rank_normalize(fcf_yield) * 0.4
        
        # P/B bajo es mejor (invertir)
        if "priceToBookRatio" in df.columns:
            pb_rank = self._rank_normalize(-df["priceToBookRatio"])  # Negativo porque menor es mejor
            score += pb_rank * 0.3
        
        # P/S bajo es mejor
        if "priceToSalesRatio" in df.columns:
            ps_rank = self._rank_normalize(-df["priceToSalesRatio"])
            score += ps_rank * 0.3
        
        return score
    
    def _calculate_profitability_score(self, df: pd.DataFrame) -> pd.Series:
        """Calcula score de rentabilidad basado en ROIC, ROE, márgenes."""
        score = pd.Series(0.0, index=df.index)
        
        # ROIC (mayor es mejor)
        if "roic" in df.columns:
            score += self._rank_normalize(df["roic"]) * 0.4
        
        # ROE (mayor es mejor)
        if "returnOnEquity" in df.columns:
            score += self._rank_normalize(df["returnOnEquity"]) * 0.3
        
        # Net Profit Margin (mayor es mejor)
        if "netProfitMargin" in df.columns:
            score += self._rank_normalize(df["netProfitMargin"]) * 0.3
        
        return score
    
    def _calculate_growth_score(self, df: pd.DataFrame) -> pd.Series:
        """Calcula score de crecimiento basado en revenue y EPS growth."""
        score = pd.Series(0.0, index=df.index)
        
        # Revenue Growth (mayor es mejor)
        if "revenueGrowth" in df.columns:
            score += self._rank_normalize(df["revenueGrowth"]) * 0.5
        
        # Net Income Growth (mayor es mejor)
        if "netIncomeGrowth" in df.columns:
            score += self._rank_normalize(df["netIncomeGrowth"]) * 0.5
        
        return score
    
    def _calculate_solvency_score(self, df: pd.DataFrame) -> pd.Series:
        """Calcula score de solvencia basado en deuda y ratios financieros."""
        score = pd.Series(0.0, index=df.index)
        
        # Debt-to-Equity bajo es mejor (invertir)
        if "debtToEquity" in df.columns:
            debt_rank = self._rank_normalize(-df["debtToEquity"])
            score += debt_rank * 0.5
        
        # Current Ratio alto es mejor
        if "currentRatio" in df.columns:
            score += self._rank_normalize(df["currentRatio"]) * 0.5
        
        return score
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """Calcula score de momentum basado en retornos y técnicos."""
        score = pd.Series(0.0, index=df.index)
        
        # 12-month return (mayor es mejor)
        if "returns_12m" in df.columns:
            score += self._rank_normalize(df["returns_12m"]) * 0.4
        elif "close" in df.columns:
            # Calcular retorno 12 meses si está disponible
            returns_12m = df.groupby("ticker")["close"].pct_change(periods=252)
            score += self._rank_normalize(returns_12m) * 0.4
        
        # Price vs SMA 200 (mayor es mejor)
        if "price_vs_sma200" in df.columns:
            score += self._rank_normalize(df["price_vs_sma200"]) * 0.3
        
        # RSI (momentum técnico)
        if "rsi" in df.columns:
            # RSI entre 30-70 es ideal, evitar extremos
            rsi_score = 1.0 - abs(df["rsi"] - 50) / 50
            score += self._rank_normalize(rsi_score) * 0.3
        
        return score
    
    def _rank_normalize(self, series: pd.Series) -> pd.Series:
        """Normaliza una serie usando ranking percentil (0-1)."""
        return series.rank(pct=True, na_option="keep").fillna(0.5)
    
    def calculate_factor_scores(
        self,
        df: pd.DataFrame,
        regime: str | None = None,
    ) -> pd.DataFrame:
        """Calcula scores de factores para todas las empresas.
        
        Args:
            df: DataFrame con fundamentals y técnicos
            regime: Régimen macro actual (para ajustar pesos)
            
        Returns:
            DataFrame con scores de factores y composite score
        """
        # Seleccionar pesos según régimen
        weights = self.regime_weights.get(regime, self.weights) if regime else self.weights
        
        # Calcular scores individuales
        df = df.copy()
        df["value_score"] = self._calculate_value_score(df)
        df["profitability_score"] = self._calculate_profitability_score(df)
        df["growth_score"] = self._calculate_growth_score(df)
        df["solvency_score"] = self._calculate_solvency_score(df)
        df["momentum_score"] = self._calculate_momentum_score(df)
        
        # Calcular composite score
        df["composite_score"] = (
            df["value_score"] * weights.value +
            df["profitability_score"] * weights.profitability +
            df["growth_score"] * weights.growth +
            df["solvency_score"] * weights.solvency +
            df["momentum_score"] * weights.momentum
        )
        
        # Rankear por fecha (cross-sectional)
        if "date" in df.columns:
            df["rank"] = df.groupby("date")["composite_score"].rank(ascending=False, method="dense").astype(int)
        else:
            df["rank"] = df["composite_score"].rank(ascending=False, method="dense").astype(int)
        
        return df
    
    def screen(
        self,
        df: pd.DataFrame,
        top_n: int = 50,
        regime: str | None = None,
        min_score: float = 0.0,
    ) -> pd.DataFrame:
        """Screena empresas y retorna top N por composite score.
        
        Args:
            df: DataFrame con fundamentals y técnicos
            top_n: Número de empresas a retornar
            regime: Régimen macro para ajustar pesos
            min_score: Score mínimo para incluir
            
        Returns:
            DataFrame con top N empresas rankeadas
        """
        scored_df = self.calculate_factor_scores(df, regime=regime)
        
        # Filtrar por score mínimo
        scored_df = scored_df[scored_df["composite_score"] >= min_score]
        
        # Seleccionar top N por fecha
        if "date" in scored_df.columns:
            top_df = scored_df.groupby("date").apply(
                lambda x: x.nlargest(top_n, "composite_score")
            ).reset_index(drop=True)
        else:
            top_df = scored_df.nlargest(top_n, "composite_score")
        
        return top_df.sort_values(["date", "rank"] if "date" in top_df.columns else ["rank"])


class RegimeAwareFactorScreener(FactorScreener):
    """Screener de factores que ajusta pesos según régimen macro."""
    
    def __init__(self) -> None:
        """Inicializa con pesos ajustados por régimen."""
        # Pesos por defecto (expansión)
        default_weights = FactorWeights(
            value=0.25,
            profitability=0.25,
            growth=0.20,
            solvency=0.15,
            momentum=0.15,
        )
        
        # Pesos ajustados por régimen
        regime_weights = {
            "expansion": FactorWeights(
                value=0.20,
                profitability=0.25,
                growth=0.25,  # Más peso a crecimiento en expansión
                solvency=0.10,
                momentum=0.20,
            ),
            "slowdown": FactorWeights(
                value=0.30,  # Más peso a valor en desaceleración
                profitability=0.30,
                growth=0.15,
                solvency=0.15,
                momentum=0.10,
            ),
            "recession": FactorWeights(
                value=0.25,
                profitability=0.25,
                growth=0.10,
                solvency=0.30,  # Máximo peso a solvencia en recesión
                momentum=0.10,
            ),
            "stress": FactorWeights(
                value=0.20,
                profitability=0.20,
                growth=0.05,
                solvency=0.40,  # Enfoque en solvencia en estrés
                momentum=0.15,
            ),
        }
        
        super().__init__(weights=default_weights, regime_weights=regime_weights)

