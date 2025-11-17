"""Servicio de valuación híbrida para empresas."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from caria.config.settings import Settings
from caria.models.valuation.company_classifier import CompanyClassifier, CompanyStage
from caria.models.valuation.dcf_valuator import DCFValuator, DCFValuation
from caria.models.valuation.monte_carlo_valuator import MonteCarloValuator, MonteCarloValuation
from caria.models.valuation.monte_carlo_presets import get_preset, map_sector_to_industry
from caria.models.valuation.scorecard_valuator import ScorecardValuator, ScorecardValuation
from caria.services.regime_service import RegimeService
from caria.services.rag_service import RAGService

LOGGER = logging.getLogger("caria.services.valuation")


class ValuationService:
    """Servicio de valuación híbrida que selecciona método según etapa de empresa."""
    
    def __init__(self, settings: Settings) -> None:
        """Inicializa el servicio de valuación."""
        self.settings = settings
        self.classifier = CompanyClassifier()
        self.dcf_valuator = DCFValuator()
        self.scorecard_valuator = ScorecardValuator()
        self.monte_carlo_valuator = MonteCarloValuator()
        self.regime_service = RegimeService(settings)
        # RAG service se inicializa lazy si se necesita
    
    def _load_company_data(self, ticker: str) -> dict[str, Any]:
        """Carga datos de una empresa desde fundamentals y prices."""
        silver_path = Path(self.settings.get("storage", "silver_path", default="data/silver"))
        
        data = {"ticker": ticker}
        
        # Cargar fundamentals
        quality_path = silver_path / "fundamentals" / "quality_signals.parquet"
        value_path = silver_path / "fundamentals" / "value_signals.parquet"
        
        if quality_path.exists():
            df_quality = pd.read_parquet(quality_path)
            df_quality = df_quality[df_quality["ticker"] == ticker].sort_values("date").tail(1)
            if not df_quality.empty:
                data.update(df_quality.iloc[0].to_dict())
        
        if value_path.exists():
            df_value = pd.read_parquet(value_path)
            df_value = df_value[df_value["ticker"] == ticker].sort_values("date").tail(1)
            if not df_value.empty:
                data.update(df_value.iloc[0].to_dict())
        
        return data
    
    def _get_nlp_projection(self, ticker: str) -> float | None:
        """Obtiene proyección de crecimiento desde NLP (Sistema II).
        
        TODO: Implementar cuando RAG service tenga capacidad de extraer
        proyecciones de earnings calls.
        """
        # Por ahora retorna None
        # En el futuro, usar RAG para analizar earnings calls y extraer proyecciones
        return None
    
    def value_company(
        self,
        ticker: str,
        current_price: float | None = None,
    ) -> dict[str, Any]:
        """Valúa una empresa usando método apropiado según su etapa.
        
        Args:
            ticker: Símbolo de la empresa
            current_price: Precio actual (si None, intenta cargar)
            
        Returns:
            Dict con resultado de valuación
        """
        # Cargar datos de la empresa
        company_data = self._load_company_data(ticker)
        
        # Obtener revenue y edad
        revenue = company_data.get("revenue", 0.0)
        if pd.isna(revenue):
            revenue = 0.0
        
        # Clasificar etapa
        stage = self.classifier.classify(
            ticker=ticker,
            revenue=revenue,
            age_years=None,  # TODO: Calcular desde fecha de IPO
        )
        
        # Obtener régimen macro
        regime_state = self.regime_service.get_current_regime()
        regime = regime_state.regime if regime_state else None
        
        # Valuar según etapa
        if stage.stage == "consolidated":
            # DCF para empresas consolidadas
            fcf = company_data.get("freeCashFlowPerShare", 0.0) * company_data.get("sharesOutstanding", 1.0)
            if fcf <= 0:
                fcf = company_data.get("freeCashFlow", 0.0)
            
            shares = company_data.get("sharesOutstanding", 1.0)
            if shares <= 0:
                shares = 1.0
            
            if current_price is None:
                # Intentar cargar precio actual
                # TODO: Cargar desde prices
                current_price = 100.0  # Placeholder
            
            # Obtener proyección NLP si está disponible
            nlp_projection = self._get_nlp_projection(ticker)
            
            valuation = self.dcf_valuator.value(
                ticker=ticker,
                current_fcf=fcf,
                shares_outstanding=shares,
                current_price=current_price,
                regime=regime,
                nlp_projection=nlp_projection,
            )
            
            return {
                "ticker": ticker,
                "method": "dcf",
                "stage": "consolidated",
                "fair_value_per_share": valuation.fair_value_per_share,
                "current_price": valuation.current_price,
                "upside_downside": valuation.upside_downside,
                "wacc": valuation.wacc,
                "explanation": valuation.explanation,
                "regime": regime,
            }
        
        else:
            # Scorecard para pre-revenue
            # Por ahora, usar valores por defecto (en producción, requeriría datos cualitativos)
            from caria.models.valuation.scorecard_valuator import ScorecardFactors
            
            factors = ScorecardFactors(
                team_quality=5.0,  # Placeholder (0-10 scale)
                technology=5.0,
                market_opportunity=5.0,
                product_progress=5.0,
                traction=5.0,
                fundraising=5.0,
                go_to_market=5.0,
            )
            
            valuation = self.scorecard_valuator.value(
                ticker=ticker,
                factors=factors,
                stage="seed",  # Default stage
                sector=None,
                recent_funding_valuation=None,
            )
            
            return {
                "ticker": ticker,
                "method": "scorecard",
                "stage": "pre_revenue",
                "estimated_value": valuation.estimated_value,
                "base_value": valuation.base_value,
                "scorecard_scores": valuation.scorecard_scores,
                "explanation": valuation.explanation,
            }
    
    def value_with_monte_carlo(
        self,
        ticker: str,
        current_price: float | None = None,
        monte_carlo_config: dict[str, Any] | None = None,
        n_paths: int = 10_000,
        seed: int | None = 42,
        country_risk: str = "low",
    ) -> dict[str, Any]:
        """Valúa una empresa usando DCF, múltiplos y Monte Carlo.
        
        Args:
            ticker: Símbolo de la empresa
            current_price: Precio actual (si None, intenta cargar)
            monte_carlo_config: Configuración personalizada para Monte Carlo (opcional)
            n_paths: Número de simulaciones para Monte Carlo
            seed: Semilla para reproducibilidad
            country_risk: Nivel de riesgo geopolítico ("low", "medium", "high")
            
        Returns:
            Dict con resultados de DCF, múltiplos y Monte Carlo
        """
        # Cargar datos de la empresa
        company_data = self._load_company_data(ticker)
        
        # Obtener revenue, sector y otros datos
        revenue = company_data.get("revenue", 0.0)
        if pd.isna(revenue):
            revenue = 0.0
        
        sector = company_data.get("sector") or company_data.get("industry")
        shares = company_data.get("sharesOutstanding", 1.0)
        if shares <= 0:
            shares = 1.0
        
        total_debt = company_data.get("totalDebt", 0.0)
        cash = company_data.get("cashAndCashEquivalents", 0.0)
        net_debt = total_debt - cash
        
        ebitda = company_data.get("ebitda", 0.0)
        if pd.isna(ebitda):
            ebitda = 0.0
        
        fcf = company_data.get("freeCashFlowPerShare", 0.0) * shares
        if fcf <= 0:
            fcf = company_data.get("freeCashFlow", 0.0)
        
        if current_price is None:
            current_price = company_data.get("price", 100.0)  # Placeholder
        
        # Clasificar etapa
        stage = self.classifier.classify(
            ticker=ticker,
            revenue=revenue,
            age_years=None,
        )
        
        # Obtener régimen macro
        regime_state = self.regime_service.get_current_regime()
        regime = regime_state.regime if regime_state else None
        
        # Preparar configuración para Monte Carlo
        # Obtener preset según industria/etapa
        industry = map_sector_to_industry(sector)
        preset = get_preset(
            industry=industry,
            stage=stage.stage,
            country_risk=country_risk,
            sector=sector,
        )
        
        # Llenar con datos reales de la empresa
        preset["ticker"] = ticker
        preset["current_price"] = current_price
        preset["shares_out"] = shares
        preset["net_debt"] = net_debt
        
        if revenue > 0:
            preset["base"]["revenue"] = revenue
            if ebitda > 0 and revenue > 0:
                preset["base"]["ebitda_margin"] = ebitda / revenue
            if fcf > 0:
                preset["base"]["fcf_start"] = fcf
        
        # Ajustar escenarios macro según régimen
        if regime == "recession":
            preset["macro"]["probs"] = [0.40, 0.50, 0.10]
        elif regime == "expansion":
            preset["macro"]["probs"] = [0.10, 0.50, 0.40]
        else:
            preset["macro"]["probs"] = [0.15, 0.60, 0.25]
        
        # Si hay overrides parciales, fusionarlos con el preset
        if monte_carlo_config:
            # Merge profundo de overrides con preset
            def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
                """Fusiona override en base recursivamente."""
                result = base.copy()
                for key, value in override.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result
            
            monte_carlo_config = deep_merge(preset, monte_carlo_config)
        else:
            monte_carlo_config = preset
        
        # Ejecutar Monte Carlo
        mc_valuation = self.monte_carlo_valuator.value(
            ticker=ticker,
            cfg=monte_carlo_config,
            n_paths=n_paths,
            seed=seed,
        )
        
        # Ejecutar DCF si es empresa consolidada
        dcf_result = None
        if stage.stage == "consolidated" and fcf > 0:
            nlp_projection = self._get_nlp_projection(ticker)
            dcf_valuation = self.dcf_valuator.value(
                ticker=ticker,
                current_fcf=fcf,
                shares_outstanding=shares,
                current_price=current_price,
                total_debt=total_debt,
                cash_and_equivalents=cash,
                regime=regime,
                nlp_projection=nlp_projection,
            )
            dcf_result = {
                "fair_value_per_share": dcf_valuation.fair_value_per_share,
                "upside_downside": dcf_valuation.upside_downside,
                "wacc": dcf_valuation.wacc,
                "explanation": dcf_valuation.explanation,
            }
        
        # Retornar resultados combinados
        result = {
            "ticker": ticker,
            "stage": stage.stage,
            "industry": map_sector_to_industry(sector),
            "sector": sector,
            "current_price": current_price,
            "regime": regime,
            "monte_carlo": {
                "percentiles": mc_valuation.percentiles,
                "mean": mc_valuation.mean,
                "median": mc_valuation.median,
                "explanation": mc_valuation.explanation,
                "methods_used": mc_valuation.methods_used,
                "visualization_histogram": mc_valuation.visualization_histogram,
                "visualization_paths": mc_valuation.visualization_paths,
                "configuration_used": mc_valuation.configuration_used,
            },
        }
        
        if dcf_result:
            result["dcf"] = dcf_result
        
        return result

