"""Servicio para screening de factores cuantitativos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from caria.config.settings import Settings
from caria.models.factors.factor_screener import RegimeAwareFactorScreener
from caria.services.regime_service import RegimeService

LOGGER = logging.getLogger("caria.services.factor")


class FactorService:
    """Servicio para screening de empresas usando factores cuantitativos."""
    
    def __init__(self, settings: Settings) -> None:
        """Inicializa el servicio de factores."""
        self.settings = settings
        self.screener = RegimeAwareFactorScreener()
        self.regime_service = RegimeService(settings)
    
    def _load_fundamentals(self) -> pd.DataFrame:
        """Carga datos de fundamentals y técnicos."""
        silver_path_str = self.settings.get("storage", "silver_path", default="data/silver")
        silver_path = self.settings.resolve_path(silver_path_str)
        
        # Cargar fundamentals
        quality_path = silver_path / "fundamentals" / "quality_signals.parquet"
        value_path = silver_path / "fundamentals" / "value_signals.parquet"
        momentum_path = silver_path / "technicals" / "momentum_signals.parquet"
        
        # Si no existen en la ubicación resuelta, buscar en ubicaciones alternativas
        if not quality_path.exists():
            # Buscar en caria_data/silver directamente
            current_file = Path(__file__).resolve()
            caria_data_root = current_file.parent.parent.parent.parent
            alt_paths = [
                caria_data_root / "silver" / "fundamentals" / "quality_signals.parquet",
                caria_data_root / "data" / "silver" / "fundamentals" / "quality_signals.parquet",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    quality_path = alt_path
                    break
        
        if not value_path.exists():
            current_file = Path(__file__).resolve()
            caria_data_root = current_file.parent.parent.parent.parent
            alt_paths = [
                caria_data_root / "silver" / "fundamentals" / "value_signals.parquet",
                caria_data_root / "data" / "silver" / "fundamentals" / "value_signals.parquet",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    value_path = alt_path
                    break
        
        if not momentum_path.exists():
            current_file = Path(__file__).resolve()
            caria_data_root = current_file.parent.parent.parent.parent
            alt_paths = [
                caria_data_root / "silver" / "technicals" / "momentum_signals.parquet",
                caria_data_root / "data" / "silver" / "technicals" / "momentum_signals.parquet",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    momentum_path = alt_path
                    break
        
        dfs = []
        if quality_path.exists():
            dfs.append(pd.read_parquet(quality_path))
        if value_path.exists():
            dfs.append(pd.read_parquet(value_path))
        if momentum_path.exists():
            dfs.append(pd.read_parquet(momentum_path))
        
        if not dfs:
            raise FileNotFoundError(
                f"No se encontraron datos de fundamentals o técnicos. Buscado en:\n"
                f"  - {quality_path}\n"
                f"  - {value_path}\n"
                f"  - {momentum_path}"
            )
        
        # Merge de todos los datasets
        df = dfs[0]
        for other_df in dfs[1:]:
            merge_keys = ["date", "ticker"]
            common_keys = [k for k in merge_keys if k in df.columns and k in other_df.columns]
            if common_keys:
                df = df.merge(other_df, on=common_keys, how="outer", suffixes=("", "_dup"))
        
        return df
    
    def screen_companies(
        self,
        top_n: int = 50,
        regime: str | None = None,
        date: str | pd.Timestamp | None = None,
    ) -> list[dict[str, Any]]:
        """Screena empresas y retorna top N.
        
        Args:
            top_n: Número de empresas a retornar
            regime: Régimen macro (si None, detecta automáticamente)
            date: Fecha específica para screening (si None, usa más reciente)
            
        Returns:
            Lista de dicts con información de empresas rankeadas
        """
        # Cargar datos
        df = self._load_fundamentals()
        
        # Filtrar por fecha si se especifica
        if date is not None:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            df = df[df["date"] <= date]
        
        # Obtener régimen si no se proporciona
        if regime is None:
            regime_state = self.regime_service.get_current_regime()
            if regime_state:
                regime = regime_state.regime
            else:
                regime = "expansion"  # Default
        
        LOGGER.info("Screening con régimen: %s", regime)
        
        # Filtrar a fecha más reciente si no se especificó fecha
        if date is None and "date" in df.columns:
            latest_date = df["date"].max()
            df = df[df["date"] == latest_date]
            LOGGER.info("Usando datos de fecha: %s", latest_date)
        
        # Screenear
        top_companies = self.screener.screen(df, top_n=top_n, regime=regime)
        
        # Convertir a formato JSON-friendly
        results = []
        for _, row in top_companies.iterrows():
            results.append({
                "ticker": str(row.get("ticker", "")),
                "date": str(row.get("date", "")),
                "composite_score": float(row.get("composite_score", 0.0)),
                "rank": int(row.get("rank", 0)),
                "factor_scores": {
                    "value": float(row.get("value_score", 0.0)),
                    "profitability": float(row.get("profitability_score", 0.0)),
                    "growth": float(row.get("growth_score", 0.0)),
                    "solvency": float(row.get("solvency_score", 0.0)),
                    "momentum": float(row.get("momentum_score", 0.0)),
                },
                "regime": regime,
            })
        
        return results

