"""Servicio para inferencia de régimen macro usando HMM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from caria.config.settings import Settings
from caria.models.regime.hmm_regime_detector import HMMRegimeDetector, RegimeState

LOGGER = logging.getLogger("caria.services.regime")


class RegimeService:
    """Servicio para detectar régimen macro actual usando HMM entrenado."""
    
    def __init__(self, settings: Settings, model_path: str | Path | None = None) -> None:
        """Inicializa el servicio de régimen.
        
        Args:
            settings: Configuración de Caria
            model_path: Ruta al modelo HMM entrenado (opcional, busca en models_path)
        """
        self.settings = settings
        self.detector = None
        self._model_path = model_path  # Guardar path para lazy loading
        self._model_loaded = False
        
    def _load_model(self) -> None:
        """Carga el modelo bajo demanda."""
        if self._model_loaded:
            return

        # Determinar ruta del modelo si no se especificó
        model_path = self._model_path
        if model_path is None:
            models_path_str = self.settings.get("storage", "models_path", default="models")
            models_path = Path(models_path_str)
            
            # Si es relativo, intentar encontrar desde caria_data/
            if not models_path.is_absolute():
                # Intentar encontrar caria_data desde el módulo actual
                current_file = Path(__file__).resolve()
                # Desde caria/services/regime_service.py, subir hasta caria_data/
                caria_data_root = current_file.parent.parent.parent.parent
                if (caria_data_root / "models").exists():
                    models_path = caria_data_root / "models"
                elif (caria_data_root.parent / "caria_data" / "models").exists():
                    models_path = caria_data_root.parent / "caria_data" / "models"
            
            model_path = models_path / "regime_hmm_model.pkl"
        else:
            model_path = Path(model_path)
        
        # Cargar modelo
        if not model_path.exists():
            LOGGER.warning("Modelo HMM no encontrado en %s. Servicio no disponible.", model_path)
            self.detector = None
        else:
            try:
                LOGGER.info("Cargando modelo HMM desde %s...", model_path)
                self.detector = HMMRegimeDetector.load(str(model_path))
                LOGGER.info("Modelo HMM cargado exitosamente")
            except Exception as exc:
                LOGGER.exception("Error cargando modelo HMM: %s", exc)
                self.detector = None
        
        self._model_loaded = True
    
    def get_current_regime(
        self,
        features: dict[str, float] | pd.DataFrame | None = None,
    ) -> RegimeState | None:
        """Obtiene el régimen macro actual.
        
        Args:
            features: Features macro actuales (opcional, carga desde DB si no se proporciona)
            
        Returns:
            Estado de régimen actual o None si el servicio no está disponible
        """
        # Carga lazy del modelo
        self._load_model()

        if self.detector is None:
            return None
        
        # Si no se proporcionan features, intentar cargar desde datos más recientes
        if features is None:
            features = self._load_latest_features()
            if features is None:
                LOGGER.debug(
                    "No se pudieron cargar features macro desde archivo. "
                    "El servicio de régimen usará fallback heurístico si está disponible."
                )
                return None
        
        try:
            regime_state = self.detector.predict_current_regime(features)
            LOGGER.info(
                "Régimen detectado: %s (confianza: %.2f%%)",
                regime_state.regime,
                regime_state.probabilities.confidence * 100,
            )
            return regime_state
        except Exception as exc:
            LOGGER.exception("Error prediciendo régimen: %s", exc)
            return None
    
    def get_regime_probabilities(
        self,
        features: dict[str, float] | pd.DataFrame | None = None,
    ) -> dict[str, float] | None:
        """Obtiene probabilidades de régimen en formato JSON-friendly.
        
        Args:
            features: Features macro actuales (opcional)
            
        Returns:
            Dict con probabilidades o None si el servicio no está disponible
        """
        regime_state = self.get_current_regime(features)
        if regime_state is None:
            return None
        
        probs = regime_state.probabilities
        return {
            "regime": regime_state.regime,
            "probabilities": {
                "expansion": float(probs.expansion),
                "slowdown": float(probs.slowdown),
                "recession": float(probs.recession),
                "stress": float(probs.stress),
            },
            "confidence": float(probs.confidence),
            "features_used": regime_state.features,
        }
    
    def _load_latest_features(self) -> pd.DataFrame | None:
        """Carga las features macro más recientes desde datos procesados."""
        try:
            silver_path_str = self.settings.get("storage", "silver_path", default="data/silver")
            silver_path = self.settings.resolve_path(silver_path_str)
            macro_file = silver_path / "macro" / "macro_features.parquet"
            
            # Si no existe en la ubicación resuelta, buscar en ubicaciones alternativas
            if not macro_file.exists():
                current_file = Path(__file__).resolve()
                caria_data_root = current_file.parent.parent.parent.parent
                alt_paths = [
                    caria_data_root / "data" / "silver" / "macro" / "macro_features.parquet",
                    caria_data_root / "silver" / "macro" / "macro_features.parquet",
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        macro_file = alt_path
                        break
            
            if not macro_file.exists():
                LOGGER.debug(
                    "Archivo de features macro no encontrado: %s. "
                    "El servicio de régimen usará fallback heurístico si está disponible.",
                    macro_file
                )
                return None
            
            df = pd.read_parquet(macro_file)
            if len(df) == 0:
                return None
            
            # Retornar la fila más reciente
            df = df.sort_values("date").tail(1)
            return df
            
        except Exception as exc:
            LOGGER.exception("Error cargando features macro: %s", exc)
            return None
    
    def is_available(self) -> bool:
        """Verifica si el servicio está disponible."""
        self._load_model()
        return self.detector is not None
