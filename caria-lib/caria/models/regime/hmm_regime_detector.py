"""Detector de régimen macroeconómico usando Hidden Markov Models (HMM) no supervisado.

Este módulo implementa detección de régimen sin etiquetas ex-post, usando HMM para
identificar estados latentes del mercado basándose en features macro observables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    logging.warning("hmmlearn no está instalado. Instala con: pip install hmmlearn")

LOGGER = logging.getLogger("caria.models.regime.hmm")


@dataclass
class RegimeProbabilities:
    """Probabilidades de estar en cada régimen."""
    expansion: float
    slowdown: float
    recession: float
    stress: float  # Régimen de estrés extremo
    confidence: float  # Confianza general en la clasificación


@dataclass
class RegimeState:
    """Estado de régimen detectado."""
    regime: str  # expansion, slowdown, recession, stress
    probabilities: RegimeProbabilities
    state_id: int  # ID del estado HMM (0-3)
    features: dict[str, float]  # Features usadas para la detección


class HMMRegimeDetector:
    """Detector de régimen macro usando HMM Gaussian no supervisado.
    
    El modelo detecta 4 estados latentes:
    - Estado 0: Expansión (crecimiento económico fuerte)
    - Estado 1: Desaceleración (crecimiento moderado)
    - Estado 2: Recesión (contracción económica)
    - Estado 3: Estrés (crisis/volatilidad extrema)
    
    Las probabilidades se calculan usando el algoritmo forward-backward de HMM.
    """
    
    def __init__(
        self,
        n_states: int = 4,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        """Inicializa el detector HMM.

        Args:
            n_states: Número de estados latentes (default: 4)
            n_iter: Iteraciones máximas para entrenamiento EM
            random_state: Semilla para reproducibilidad
        """
        if not HAS_HMMLEARN:
            raise RuntimeError(
                "hmmlearn no está instalado. Instala con: pip install hmmlearn"
            )

        self.n_states = n_states
        self.model: hmm.GaussianHMM | None = None
        self.feature_names: list[str] = []
        self.state_labels = {
            0: "expansion",
            1: "slowdown",
            2: "recession",
            3: "stress",
        }
        self._random_state = random_state
        self._n_iter = n_iter
        # Guardar estadísticas de normalización para consistencia en predicción
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        
    def _prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Prepara features para el HMM.
        
        Selecciona y normaliza features macro clave:
        - yield_curve_slope: Pendiente de curva de rendimientos
        - vix o volatilidad: Indicador de volatilidad/riesgo
        - sentiment_score: Sentimiento del mercado
        - recession_probability: Probabilidad de recesión (si disponible)
        - credit_spread: Spread crediticio (si disponible)
        """
        feature_cols = []
        
        # Features principales
        if "yield_curve_slope" in df.columns:
            feature_cols.append("yield_curve_slope")
        elif "DGS10" in df.columns and "DGS2" in df.columns:
            # Calcular si no existe
            df["yield_curve_slope"] = df["DGS10"] - df["DGS2"]
            feature_cols.append("yield_curve_slope")
        
        # Volatilidad (VIX o calcular desde prices)
        if "vix" in df.columns:
            feature_cols.append("vix")
        elif "VIXCLS" in df.columns:
            feature_cols.append("VIXCLS")
        
        # Sentiment
        if "sentiment_score" in df.columns:
            feature_cols.append("sentiment_score")
        elif "UMCSENT" in df.columns:
            # Normalizar consumer sentiment
            df["sentiment_score"] = (df["UMCSENT"] - 50) / 50  # Normalizar a [-1, 1]
            feature_cols.append("sentiment_score")
        
        # Recession probability (si está calculada)
        if "recession_probability" in df.columns:
            feature_cols.append("recession_probability")
        
        # Credit spread
        if "credit_spread" in df.columns:
            feature_cols.append("credit_spread")
        elif "BAA" in df.columns and "AAA" in df.columns:
            df["credit_spread"] = df["BAA"] - df["AAA"]
            feature_cols.append("credit_spread")
        
        # Si no tenemos suficientes features, agregar más
        if len(feature_cols) < 3:
            # Agregar unemployment rate si está disponible
            if "UNRATE" in df.columns:
                feature_cols.append("UNRATE")
            # Agregar PMI si está disponible
            if "MANPMI" in df.columns:
                feature_cols.append("MANPMI")
        
        if len(feature_cols) < 2:
            raise ValueError(
                f"Se necesitan al menos 2 features macro. Disponibles: {df.columns.tolist()}"
            )
        
        # Seleccionar y limpiar datos
        feature_data = df[feature_cols].copy()
        feature_data = feature_data.dropna()
        
        if len(feature_data) < 100:
            raise ValueError(
                f"Se necesitan al menos 100 observaciones. Disponibles: {len(feature_data)}"
            )
        
        # Normalizar features (z-score)
        feature_array = feature_data.values
        mean = np.nanmean(feature_array, axis=0)
        std = np.nanstd(feature_array, axis=0)
        std[std == 0] = 1.0  # Evitar división por cero
        feature_array = (feature_array - mean) / std
        
        # Reemplazar cualquier NaN restante con 0
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        
        return feature_array, feature_cols
    
    def fit(self, df: pd.DataFrame) -> None:
        """Entrena el modelo HMM con datos históricos.

        Args:
            df: DataFrame con features macro (debe incluir 'date' y features macro)
        """
        LOGGER.info("Preparando features para entrenamiento HMM...")

        # Preparar features y guardar estadísticas de normalización
        feature_array, feature_names = self._prepare_features(df)
        self.feature_names = feature_names

        # IMPORTANTE: Guardar estadísticas de normalización ANTES de normalizar
        # para poder reutilizarlas en predicción
        feature_cols = df[feature_names].copy()
        feature_cols_clean = feature_cols.dropna()
        self._feature_mean = np.nanmean(feature_cols_clean.values, axis=0)
        self._feature_std = np.nanstd(feature_cols_clean.values, axis=0)
        self._feature_std[self._feature_std == 0] = 1.0  # Evitar división por cero

        LOGGER.info(
            "Entrenando HMM con %d observaciones y %d features: %s",
            len(feature_array),
            len(feature_names),
            feature_names,
        )
        LOGGER.info("Feature means guardados: %s", self._feature_mean)
        LOGGER.info("Feature stds guardados: %s", self._feature_std)

        # Crear y entrenar modelo HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self._n_iter,
            random_state=self._random_state,
            verbose=True,
        )

        self.model.fit(feature_array)

        LOGGER.info("HMM entrenado exitosamente")
        LOGGER.info("Probabilidades iniciales de estados: %s", self.model.startprob_)
        LOGGER.info("Matriz de transición:\n%s", self.model.transmat_)
    
    def predict_proba(self, features: dict[str, float] | pd.DataFrame) -> RegimeProbabilities:
        """Predice probabilidades de régimen para features dadas.

        Args:
            features: Dict con features o DataFrame con una fila

        Returns:
            Probabilidades de estar en cada régimen
        """
        if self.model is None:
            raise RuntimeError("Modelo no entrenado. Llama fit() primero.")

        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError(
                "Estadísticas de normalización no disponibles. "
                "Asegúrate de que el modelo fue entrenado con la versión actualizada."
            )

        # Convertir features a array
        if isinstance(features, dict):
            feature_array = np.array([
                features.get(name, 0.0) for name in self.feature_names
            ]).reshape(1, -1)
        else:
            # DataFrame - agregar columnas faltantes con valor 0.0
            missing_cols = [col for col in self.feature_names if col not in features.columns]
            if missing_cols:
                import logging
                logger = logging.getLogger("caria.models.regime")
                logger.warning(
                    "Faltan columnas en features: %s. Usando valor 0.0 por defecto.",
                    missing_cols
                )
                for col in missing_cols:
                    features[col] = 0.0

            # Extraer solo los valores sin normalizar de nuevo
            feature_values = features[self.feature_names].values
            if len(feature_values) > 1:
                feature_array = feature_values[-1:].reshape(1, -1)
            else:
                feature_array = feature_values

        # CORREGIDO: Normalizar usando estadísticas del ENTRENAMIENTO (no de datos actuales)
        # Esto asegura consistencia entre train y predict
        feature_array = (feature_array - self._feature_mean) / self._feature_std
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        
        # Calcular probabilidades usando algoritmo forward
        logprob, state_probs = self.model.score_samples(feature_array)
        
        # Convertir log-probabilidades a probabilidades
        state_probs = np.exp(state_probs[0])
        state_probs = state_probs / state_probs.sum()  # Normalizar
        
        # Mapear estados HMM a regímenes semánticos
        # Asumimos orden: expansion, slowdown, recession, stress
        # En producción, deberíamos hacer análisis de características de cada estado
        probabilities = RegimeProbabilities(
            expansion=float(state_probs[0]),
            slowdown=float(state_probs[1]),
            recession=float(state_probs[2]),
            stress=float(state_probs[3]) if len(state_probs) > 3 else 0.0,
            confidence=float(np.max(state_probs)),  # Confianza = probabilidad máxima
        )
        
        return probabilities
    
    def predict_current_regime(
        self,
        features: dict[str, float] | pd.DataFrame,
    ) -> RegimeState:
        """Predice el régimen actual y retorna estado completo.
        
        Args:
            features: Features macro actuales
            
        Returns:
            Estado de régimen con probabilidades y metadata
        """
        probabilities = self.predict_proba(features)
        
        # Determinar régimen más probable
        regime_probs = {
            "expansion": probabilities.expansion,
            "slowdown": probabilities.slowdown,
            "recession": probabilities.recession,
            "stress": probabilities.stress,
        }
        regime = max(regime_probs, key=regime_probs.get)
        
        # Obtener ID del estado HMM
        state_id = list(regime_probs.keys()).index(regime)
        
        # Extraer features usadas
        if isinstance(features, dict):
            features_dict = features
        else:
            features_dict = features.iloc[-1].to_dict()
        
        return RegimeState(
            regime=regime,
            probabilities=probabilities,
            state_id=state_id,
            features={k: v for k, v in features_dict.items() if k in self.feature_names},
        )
    
    def predict_historical_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predice regímenes para todo el historial.
        
        Args:
            df: DataFrame con features macro históricas
            
        Returns:
            DataFrame con columnas: date, regime, expansion_prob, slowdown_prob, etc.
        """
        if self.model is None:
            raise RuntimeError("Modelo no entrenado. Carga un modelo primero.")
        
        if self._feature_mean is None or self._feature_std is None:
            LOGGER.warning(
                "Estadísticas de normalización no disponibles. "
                "Usando normalización desde cero (puede dar resultados inconsistentes)."
            )
            feature_array, _ = self._prepare_features(df)
        else:
            # Preparar features usando las mismas columnas que el modelo espera
            feature_cols = []
            
            # Features principales (mismo orden que en _prepare_features)
            if "yield_curve_slope" in df.columns:
                feature_cols.append("yield_curve_slope")
            elif "DGS10" in df.columns and "DGS2" in df.columns:
                df["yield_curve_slope"] = df["DGS10"] - df["DGS2"]
                feature_cols.append("yield_curve_slope")
            
            # Sentiment
            if "sentiment_score" in df.columns:
                feature_cols.append("sentiment_score")
            elif "UMCSENT" in df.columns:
                df["sentiment_score"] = (df["UMCSENT"] - 50) / 50
                feature_cols.append("sentiment_score")
            
            # Credit spread
            if "credit_spread" in df.columns:
                feature_cols.append("credit_spread")
            elif "BAA" in df.columns and "AAA" in df.columns:
                df["credit_spread"] = df["BAA"] - df["AAA"]
                feature_cols.append("credit_spread")
            
            # Verificar que tenemos las columnas correctas
            if set(feature_cols) != set(self.feature_names):
                LOGGER.warning(
                    "Features disponibles (%s) no coinciden con features del modelo (%s). "
                    "Usando features del modelo.",
                    feature_cols, self.feature_names
                )
                feature_cols = self.feature_names.copy()
            
            # Seleccionar y limpiar datos
            feature_data = df[feature_cols].copy()
            feature_data = feature_data.dropna()
            
            if len(feature_data) < 100:
                raise ValueError(
                    f"Se necesitan al menos 100 observaciones. Disponibles: {len(feature_data)}"
                )
            
            # Normalizar usando estadísticas del ENTRENAMIENTO (no desde cero)
            feature_array = feature_data.values
            feature_array = (feature_array - self._feature_mean) / self._feature_std
            feature_array = np.nan_to_num(feature_array, nan=0.0)
        
        # Predecir estados para toda la secuencia
        states = self.model.predict(feature_array)
        
        # Calcular probabilidades para cada punto
        logprobs, state_probs = self.model.score_samples(feature_array)
        state_probs = np.exp(state_probs)
        state_probs = state_probs / state_probs.sum(axis=1, keepdims=True)
        
        # Crear DataFrame de resultados
        # Asegurar que tenemos las fechas correctas (puede haber menos filas después de dropna)
        dates = df["date"].values
        if len(dates) > len(states):
            # Si se eliminaron filas por NaN, necesitamos alinear las fechas
            feature_data_with_dates = df[feature_cols + ["date"]].dropna()
            dates = feature_data_with_dates["date"].values
        
        results = pd.DataFrame({
            "date": dates[:len(states)],
            "regime": [self.state_labels.get(s, "unknown") for s in states],
            "expansion_prob": state_probs[:, 0],
            "slowdown_prob": state_probs[:, 1],
            "recession_prob": state_probs[:, 2],
            "stress_prob": state_probs[:, 3] if state_probs.shape[1] > 3 else 0.0,
            "confidence": np.max(state_probs, axis=1),
        })
        
        return results
    
    def save(self, path: str) -> None:
        """Guarda el modelo entrenado con estadísticas de normalización."""
        import pickle

        if self.model is None:
            raise RuntimeError("No hay modelo entrenado para guardar")

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "n_states": self.n_states,
            "state_labels": self.state_labels,
            # NUEVO: Guardar estadísticas de normalización
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        LOGGER.info("Modelo HMM guardado en %s (con estadísticas de normalización)", path)
    
    @classmethod
    def load(cls, path: str) -> "HMMRegimeDetector":
        """Carga un modelo entrenado con estadísticas de normalización."""
        import pickle

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        detector = cls(n_states=model_data["n_states"])
        detector.model = model_data["model"]
        detector.feature_names = model_data["feature_names"]
        detector.state_labels = model_data["state_labels"]

        # NUEVO: Cargar estadísticas de normalización (backward compatible)
        detector._feature_mean = model_data.get("feature_mean", None)
        detector._feature_std = model_data.get("feature_std", None)

        if detector._feature_mean is None or detector._feature_std is None:
            LOGGER.warning(
                "Modelo cargado sin estadísticas de normalización (versión antigua). "
                "Re-entrena el modelo para tener predicciones consistentes."
            )
        else:
            LOGGER.info("Modelo HMM cargado desde %s (con estadísticas de normalización)", path)

        return detector

