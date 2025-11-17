"""Pipeline para entrenar y evaluar el detector HMM de régimen macro."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from prefect import flow, task

from caria.config.settings import Settings
from caria.models.regime.hmm_regime_detector import HMMRegimeDetector

LOGGER = logging.getLogger("caria.pipelines.regime_hmm")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@task
def load_macro_features(silver_path: Path, macro_file: str) -> pd.DataFrame:
    """Carga features macro procesadas."""
    file_path = silver_path / macro_file
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo de features macro no encontrado: {file_path}")
    
    df = pd.read_parquet(file_path)
    LOGGER.info("Cargadas %d observaciones de features macro", len(df))
    return df


@task
def prepare_training_data(df: pd.DataFrame, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Prepara datos para entrenamiento HMM."""
    df = df.copy()
    
    # Filtrar por fecha si se especifica
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
    
    # Ordenar por fecha
    df = df.sort_values("date").reset_index(drop=True)
    
    LOGGER.info("Datos preparados: %d observaciones desde %s hasta %s", 
                len(df), 
                df["date"].min() if "date" in df.columns else "N/A",
                df["date"].max() if "date" in df.columns else "N/A")
    
    return df


@task
def train_hmm_model(df: pd.DataFrame, n_states: int = 4, n_iter: int = 100) -> HMMRegimeDetector:
    """Entrena el modelo HMM."""
    detector = HMMRegimeDetector(n_states=n_states, n_iter=n_iter)
    detector.fit(df)
    return detector


@task
def evaluate_hmm_model(detector: HMMRegimeDetector, test_df: pd.DataFrame) -> dict[str, Any]:
    """Evalúa el modelo HMM en datos de test."""
    # Predecir regímenes históricos
    predictions = detector.predict_historical_regimes(test_df)
    
    # Calcular métricas básicas
    regime_counts = predictions["regime"].value_counts().to_dict()
    avg_confidence = predictions["confidence"].mean()
    
    metrics = {
        "regime_distribution": regime_counts,
        "average_confidence": float(avg_confidence),
        "total_predictions": len(predictions),
        "date_range": {
            "start": str(predictions["date"].min()),
            "end": str(predictions["date"].max()),
        },
    }
    
    LOGGER.info("Evaluación completada: %s", metrics)
    return metrics


@task
def save_model_and_predictions(
    detector: HMMRegimeDetector,
    predictions: pd.DataFrame,
    model_path: Path,
    predictions_path: Path,
) -> None:
    """Guarda modelo y predicciones históricas."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    
    detector.save(str(model_path))
    predictions.to_parquet(predictions_path, index=False)
    
    LOGGER.info("Modelo guardado en %s", model_path)
    LOGGER.info("Predicciones guardadas en %s", predictions_path)


@flow(name="caria-regime-hmm-train")
def regime_hmm_train_flow(settings: Settings, config_path: str) -> None:
    """Pipeline principal para entrenar detector HMM de régimen."""
    config = _load_yaml(Path(config_path))
    
    silver_path = Path(settings.get("storage", "silver_path", default="data/silver"))
    models_path = Path(settings.get("storage", "models_path", default="models"))
    
    # Cargar datos macro
    macro_file = config.get("macro_features_file", "macro/macro_features.parquet")
    df = load_macro_features(silver_path, macro_file)
    
    # Preparar datos de entrenamiento
    train_start = config.get("train_start_date")
    train_end = config.get("train_end_date")
    train_df = prepare_training_data(df, train_start, train_end)
    
    # Entrenar modelo
    n_states = config.get("n_states", 4)
    n_iter = config.get("n_iter", 100)
    detector = train_hmm_model(train_df, n_states=n_states, n_iter=n_iter)
    
    # Evaluar en datos de test (si están disponibles)
    test_start = config.get("test_start_date")
    test_end = config.get("test_end_date")
    if test_start or test_end:
        test_df = prepare_training_data(df, test_start, test_end)
        metrics = evaluate_hmm_model(detector, test_df)
        LOGGER.info("Métricas de evaluación: %s", metrics)
    
    # Predecir regímenes históricos completos
    historical_predictions = detector.predict_historical_regimes(df)
    
    # Guardar modelo y predicciones
    model_path = models_path / "regime_hmm_model.pkl"
    predictions_path = silver_path / "regime" / "hmm_regime_predictions.parquet"
    
    save_model_and_predictions(detector, historical_predictions, model_path, predictions_path)


def run(settings: Settings, pipeline_config_path: str | None = None) -> None:
    """Función de entrada para ejecutar el pipeline."""
    if not pipeline_config_path:
        raise ValueError("pipeline_config_path es obligatorio para regime_hmm_train_flow")
    regime_hmm_train_flow(settings=settings, config_path=pipeline_config_path)

