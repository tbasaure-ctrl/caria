"""Reentrenar modelo HMM con datos extendidos y más features."""

import sys
import os
from pathlib import Path

# Add caria-lib to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "caria-lib"))

import pandas as pd
import numpy as np
from loguru import logger

try:
    from caria.models.regime.hmm_regime_detector import HMMRegimeDetector
except ImportError:
    print("ERROR: No se puede importar HMMRegimeDetector. Verifica que caria-lib esté instalado.")
    sys.exit(1)

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[1]
MACRO_DATA_PATH = REPO_ROOT / "silver" / "macro" / "fred_us.parquet"
MODEL_PATH = REPO_ROOT / "caria_data" / "models" / "regime_hmm_model.pkl"
OUTPUT_MODEL_PATH = REPO_ROOT / "caria_data" / "models" / "regime_hmm_model_extended.pkl"

# Período de entrenamiento
TRAIN_START = "1990-01-01"  # Datos más confiables desde 1990
TRAIN_END = "2024-12-31"


def load_and_prepare_data() -> pd.DataFrame:
    """Carga y prepara datos macro para entrenamiento."""
    logger.info(f"Cargando datos macro desde {MACRO_DATA_PATH}")
    
    if not MACRO_DATA_PATH.exists():
        raise FileNotFoundError(f"Datos macro no encontrados en {MACRO_DATA_PATH}")
    
    df = pd.read_parquet(MACRO_DATA_PATH)
    logger.info(f"Cargados {len(df)} registros")
    logger.info(f"Rango de fechas: {df['date'].min()} a {df['date'].max()}")
    logger.info(f"Columnas disponibles: {len(df.columns)}")
    
    # Filtrar por período de entrenamiento
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)]
    logger.info(f"Filtrado a {len(df)} registros ({TRAIN_START} a {TRAIN_END})")
    
    return df


def train_extended_hmm(df: pd.DataFrame) -> HMMRegimeDetector:
    """Entrena modelo HMM con features extendidas."""
    logger.info("Iniciando entrenamiento de HMM extendido...")
    
    # Crear detector con 4 estados
    detector = HMMRegimeDetector(
        n_states=4,
        n_iter=200,  # Más iteraciones para mejor convergencia
        random_state=42
    )
    
    # Entrenar
    detector.fit(df)
    
    logger.info("✅ Modelo HMM entrenado exitosamente")
    
    # Mostrar estadísticas
    predictions = detector.predict_historical_regimes(df)
    logger.info("\nDistribución de regímenes:")
    logger.info(predictions['regime'].value_counts(normalize=True))
    logger.info(f"\nConfianza promedio: {predictions['confidence'].mean():.3f}")
    
    return detector


def main():
    try:
        # Cargar datos
        df = load_and_prepare_data()
        
        # Entrenar modelo
        detector = train_extended_hmm(df)
        
        # Guardar modelo
        OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        detector.save(str(OUTPUT_MODEL_PATH))
        logger.info(f"\n✅ Modelo guardado en: {OUTPUT_MODEL_PATH}")
        
        # También guardar como modelo principal (backup primero)
        if MODEL_PATH.exists():
            backup_path = MODEL_PATH.with_suffix('.pkl.backup')
            import shutil
            shutil.copy(MODEL_PATH, backup_path)
            logger.info(f"Backup del modelo anterior guardado en: {backup_path}")
        
        detector.save(str(MODEL_PATH))
        logger.info(f"✅ Modelo también guardado como principal en: {MODEL_PATH}")
        
        logger.info("\n" + "=" * 60)
        logger.info("ENTRENAMIENTO COMPLETADO")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

