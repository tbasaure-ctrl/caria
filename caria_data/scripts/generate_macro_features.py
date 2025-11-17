"""Script para generar macro_features.parquet desde fred_data.parquet.

Este script procesa los datos macro de FRED y genera features macro
necesarias para el modelo HMM de régimen.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Agregar src al path
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import logging
import pandas as pd
import numpy as np

from caria.config.settings import Settings

# Importar función de feature engineering existente
# Agregar scripts al path para importar macro_features
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from feature_engineering.macro_features import calculate_macro_features

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def find_fred_data(base_path: Path) -> Path | None:
    """Busca fred_data.parquet en ubicaciones comunes."""
    possible_paths = [
        base_path / "data" / "silver" / "macro" / "fred_data.parquet",
        base_path / "silver" / "macro" / "fred_data.parquet",
        Path("data") / "silver" / "macro" / "fred_data.parquet",
        Path("caria_data") / "data" / "silver" / "macro" / "fred_data.parquet",
    ]
    
    for path in possible_paths:
        if path.exists():
            LOGGER.info("Encontrado fred_data.parquet en: %s", path)
            return path
    
    return None


def generate_macro_features(
    fred_data_path: Path | None = None,
    output_path: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Genera macro_features.parquet desde fred_data.parquet.
    
    Args:
        fred_data_path: Path al archivo fred_data.parquet (si None, busca automáticamente)
        output_path: Path donde guardar macro_features.parquet (si None, usa configuración)
        settings: Configuración de CARIA (si None, carga desde base.yaml)
    
    Returns:
        Path al archivo generado
    """
    # Cargar configuración si no se proporciona
    if settings is None:
        config_path = BASE_DIR / "configs" / "base.yaml"
        if config_path.exists():
            settings = Settings.from_yaml(config_path)
        else:
            settings = Settings()
    
    # Buscar fred_data.parquet si no se proporciona
    if fred_data_path is None:
        fred_data_path = find_fred_data(BASE_DIR)
        if fred_data_path is None:
            # Intentar desde el directorio actual de trabajo
            fred_data_path = find_fred_data(Path.cwd())
    
    if fred_data_path is None or not fred_data_path.exists():
        raise FileNotFoundError(
            f"No se encontró fred_data.parquet. Buscado en:\n"
            f"  - {BASE_DIR / 'data' / 'silver' / 'macro'}\n"
            f"  - {BASE_DIR / 'silver' / 'macro'}\n"
            f"  - {Path('data') / 'silver' / 'macro'}\n"
            f"  - {Path('caria_data') / 'data' / 'silver' / 'macro'}\n"
            f"\nPor favor, asegúrate de que el archivo existe en una de estas ubicaciones."
        )
    
    LOGGER.info("Leyendo fred_data.parquet desde: %s", fred_data_path)
    df = pd.read_parquet(fred_data_path)
    
    # Verificar que tenga columna 'date'
    if "date" not in df.columns:
        raise ValueError("fred_data.parquet debe tener una columna 'date'")
    
    # Convertir date a datetime si es necesario
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    LOGGER.info("Datos cargados: %d filas, %d columnas", len(df), len(df.columns))
    LOGGER.info("Rango de fechas: %s a %s", df["date"].min(), df["date"].max())
    
    # Calcular features macro usando la función existente
    LOGGER.info("Calculando features macro...")
    df_features = calculate_macro_features(df)
    
    # Asegurar que 'date' esté presente
    if "date" not in df_features.columns:
        df_features["date"] = df["date"]
    
    # Ordenar por fecha
    df_features = df_features.sort_values("date").reset_index(drop=True)
    
    LOGGER.info("Features calculadas: %d filas, %d columnas", len(df_features), len(df_features.columns))
    LOGGER.info("Columnas generadas: %s", df_features.columns.tolist()[:20])
    
    # Determinar path de salida
    if output_path is None:
        silver_path = Path(settings.get("storage", "silver_path", default="data/silver"))
        
        # Si es relativo, resolverlo relativo a BASE_DIR
        if not silver_path.is_absolute():
            # Intentar desde BASE_DIR primero
            possible_silver = BASE_DIR / silver_path
            if possible_silver.exists():
                silver_path = possible_silver
            else:
                # Si no existe, crear en BASE_DIR
                silver_path = BASE_DIR / silver_path
        
        output_path = silver_path / "macro" / "macro_features.parquet"
    
    # Crear directorio si no existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar
    LOGGER.info("Guardando macro_features.parquet en: %s", output_path)
    df_features.to_parquet(output_path, index=False)
    
    LOGGER.info("✅ macro_features.parquet generado exitosamente")
    LOGGER.info("   Ubicación: %s", output_path)
    LOGGER.info("   Filas: %d", len(df_features))
    LOGGER.info("   Columnas: %d", len(df_features.columns))
    
    return output_path


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Genera macro_features.parquet desde fred_data.parquet"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path al archivo fred_data.parquet (si no se especifica, busca automáticamente)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path donde guardar macro_features.parquet (si no se especifica, usa configuración)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path al archivo de configuración base.yaml (opcional)",
    )
    
    args = parser.parse_args()
    
    # Cargar configuración si se especifica
    settings = None
    if args.config:
        settings = Settings.from_yaml(args.config)
    
    try:
        output_path = generate_macro_features(
            fred_data_path=args.input,
            output_path=args.output,
            settings=settings,
        )
        print(f"\n✅ Éxito: macro_features.parquet generado en {output_path}")
        return 0
    except Exception as e:
        LOGGER.exception("Error generando macro_features.parquet: %s", e)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

