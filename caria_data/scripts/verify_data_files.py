"""Script para verificar que todos los archivos de datos requeridos existan.

Este script verifica la existencia y estructura de los archivos de datos
necesarios para que CARIA funcione correctamente.
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

from caria.config.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


# Archivos requeridos y sus ubicaciones esperadas
REQUIRED_FILES = {
    "macro_features": {
        "paths": [
            "data/silver/macro/macro_features.parquet",
            "silver/macro/macro_features.parquet",
        ],
        "required_columns": ["date"],
        "description": "Features macro para el modelo HMM de régimen",
    },
    "quality_signals": {
        "paths": [
            "data/silver/fundamentals/quality_signals.parquet",
            "silver/fundamentals/quality_signals.parquet",
        ],
        "required_columns": ["date", "ticker"],
        "description": "Señales de calidad para factor screening",
    },
    "value_signals": {
        "paths": [
            "data/silver/fundamentals/value_signals.parquet",
            "silver/fundamentals/value_signals.parquet",
        ],
        "required_columns": ["date", "ticker"],
        "description": "Señales de valor para factor screening",
    },
    "momentum_signals": {
        "paths": [
            "data/silver/technicals/momentum_signals.parquet",
            "silver/technicals/momentum_signals.parquet",
        ],
        "required_columns": ["date", "ticker"],
        "description": "Señales de momentum para factor screening",
    },
    "regime_hmm_model": {
        "paths": [
            "models/regime_hmm_model.pkl",
        ],
        "required_columns": None,  # Es un archivo pickle, no parquet
        "description": "Modelo HMM entrenado para detección de régimen",
    },
}

# Archivos opcionales
OPTIONAL_FILES = {
    "fred_data": {
        "paths": [
            "data/silver/macro/fred_data.parquet",
            "silver/macro/fred_data.parquet",
        ],
        "description": "Datos raw de FRED (usado para generar macro_features)",
    },
}


def find_file(settings: Settings, file_config: dict) -> tuple[Path | None, str]:
    """Busca un archivo en las ubicaciones posibles.
    
    Returns:
        Tuple de (path encontrado, mensaje)
    """
    for rel_path in file_config["paths"]:
        full_path = settings.resolve_path(rel_path)
        if full_path.exists():
            return full_path, f"✅ Encontrado en: {full_path}"
    
    # Si no se encuentra, retornar el primer path esperado para el mensaje de error
    first_path = settings.resolve_path(file_config["paths"][0])
    return None, f"❌ No encontrado. Buscado en: {', '.join(file_config['paths'])}"


def verify_file_structure(file_path: Path, required_columns: list[str] | None) -> tuple[bool, str]:
    """Verifica que un archivo tenga la estructura esperada.
    
    Returns:
        Tuple de (es válido, mensaje)
    """
    try:
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
            
            if len(df) == 0:
                return False, "⚠️  Archivo vacío (0 filas)"
            
            if required_columns:
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    return False, f"⚠️  Faltan columnas requeridas: {missing_cols}"
            
            return True, f"✅ Estructura válida: {len(df)} filas, {len(df.columns)} columnas"
        elif file_path.suffix == ".pkl":
            # Para archivos pickle, solo verificar que exista
            return True, "✅ Archivo pickle encontrado"
        else:
            return True, f"✅ Archivo encontrado (tipo: {file_path.suffix})"
    except Exception as e:
        return False, f"❌ Error leyendo archivo: {e}"


def verify_all_files(settings: Settings) -> tuple[bool, list[str]]:
    """Verifica todos los archivos requeridos.
    
    Returns:
        Tuple de (todos los archivos existen, lista de mensajes)
    """
    all_ok = True
    messages = []
    
    messages.append("\n" + "=" * 70)
    messages.append("VERIFICACIÓN DE ARCHIVOS DE DATOS DE CARIA")
    messages.append("=" * 70 + "\n")
    
    # Verificar archivos requeridos
    messages.append("📋 ARCHIVOS REQUERIDOS:")
    messages.append("-" * 70)
    
    for file_name, file_config in REQUIRED_FILES.items():
        messages.append(f"\n{file_name.upper()}:")
        messages.append(f"  Descripción: {file_config['description']}")
        
        file_path, find_msg = find_file(settings, file_config)
        messages.append(f"  {find_msg}")
        
        if file_path is None:
            all_ok = False
        else:
            is_valid, struct_msg = verify_file_structure(
                file_path, file_config.get("required_columns")
            )
            messages.append(f"  {struct_msg}")
            if not is_valid:
                all_ok = False
    
    # Verificar archivos opcionales
    messages.append("\n\n📋 ARCHIVOS OPCIONALES:")
    messages.append("-" * 70)
    
    for file_name, file_config in OPTIONAL_FILES.items():
        messages.append(f"\n{file_name.upper()}:")
        messages.append(f"  Descripción: {file_config['description']}")
        
        file_path, find_msg = find_file(settings, file_config)
        messages.append(f"  {find_msg}")
    
    messages.append("\n" + "=" * 70)
    
    if all_ok:
        messages.append("✅ TODOS LOS ARCHIVOS REQUERIDOS ESTÁN PRESENTES")
    else:
        messages.append("❌ FALTAN ALGUNOS ARCHIVOS REQUERIDOS")
        messages.append("\nPara generar macro_features.parquet, ejecuta:")
        messages.append("  python scripts/generate_macro_features.py")
    
    messages.append("=" * 70 + "\n")
    
    return all_ok, messages


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verifica que todos los archivos de datos requeridos existan"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path al archivo de configuración base.yaml (opcional)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Solo mostrar errores (código de salida)",
    )
    
    args = parser.parse_args()
    
    # Cargar configuración
    if args.config:
        settings = Settings.from_yaml(args.config)
    else:
        config_path = BASE_DIR / "configs" / "base.yaml"
        if config_path.exists():
            settings = Settings.from_yaml(config_path)
        else:
            settings = Settings()
    
    # Verificar archivos
    all_ok, messages = verify_all_files(settings)
    
    # Mostrar mensajes
    if not args.quiet:
        for msg in messages:
            print(msg)
    
    # Retornar código de salida
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

