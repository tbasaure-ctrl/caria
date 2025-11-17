"""Ejecuta el pipeline de entrenamiento del detector HMM de régimen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Configurar paths antes de importar módulos de caria
# Este script está en scripts/orchestration/, necesitamos subir 2 niveles para llegar a caria_data/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv

from caria.config.settings import Settings
from caria.pipelines.regime_hmm_pipeline import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Caria HMM regime detection pipeline")
    # Paths relativos a BASE_DIR (caria_data/)
    default_config = str(BASE_DIR / "configs" / "base.yaml")
    default_pipeline_config = str(BASE_DIR / "configs" / "pipelines" / "regime_hmm.yaml")
    
    parser.add_argument("--config", default=default_config, help="Ruta al archivo base YAML")
    parser.add_argument(
        "--pipeline-config",
        default=default_pipeline_config,
        help="Archivo YAML con parámetros de entrenamiento HMM",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    
    # Convertir paths a absolutos si son relativos
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    
    pipeline_config_path = Path(args.pipeline_config)
    if not pipeline_config_path.is_absolute():
        pipeline_config_path = BASE_DIR / pipeline_config_path
    
    settings = Settings.from_yaml(config_path)
    run(settings=settings, pipeline_config_path=str(pipeline_config_path))


if __name__ == "__main__":
    main()

