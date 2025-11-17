"""Ejecuta el pipeline de anotación de regímenes y noticias."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from caria.config.settings import Settings
from caria.pipelines.regime_annotation_pipeline import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Caria regime annotation pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Ruta al archivo base YAML")
    parser.add_argument(
        "--pipeline-config",
        default="configs/pipelines/regime_annotation.yaml",
        help="Archivo YAML con parámetros de anotación",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_yaml(Path(args.config))
    run(settings=settings, pipeline_config_path=args.pipeline_config)


if __name__ == "__main__":
    main()
