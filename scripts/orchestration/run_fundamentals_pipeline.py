"""Ejecuta el pipeline de fundamentals y price action."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from caria.config.settings import Settings
from caria.pipelines.fundamentals_pipeline import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Caria fundamentals pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Ruta al archivo base YAML")
    parser.add_argument(
        "--pipeline-config",
        default="configs/pipelines/fundamentals.yaml",
        help="Archivo YAML con universe y parámetros",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_yaml(Path(args.config))
    run(settings=settings, pipeline_config_path=args.pipeline_config)


if __name__ == "__main__":
    main()

