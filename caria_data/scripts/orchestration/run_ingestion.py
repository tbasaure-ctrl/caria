"""Ejecuta el pipeline de ingesta completo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from caria.config.settings import Settings
from caria.pipelines.ingestion_pipeline import ingestion_flow, yaml_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Caria ingestion pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Ruta al archivo base YAML")
    parser.add_argument(
        "--pipeline-config",
        default="configs/pipelines/ingestion.yaml",
        help="Ruta al archivo de configuración del pipeline",
    )
    return parser.parse_args()


def read_pipeline_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".json":
            return json.load(handle)
        return yaml_load(handle)


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_yaml(Path(args.config))
    pipeline_config = read_pipeline_config(Path(args.pipeline_config))
    ingestion_flow(config=pipeline_config, settings=settings)


if __name__ == "__main__":
    main()

