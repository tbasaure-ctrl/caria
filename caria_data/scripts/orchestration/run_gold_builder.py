"""Ejecuta el pipeline de construcción de datasets gold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parents[2]
POTENTIAL_SRC_DIRS = [
    BASE_DIR / "src",
    BASE_DIR / "caria" / "src",
]
for candidate in POTENTIAL_SRC_DIRS:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from caria.config.settings import Settings
from caria.pipelines.gold_builder_pipeline import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Caria gold builder pipeline")
    parser.add_argument("--config", default="caria/configs/base.yaml", help="Ruta al archivo base YAML")
    parser.add_argument(
        "--pipeline-config",
        default="caria/configs/pipelines/gold_builder.yaml",
        help="Archivo YAML con datasets y splits",
    )
    return parser.parse_args()


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def main() -> None:
    load_dotenv()
    args = parse_args()
    config_path = _resolve_path(args.config)
    pipeline_config_path = _resolve_path(args.pipeline_config)
    settings = Settings.from_yaml(config_path)
    run(settings=settings, pipeline_config_path=str(pipeline_config_path))


if __name__ == "__main__":
    main()
