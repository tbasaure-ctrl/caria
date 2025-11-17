"""Ejecuta el pipeline de entrenamiento."""

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
from caria.pipelines.training_pipeline import training_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Caria training pipeline")
    parser.add_argument("--config", default="caria/configs/base.yaml", help="Ruta al archivo base YAML")
    return parser.parse_args()


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_yaml(_resolve_path(args.config))
    training_flow(settings=settings)


if __name__ == "__main__":
    main()
