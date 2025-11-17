"""Ejecuta el pipeline de features."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from caria.config.settings import Settings
from caria.pipelines.feature_pipeline import feature_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Caria feature pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Ruta al archivo base YAML")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_yaml(Path(args.config))
    feature_flow(settings=settings)


if __name__ == "__main__":
    main()

