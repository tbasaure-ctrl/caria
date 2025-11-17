"""CLI de apoyo para Caria.

Permite ejecutar pipelines declarados (ingesta, features, entrenamiento) y
comandar tareas operativas desde terminal.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path

from caria.config.settings import Settings


LOGGER = logging.getLogger("caria.cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="Caria command line interface")
    parser.add_argument(
        "command",
        choices=["ingest", "features", "embeddings", "train"],
        help="Pipeline a ejecutar",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Ruta al archivo de configuración base (Hydra)",
    )
    parser.add_argument(
        "--pipeline-config",
        required=False,
        help="Archivo YAML específico del pipeline",
    )

    args = parser.parse_args()
    settings = Settings.from_yaml(Path(args.config))

    command_module = importlib.import_module(f"caria.pipelines.{args.command}_pipeline")
    if not hasattr(command_module, "run"):
        msg = f"El pipeline {args.command} no implementa la función run()"
        raise RuntimeError(msg)

    LOGGER.info("Ejecutando pipeline %s", args.command)
    command_module.run(settings=settings, pipeline_config_path=args.pipeline_config)


if __name__ == "__main__":
    main()

