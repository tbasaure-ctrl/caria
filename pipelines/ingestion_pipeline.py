"""Pipeline de ingesta orquestado con Prefect."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, task

from caria.config.settings import Settings
from caria.ingestion.registry import build_source


LOGGER = logging.getLogger("caria.pipelines.ingestion")


@task(name="extract-source")
def extract_source(source_conf: dict[str, Any], output_dir: Path) -> Path:
    source = build_source(source_conf["type"], output_dir=output_dir)
    params = source_conf.get("params", {})
    partition = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    LOGGER.info("Ejecutando fuente %s", source_conf["name"])
    return source.run(partition=partition, **params)


@task(name="log-output")
def log_output(path: Path) -> None:
    LOGGER.info("Datos guardados en %s", path)


@flow(name="caria-ingestion")
def ingestion_flow(config: dict[str, Any], settings: Settings) -> list[Path]:
    LOGGER.info("Iniciando flow de ingesta con %s fuentes", len(config.get("sources", [])))
    output_root = Path(settings.get("storage", "raw_path", default="data/raw"))
    created_paths = []
    for source_conf in config.get("sources", []):
        path = extract_source(source_conf, output_dir=output_root)
        log_output(path)
        created_paths.append(path)
    return created_paths


def run(settings: Settings, pipeline_config_path: str | None) -> None:
    if not pipeline_config_path:
        raise ValueError("Se requiere pipeline_config_path para la ingesta")
    with open(pipeline_config_path, "r", encoding="utf-8") as handle:
        pipeline_config = json.load(handle) if pipeline_config_path.endswith(".json") else yaml_load(handle)
    ingestion_flow(config=pipeline_config, settings=settings)


def yaml_load(handle: Any) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(handle) or {}

