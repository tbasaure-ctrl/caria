"""Pipeline de validación de datos usando reglas declarativas."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from prefect import flow, task

from caria.config.settings import Settings


LOGGER = logging.getLogger("caria.pipelines.validation")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@task
def load_validation_config(path: str) -> dict[str, Any]:
    return _load_yaml(Path(path))


def _check_required_columns(df: pd.DataFrame, required: list[str], dataset: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset {dataset} sin columnas requeridas: {missing}")


def _check_not_all_null(df: pd.DataFrame, columns: list[str], dataset: str) -> None:
    problematic = [col for col in columns if df[col].isna().all()]
    if problematic:
        raise ValueError(f"Dataset {dataset} con columnas completamente vacías: {problematic}")


def _check_unique(df: pd.DataFrame, keys: list[str], dataset: str) -> None:
    if not keys or any(key not in df.columns for key in keys):
        return
    if df.duplicated(subset=keys).any():
        raise ValueError(f"Dataset {dataset} contiene duplicados para claves {keys}")


def _check_value_ranges(df: pd.DataFrame, ranges: dict[str, dict[str, float]], dataset: str) -> None:
    for column, bounds in ranges.items():
        if column not in df.columns:
            continue
        min_val = bounds.get("min")
        max_val = bounds.get("max")
        if min_val is not None and (df[column] < min_val).any():
            raise ValueError(f"Dataset {dataset} tiene valores menores a {min_val} en {column}")
        if max_val is not None and (df[column] > max_val).any():
            raise ValueError(f"Dataset {dataset} tiene valores mayores a {max_val} en {column}")


@task
def validate_dataset(base_path: Path, dataset_cfg: dict[str, Any]) -> None:
    relative_path = dataset_cfg["path"]
    dataset_path = base_path / relative_path
    if not dataset_path.exists():
        LOGGER.warning("Dataset %s no encontrado. Se omite validación", dataset_path)
        return
    df = pd.read_parquet(dataset_path)
    LOGGER.info("Validando %s (%d filas, %d columnas)", dataset_path, len(df), len(df.columns))

    required = dataset_cfg.get("required_columns", [])
    if required:
        _check_required_columns(df, required, relative_path)

    not_null = dataset_cfg.get("not_null_columns", [])
    if not_null:
        _check_not_all_null(df, not_null, relative_path)

    unique_keys = dataset_cfg.get("unique_keys", [])
    if unique_keys:
        _check_unique(df, unique_keys, relative_path)

    ranges = dataset_cfg.get("value_ranges", {})
    if ranges:
        _check_value_ranges(df, ranges, relative_path)


@flow(name="caria-validation")
def validation_flow(settings: Settings, config_path: str) -> None:
    config = load_validation_config(config_path)
    datasets = config.get("datasets", [])
    silver_path = Path(settings.get("storage", "silver_path", default="data/silver"))
    gold_path = Path(settings.get("storage", "gold_path", default="data/gold"))

    for dataset_cfg in datasets:
        location = dataset_cfg.get("location", "silver")
        base_path = silver_path if location == "silver" else gold_path
        validate_dataset.submit(base_path, dataset_cfg)


def run(settings: Settings, pipeline_config_path: str | None = None) -> None:
    if not pipeline_config_path:
        raise ValueError("pipeline_config_path es obligatorio para validation_flow")
    validation_flow(settings=settings, config_path=pipeline_config_path)

