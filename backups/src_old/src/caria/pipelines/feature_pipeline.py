"""Pipeline para construir capas Bronze/Silver y publicar en Feast."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from prefect import flow, task

from caria.config.settings import Settings


LOGGER = logging.getLogger("caria.pipelines.features")


@task(name="bronze-to-silver")
def transform_bronze_to_silver(raw_path: Path, silver_path: Path) -> list[Path]:
    silver_path.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for dataset in ("macro", "micro", "sentiment"):
        source = raw_path / dataset
        if not source.exists():
            LOGGER.warning("Dataset %s no encontrado en %s", dataset, source)
            continue
        table = pl.read_parquet(source.glob("*.parquet"))
        enriched = table.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
        out_file = silver_path / f"{dataset}.parquet"
        enriched.write_parquet(out_file)
        outputs.append(out_file)
    return outputs


@task(name="publish-features")
def publish_features(silver_files: list[Path]) -> None:
    LOGGER.info("Publicando %s features en Feast", len(silver_files))
    # Integración con Feast pendiente; se documenta en docs/architecture.md


@flow(name="caria-feature-pipeline")
def feature_flow(settings: Settings) -> list[Path]:
    raw_path = Path(settings.get("storage", "bronze_path", default="data/bronze"))
    silver_path = Path(settings.get("storage", "silver_path", default="data/silver"))
    silver_files = transform_bronze_to_silver(raw_path=raw_path, silver_path=silver_path)
    publish_features(silver_files)
    return silver_files


def run(settings: Settings, pipeline_config_path: str | None = None) -> None:
    feature_flow(settings=settings)

