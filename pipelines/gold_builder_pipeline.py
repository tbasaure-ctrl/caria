"""Pipeline para consolidar datos silver en datasets gold listos para entrenamiento."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from prefect import flow, task

from caria.config.settings import Settings


LOGGER = logging.getLogger("caria.pipelines.gold_builder")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@task
def load_gold_config(path: str) -> dict[str, Any]:
    return _load_yaml(Path(path))


@task
def load_dataset(path: Path, join_keys: list[str]) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Dataset %s no existe. Se omite", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    missing = [key for key in join_keys if key not in df.columns]
    for key in missing:
        if key == "ticker":
            df[key] = None
        elif key == "date":
            df[key] = pd.NaT
    return df


def _merge_datasets(base: pd.DataFrame, df: pd.DataFrame, join_keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return base
    if base is None or base.empty:
        return df
    merge_keys = [key for key in join_keys if key in df.columns and key in base.columns]
    if not merge_keys:
        LOGGER.warning("Sin claves comunes para merge. Se usa left join por columnas existentes")
        merge_keys = list(set(df.columns).intersection(base.columns))
    return base.merge(df, on=merge_keys, how="left")


def _compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    if "close" in df.columns:
        df = df.sort_values(["ticker", "date"])
        df["target_return_20d"] = (
            df.groupby("ticker")["close"].pct_change(periods=20).shift(-20)
        )
    elif "returns_20d" in df.columns:
        df["target_return_20d"] = df["returns_20d"].shift(-1)
    else:
        df["target_return_20d"] = np.nan

    if "drawdown" in df.columns:
        df["target_drawdown_prob"] = (df["drawdown"].rolling(window=20).min() < -0.1).astype(float)
    else:
        df["target_drawdown_prob"] = np.nan

    if "type" in df.columns and "start_date" in df.columns:
        df["target_regime"] = df.get("regime_label", np.nan)
    elif "regime_label" in df.columns:
        df["target_regime"] = df["regime_label"]
    else:
        df["target_regime"] = np.nan
    return df


def _label_regimes(merged: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return merged
    events = events.copy()
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])
    merged["regime_label"] = 0
    merged["regime_name"] = "normal"
    for _, row in events.iterrows():
        mask = (merged["date"] >= row["start_date"]) & (merged["date"] <= row["end_date"])
        label = row.get("type", "event")
        code = {
            "recession": 1,
            "depression": 2,
            "crisis": 3,
            "mania": 4,
        }.get(label, 5)
        merged.loc[mask, "regime_label"] = code
        merged.loc[mask, "regime_name"] = label
    return merged


@flow(name="caria-gold-builder")
def gold_builder_flow(settings: Settings, config_path: str) -> None:
    config = load_gold_config(config_path)
    silver_path = Path(settings.get("storage", "silver_path", default="data/silver"))
    gold_path = Path(settings.get("storage", "gold_path", default="data/gold"))

    datasets = config.get("datasets", [])
    merged: pd.DataFrame | None = None

    events_df = pd.DataFrame()

    for dataset in datasets:
        rel_path = dataset["path"]
        join_keys = dataset.get("join_keys", ["date", "ticker"])
        df = load_dataset(silver_path / rel_path, join_keys)
        if dataset.get("type") == "events":
            events_df = df
            continue
        merged = _merge_datasets(merged, df, join_keys)

    if merged is None or merged.empty:
        LOGGER.warning("No se generó dataset gold por falta de datos mergeables")
        return

    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"])
    else:
        LOGGER.warning("Dataset gold sin columna date. Se agregará fecha nula")
        merged["date"] = pd.NaT

    if events_df is not None and not events_df.empty:
        merged = _label_regimes(merged, events_df)

    merged = _compute_targets(merged)

    splits = config.get("splits", {})
    for split_name, date_range in splits.items():
        start, end = date_range
        mask = (merged["date"] >= pd.to_datetime(start)) & (merged["date"] <= pd.to_datetime(end))
        split_df = merged.loc[mask].copy()
        path = gold_path / f"{split_name}.parquet"
        _ensure_dir(path.parent)
        split_df.to_parquet(path, index=False)
        LOGGER.info("Guardado split %s con %d filas", split_name, len(split_df))


def run(settings: Settings, pipeline_config_path: str | None = None) -> None:
    if not pipeline_config_path:
        raise ValueError("pipeline_config_path es obligatorio para gold_builder_flow")
    gold_builder_flow(settings=settings, config_path=pipeline_config_path)

