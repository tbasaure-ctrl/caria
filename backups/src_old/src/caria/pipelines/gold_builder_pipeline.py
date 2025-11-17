"""Pipeline para consolidar datos silver en datasets gold listos para entrenamiento."""

from __future__ import annotations

import json
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
    df = df.replace([np.inf, -np.inf], np.nan)
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

    # Handle date-only joins (macro/commodities data broadcast to all tickers)
    if "date" in merge_keys and "ticker" not in merge_keys:
        # Ensure date columns are datetime
        base["date"] = pd.to_datetime(base["date"])
        df["date"] = pd.to_datetime(df["date"])

        # Sort by date for merge_asof
        base = base.sort_values(["date"] + (["ticker"] if "ticker" in base.columns else []))
        df = df.sort_values("date")

        # Use merge_asof to forward-fill macro data to daily stock data
        # This handles mixed-frequency (monthly/quarterly macro -> daily stocks)
        return pd.merge_asof(
            base,
            df,
            on="date",
            direction="backward",
            suffixes=("", "_dup")
        )

    # Use merge_asof for mixed-frequency data (daily technicals + quarterly fundamentals)
    if "date" in merge_keys and "ticker" in merge_keys:
        # Check if frequencies are different (indicator: large difference in date counts)
        base_dates = base["date"].nunique() if "date" in base.columns else 0
        df_dates = df["date"].nunique() if "date" in df.columns else 0

        # If one has >10x more dates than the other, likely mixed frequency
        if base_dates > 0 and df_dates > 0 and (base_dates / df_dates > 10 or df_dates / base_dates > 10):
            LOGGER.info(f"Detected mixed frequency data (base: {base_dates} dates, new: {df_dates} dates). Using merge_asof")

            # Ensure date columns are datetime
            base["date"] = pd.to_datetime(base["date"])
            df["date"] = pd.to_datetime(df["date"])

            # Sort both dataframes by date and ticker
            base = base.sort_values(["ticker", "date"])
            df = df.sort_values(["ticker", "date"])

            # Use merge_asof to match each date with most recent available data
            # direction='backward' means use the last known value before or at the date
            return pd.merge_asof(
                base,
                df,
                on="date",
                by="ticker",
                direction="backward",
                suffixes=("", "_dup")
            )

    return base.merge(df, on=merge_keys, how="left")


def _compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    if "close" in df.columns:
        df = df.sort_values(["ticker", "date"])
        # Changed from 20 quarters (5 years) to 4 quarters (1 year) for more realistic prediction horizon
        df["target_return_20d"] = (
            df.groupby("ticker")["close"].pct_change(periods=4).shift(-4)
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


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    exclude_columns = {
        "date",
        "ticker",
        "period",
        "pair",
        "start_date",
        "end_date",
        "type",
        "region",
        "description",
        "sources",
        "regime_name",
        "feature_columns",
        "wisdom_features",  # Se maneja por separado
    }
    # ensure string-encoded json columns remain strings
    if "sources" in df.columns:
        df["sources"] = df["sources"].apply(
            lambda val: json.dumps(val) if isinstance(val, (list, dict)) else val
        )

    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number, "float", "int", "bool"]).columns
        if col not in exclude_columns and not col.startswith("target_")
    ]

    if not numeric_cols:
        LOGGER.warning("No se encontraron columnas numéricas para construir 'features'")
        df["features"] = [[] for _ in range(len(df))]
        return df, []

    feature_matrix = (
        df[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    feature_array = np.nan_to_num(
        feature_matrix.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0
    )

    # Agregar wisdom_features si existe
    if "wisdom_features" in df.columns:
        wisdom_array = np.array(df["wisdom_features"].tolist(), dtype=np.float32)
        wisdom_array = np.nan_to_num(wisdom_array, nan=0.0, posinf=0.0, neginf=0.0)
        # Concatenar features numéricas con wisdom_features
        feature_array = np.concatenate([feature_array, wisdom_array], axis=1)
        LOGGER.info(f"Wisdom features agregadas: {wisdom_array.shape[1]} dimensiones")

    df = df.copy()
    df.loc[:, numeric_cols] = feature_array[:, : len(numeric_cols)]
    df["features"] = feature_array.tolist()
    df["feature_columns"] = [numeric_cols] * len(df)
    return df, numeric_cols


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
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["target_return_20d"])
    if merged.empty:
        LOGGER.warning("Después de limpiar targets no quedó información utilizable para gold")
        return
    merged, feature_cols = _build_feature_matrix(merged)
    if not feature_cols:
        LOGGER.warning("Dataset gold sin features utilizable. Se aborta generación de splits")
        return

    splits = config.get("splits", {})
    for split_name, date_range in splits.items():
        start, end = date_range
        mask = (merged["date"] >= pd.to_datetime(start)) & (merged["date"] <= pd.to_datetime(end))
        split_df = merged.loc[mask].copy()
        if split_df.empty:
            LOGGER.warning("Split %s sin filas en el rango %s - %s", split_name, start, end)
            continue
        split_df = split_df.dropna(subset=["target_return_20d"])
        if split_df.empty:
            LOGGER.warning("Split %s quedó vacío tras eliminar targets nulos", split_name)
            continue
        if "target_return_20d" not in split_df.columns:
            LOGGER.warning("Split %s sin columna target_return_20d; se rellena con 0", split_name)
            split_df["target_return_20d"] = 0.0

        split_df.loc[:, feature_cols] = (
            split_df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        split_df.loc[:, feature_cols] = np.nan_to_num(
            split_df[feature_cols].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0
        )
        split_df["features"] = split_df[feature_cols].astype(np.float32).to_numpy().tolist()
        split_df["target"] = split_df["target_return_20d"].astype(np.float32)
        path = gold_path / f"{split_name}.parquet"
        _ensure_dir(path.parent)
        split_df.to_parquet(path, index=False)
        LOGGER.info("Guardado split %s con %d filas", split_name, len(split_df))


def run(settings: Settings, pipeline_config_path: str | None = None) -> None:
    if not pipeline_config_path:
        raise ValueError("pipeline_config_path es obligatorio para gold_builder_flow")
    gold_builder_flow(settings=settings, config_path=pipeline_config_path)


