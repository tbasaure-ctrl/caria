"""Ingesta de tasas de cambio vía FRED."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from functools import lru_cache

import pandas as pd
import requests

from .base import IngestionSource


LOGGER = logging.getLogger("caria.ingestion.fx")


PAIR_TO_SERIES: dict[str, str] = {
    "DXY": "DTWEXBGS",
    "EURUSD": "DEXUSEU",
    "GBPUSD": "DEXUSUK",
    "USDJPY": "DEXJPUS",
    "USDCNY": "DEXCHUS",
}


INVERT_SERIES: set[str] = {"DEXUSEU", "DEXUSUK"}


class FXSource(IngestionSource):
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    series_url = "https://api.stlouisfed.org/fred/series"

    def __init__(self, output_dir: Path, api_key: str | None = None) -> None:
        super().__init__(output_dir)
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY no configurado para FXSource")

    @lru_cache(maxsize=32)
    def _series_metadata(self, series_id: str) -> dict[str, Any]:
        params = {"series_id": series_id, "api_key": self.api_key, "file_type": "json"}
        resp = requests.get(self.series_url, params=params, timeout=30)
        if resp.status_code == 400:
            LOGGER.warning("Serie FX %s sin metadatos via fred/series; usando fallback updates", series_id)
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/updates",
                params={"series_id": series_id, "api_key": self.api_key, "file_type": "json"},
                timeout=30,
            )
        resp.raise_for_status()
        data = resp.json().get("seriess", [])
        return data[0] if data else {}

    def extract(
        self,
        pairs: list[str],
        start_date: str,
        end_date: str | None = None,
        frequency: str = "d",
        **_: Any,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for pair in pairs:
            series_id = PAIR_TO_SERIES.get(pair.upper())
            if not series_id:
                LOGGER.warning("Par FX %s no mapeado. Se omite", pair)
                continue

            meta = self._series_metadata(series_id)
            effective_start = pd.to_datetime(start_date) if start_date else None
            meta_start = pd.to_datetime(meta.get("observation_start")) if meta else None
            if effective_start is not None and meta_start is not None and effective_start < meta_start:
                LOGGER.warning(
                    "Ajustando inicio FX %s (%s) a %s por disponibilidad FRED",
                    pair,
                    series_id,
                    meta_start.date(),
                )
                effective_start = meta_start

            LOGGER.info("Descargando FX %s (serie %s)", pair, series_id)
            resp = requests.get(
                self.base_url,
                params={
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "frequency": frequency,
                    "observation_start": (effective_start.strftime("%Y-%m-%d") if effective_start else start_date),
                    "observation_end": end_date,
                },
                timeout=45,
            )
            resp.raise_for_status()
            payload = resp.json()
            records.append(
                {
                    "pair": pair.upper(),
                    "series_id": series_id,
                    "observations": payload.get("observations", []),
                    "frequency": payload.get("frequency", frequency),
                    "units": payload.get("units", ""),
                }
            )
        return records

    def persist(self, records: list[dict[str, Any]], partition: str | None = None, **_: Any) -> Path:
        partition = partition or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        output_path = self.output_dir / "fx" / partition
        output_path.mkdir(parents=True, exist_ok=True)

        frames: list[pd.DataFrame] = []
        for record in records:
            frame = pd.DataFrame(record["observations"])
            if frame.empty:
                continue
            frame["date"] = pd.to_datetime(frame["date"])
            frame["value_raw"] = pd.to_numeric(frame["value"], errors="coerce")
            series_id = record["series_id"]
            invert = series_id in INVERT_SERIES
            if invert:
                frame["close"] = 1 / frame["value_raw"]
            else:
                frame["close"] = frame["value_raw"]
            frame["pair"] = record["pair"]
            frame["series_id"] = series_id
            frame["frequency"] = record.get("frequency")
            frame["units"] = record.get("units")
            frame["invert_rate"] = invert
            frames.append(frame[["date", "pair", "close", "series_id", "frequency", "units", "invert_rate"]])

        if not frames:
            LOGGER.warning("No se generaron observaciones FX")
            return output_path / "fx_rates.parquet"

        combined = pd.concat(frames, ignore_index=True)
        file_path = output_path / "fx_rates.parquet"
        combined.to_parquet(file_path, index=False)
        LOGGER.info("Guardado FX en %s", file_path)
        return file_path

