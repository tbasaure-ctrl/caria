"""Ingesta de series de commodities vía FRED."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from functools import lru_cache

import pandas as pd
import requests

from .base import IngestionSource


LOGGER = logging.getLogger("caria.ingestion.commodities")


SYMBOL_TO_SERIES: dict[str, str] = {
    "WTI": "DCOILWTICO",
    "BRENT": "DCOILBRENTEU",
    "GOLD": "GOLDPMGBD228NLBM",
    "SILVER": "SLVPRUSD",
    "COPPER": "PCOPPUSDM",
    "WHEAT": "PWHEAMTUSDM",
    "CORN": "PMAIZMTUSDM",
    "SOYBEAN": "PSOYBUSDQ",
}


class CommoditiesSource(IngestionSource):
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    series_url = "https://api.stlouisfed.org/fred/series"

    def __init__(self, output_dir: Path, api_key: str | None = None) -> None:
        super().__init__(output_dir)
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY no configurado para CommoditiesSource")

    @lru_cache(maxsize=32)
    def _series_metadata(self, series_id: str) -> dict[str, Any]:
        params = {"series_id": series_id, "api_key": self.api_key, "file_type": "json"}
        resp = requests.get(self.series_url, params=params, timeout=30)
        if resp.status_code == 400:
            LOGGER.warning("Serie %s sin metadatos via fred/series; usando fallback updates", series_id)
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
        symbols: list[str],
        start_date: str,
        end_date: str | None = None,
        frequency: str = "d",
        **_: Any,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for symbol in symbols:
            series_id = SYMBOL_TO_SERIES.get(symbol.upper())
            if not series_id:
                LOGGER.warning("Símbolo de commodity %s no mapeado. Se omite", symbol)
                continue
            meta = self._series_metadata(series_id)
            effective_start = pd.to_datetime(start_date) if start_date else None
            meta_start = pd.to_datetime(meta.get("observation_start")) if meta else None
            if effective_start is not None and meta_start is not None and effective_start < meta_start:
                LOGGER.warning(
                    "Ajustando inicio de %s (%s) a %s por disponibilidad FRED",
                    symbol,
                    series_id,
                    meta_start.date(),
                )
                effective_start = meta_start

            LOGGER.info("Descargando commodity %s (serie %s)", symbol, series_id)
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "frequency": frequency,
                "observation_start": (effective_start.strftime("%Y-%m-%d") if effective_start else start_date),
                "observation_end": end_date,
            }
            resp = requests.get(self.base_url, params=params, timeout=45)
            if resp.status_code == 400 and frequency:
                LOGGER.warning(
                    "400 para %s (%s) con frequency=%s; reintentando usando frecuencia original",
                    symbol,
                    series_id,
                    frequency,
                )
                params.pop("frequency", None)
                resp = requests.get(self.base_url, params=params, timeout=45)
            if resp.status_code == 400:
                LOGGER.error(
                    "Serie %s (%s) sigue fallando con 400. Se omite. Respuesta: %s",
                    symbol,
                    series_id,
                    resp.text,
                )
                continue
            resp.raise_for_status()
            payload = resp.json()
            observations = payload.get("observations", [])
            records.append(
                {
                    "symbol": symbol.upper(),
                    "series_id": series_id,
                    "observations": observations,
                    "frequency": payload.get("frequency", frequency),
                    "units": payload.get("units", ""),
                    "last_updated": payload.get("realtime_end"),
                }
            )
        return records

    def persist(self, records: list[dict[str, Any]], partition: str | None = None, **_: Any) -> Path:
        partition = partition or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        output_path = self.output_dir / "commodities" / partition
        output_path.mkdir(parents=True, exist_ok=True)

        frames: list[pd.DataFrame] = []
        for record in records:
            frame = pd.DataFrame(record["observations"])
            if frame.empty:
                continue
            frame["date"] = pd.to_datetime(frame["date"])
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame["symbol"] = record["symbol"]
            frame["series_id"] = record["series_id"]
            frame["frequency"] = record.get("frequency")
            frame["units"] = record.get("units")
            frames.append(frame[["date", "symbol", "value", "series_id", "frequency", "units"]])

        if not frames:
            LOGGER.warning("No se generaron observaciones de commodities")
            return output_path / "commodities.parquet"

        combined = pd.concat(frames, ignore_index=True)
        file_path = output_path / "commodities.parquet"
        combined.to_parquet(file_path, index=False)
        LOGGER.info("Guardado commodities en %s", file_path)
        return file_path

