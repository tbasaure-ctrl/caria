"""Cliente para la API de FRED."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .base import IngestionSource


LOGGER = logging.getLogger("caria.ingestion.fred")


class FREDSource(IngestionSource):
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, output_dir: Path, api_key: str | None = None) -> None:
        super().__init__(output_dir)
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY no configurado en entorno")

    def extract(
        self,
        series_ids: list[str],
        start_date: str,
        end_date: str | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for series_id in series_ids:
            LOGGER.info("Descargando serie FRED %s", series_id)
            resp = requests.get(
                self.base_url,
                params={
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "observation_start": start_date,
                    "observation_end": end_date,
                },
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
            records.append({"series_id": series_id, "observations": payload.get("observations", [])})
        return records

    def persist(self, records: list[dict[str, Any]], partition: str | None = None, **_: Any) -> Path:
        partition = partition or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        output_path = self.output_dir / "fred" / partition
        output_path.mkdir(parents=True, exist_ok=True)

        frames = []
        for record in records:
            frame = pd.DataFrame(record["observations"])
            frame["series_id"] = record["series_id"]
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True)
        file_path = output_path / "fred_data.parquet"
        combined.to_parquet(file_path, index=False)
        LOGGER.info("Guardado FRED en %s", file_path)
        return file_path

