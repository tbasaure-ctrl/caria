"""Cliente para NewsAPI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .base import IngestionSource


LOGGER = logging.getLogger("caria.ingestion.newsapi")


class NewsAPISource(IngestionSource):
    base_url = "https://newsapi.org/v2/everything"

    def __init__(self, output_dir: Path, api_key: str | None = None) -> None:
        super().__init__(output_dir)
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise RuntimeError("NEWS_API_KEY no configurado en entorno")

    def extract(self, query: str, page_size: int = 50, language: str = "en", **_: Any) -> list[dict[str, Any]]:
        resp = requests.get(
            self.base_url,
            params={"q": query, "pageSize": page_size, "language": language},
            headers={"Authorization": self.api_key},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("articles", [])

    def persist(self, records: list[dict[str, Any]], partition: str | None = None, **_: Any) -> Path:
        partition = partition or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        output_path = self.output_dir / "newsapi" / partition
        output_path.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(records)
        file_path = output_path / "newsapi_articles.parquet"
        frame.to_parquet(file_path, index=False)
        LOGGER.info("Guardado NewsAPI en %s", file_path)
        return file_path

