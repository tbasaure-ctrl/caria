"""Cliente para Twitter API v2 (bearer token)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .base import IngestionSource


LOGGER = logging.getLogger("caria.ingestion.twitter")


class TwitterSource(IngestionSource):
    search_url = "https://api.twitter.com/2/tweets/search/recent"

    def __init__(self, output_dir: Path, bearer_token: str | None = None) -> None:
        super().__init__(output_dir)
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        if not self.bearer_token:
            raise RuntimeError("TWITTER_BEARER_TOKEN no configurado")

    def extract(
        self,
        query: str,
        max_results: int = 100,
        tweet_fields: str = "created_at,public_metrics,lang",
        **_: Any,
    ) -> list[dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        params = {"query": query, "max_results": max_results, "tweet.fields": tweet_fields}
        attempts = 0
        backoff = 5
        while attempts < 3:
            resp = requests.get(self.search_url, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                attempts += 1
                reset = resp.headers.get("x-rate-limit-reset")
                LOGGER.warning(
                    "Rate limit Twitter alcanzado (intento %s/3). Headers reset=%s. Esperando %ss",
                    attempts,
                    reset,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp.json().get("data", [])

        LOGGER.error("No se pudo obtener datos de Twitter tras múltiples intentos. Se omite la fuente.")
        return []

    def persist(self, records: list[dict[str, Any]], partition: str | None = None, **_: Any) -> Path:
        partition = partition or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        output_path = self.output_dir / "twitter" / partition
        output_path.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(records)
        file_path = output_path / "twitter_tweets.parquet"
        if frame.empty:
            LOGGER.warning("Twitter no devolvió registros; se guardará Parquet vacío")
            frame.to_parquet(file_path, index=False)
        else:
            frame.to_parquet(file_path, index=False)
            LOGGER.info("Guardado Twitter en %s", file_path)
        return file_path

