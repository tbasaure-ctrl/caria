"""Cliente para la API de Reddit utilizando OAuth."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
import json
import math

import pandas as pd
import requests

from .base import IngestionSource


LOGGER = logging.getLogger("caria.ingestion.reddit")


class RedditSource(IngestionSource):
    token_url = "https://www.reddit.com/api/v1/access_token"
    base_url = "https://oauth.reddit.com"

    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = os.getenv("REDDIT_USER_AGENT", "caria-bot/0.1")
        if not self.client_id or not self.client_secret:
            raise RuntimeError("Credenciales de Reddit no configuradas")
        self._token: str | None = None

    def _ensure_token(self) -> str:
        if self._token:
            return self._token
        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        data = {"grant_type": "client_credentials"}
        headers = {"User-Agent": self.user_agent}
        resp = requests.post(self.token_url, auth=auth, data=data, headers=headers, timeout=30)
        resp.raise_for_status()
        token = resp.json()["access_token"]
        self._token = token
        return token

    def extract(self, subreddits: list[str], limit: int = 100, **_: Any) -> list[dict[str, Any]]:
        token = self._ensure_token()
        headers = {"Authorization": f"bearer {token}", "User-Agent": self.user_agent}
        posts: list[dict[str, Any]] = []
        for subreddit in subreddits:
            LOGGER.info("Descargando posts de r/%s", subreddit)
            resp = requests.get(
                f"{self.base_url}/r/{subreddit}/new",
                params={"limit": limit},
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
            for child in children:
                posts.append(child.get("data", {}))
        return posts

    def persist(self, records: list[dict[str, Any]], partition: str | None = None, **_: Any) -> Path:
        partition = partition or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        output_path = self.output_dir / "reddit" / partition
        output_path.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(records)

        if frame.empty:
            LOGGER.warning("No se recibieron posts de Reddit, omitiendo escritura")
            return output_path / "reddit_posts.parquet"

        if "edited" in frame.columns:
            def _normalize_edited(value: Any) -> bool:
                if value is None or value is pd.NA:
                    return False
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    if isinstance(value, float) and math.isnan(value):
                        return False
                    return value != 0
                if isinstance(value, str):
                    lowered = value.lower()
                    if lowered in {"false", "0", "", "none", "null", "nan"}:
                        return False
                    return True
                return bool(value)

            frame["edited"] = frame["edited"].apply(_normalize_edited).astype(bool)

        object_columns = frame.select_dtypes(include=["object"]).columns
        for column in object_columns:
            if frame[column].apply(lambda value: isinstance(value, (dict, list))).any():
                frame[column] = frame[column].apply(
                    lambda value: json.dumps(value) if isinstance(value, (dict, list)) else value
                )

        frame = frame.convert_dtypes()

        file_path = output_path / "reddit_posts.parquet"
        frame.to_parquet(file_path, index=False)
        LOGGER.info("Guardado Reddit en %s", file_path)
        return file_path

