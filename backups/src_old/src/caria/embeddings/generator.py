"""Generador de embeddings para contenidos de sabiduría y noticias."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from caria.config.settings import Settings


LOGGER = logging.getLogger("caria.embeddings.generator")


class EmbeddingGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = (
            str(self.settings.get("retrieval", "provider", default="openai"))
            .strip()
            .lower()
        )
        self.embedding_model = self.settings.get(
            "retrieval",
            "embedding_model",
            default="text-embedding-3-small",
        )
        self.embedding_dim = int(
            self.settings.get("retrieval", "embedding_dim", default=1536)
        )
        if self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY no configurado")
        elif self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY no configurado")
            try:
                import google.generativeai as genai  # type: ignore[import]
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "google-generativeai no está instalado; ejecuta `poetry install`"
                ) from exc
            genai.configure(api_key=self.api_key)
            self._gemini = genai
            if self.embedding_model == "text-embedding-3-small":
                # Ajuste de valor por defecto cuando se usa Gemini
                self.embedding_model = "models/text-embedding-004"
                self.embedding_dim = 768
        else:
            raise ValueError(f"Proveedor de embeddings no soportado: {self.provider}")

    def embed_file(self, dataset_path: Path) -> list[dict[str, Any]]:
        frame = pd.read_parquet(dataset_path)
        records: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            content = row.pop("content", "")
            if not content:
                LOGGER.warning("Registro sin contenido detectado, se omite: %s", row.get("id"))
                continue
            embedding = self.embed_text(content)
            metadata = row.copy()
            metadata["content"] = content
            record = {
                "id": metadata.get("id"),
                "embedding": embedding,
                "metadata": metadata,
            }
            records.append(record)
        return records

    def embed_text(self, text: str) -> list[float]:
        if self.provider == "openai":
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.embedding_model, "input": text},
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        if self.provider == "gemini":
            result = self._gemini.embed_content(
                model=self.embedding_model,
                content=text,
            )
            embedding = result.get("embedding") if result else None
            if not embedding:
                raise RuntimeError("Gemini no generó un embedding válido")
            return list(embedding)
        raise RuntimeError(f"Proveedor desconocido: {self.provider}")

    def _embed_text(self, text: str) -> list[float]:  # noqa: D401
        """Mantiene compatibilidad retroactiva con llamadas anteriores."""

        return self.embed_text(text)

