"""Generador de embeddings para contenidos de sabiduría y noticias.

Soporta múltiples proveedores:
- local: Modelos locales usando sentence-transformers (recomendado)
- openai: APIs compatibles con el endpoint de embeddings de OpenAI
"""

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
            str(self.settings.get("retrieval", "provider", default="local"))
            .strip()
            .lower()
        )
        self.embedding_model = self.settings.get(
            "retrieval",
            "embedding_model",
            default="mixedbread-ai/mxbai-embed-large-v1",  # Modelo local recomendado
        )
        self.embedding_dim = int(
            self.settings.get("retrieval", "embedding_dim", default=1024)
        )
        
        # Inicializar variables de estado para lazy loading
        self._local_model = None
        
        if self.provider == "local":
            # Verificación preliminar de dependencias, pero no carga modelo
            try:
                import sentence_transformers  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers no está instalado; ejecuta `pip install sentence-transformers`"
                ) from exc
            
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY no configurado")
        else:
            raise ValueError(f"Proveedor de embeddings no soportado: {self.provider}")

    def _ensure_model_loaded(self) -> None:
        """Carga el modelo o cliente necesario bajo demanda."""
        if self.provider == "local":
            if self._local_model is None:
                from sentence_transformers import SentenceTransformer
                
                LOGGER.info("Cargando modelo local de embeddings (esto puede tomar unos segundos): %s", self.embedding_model)
                self._local_model = SentenceTransformer(self.embedding_model)
                
                # Obtener dimensión real del modelo
                actual_dim = self._local_model.get_sentence_embedding_dimension()
                if actual_dim != self.embedding_dim:
                    LOGGER.warning(
                        "Dimensión configurada (%d) no coincide con modelo (%d). Usando dimensión del modelo.",
                        self.embedding_dim,
                        actual_dim,
                    )
                    self.embedding_dim = actual_dim
                LOGGER.info("Modelo local cargado con dimensión: %d", self.embedding_dim)
                
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
        """Genera embedding para un texto usando el proveedor configurado."""
        # Asegurar carga del modelo antes de usar
        self._ensure_model_loaded()

        if self.provider == "local":
            # Modelo local usando sentence-transformers
            if self._local_model is None: # Should be loaded by _ensure_model_loaded
                 raise RuntimeError("Error interno: Modelo local no cargado")
                 
            embedding = self._local_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        
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
        
        raise RuntimeError(f"Proveedor desconocido: {self.provider}")

    def _embed_text(self, text: str) -> list[float]:  # noqa: D401
        """Mantiene compatibilidad retroactiva con llamadas anteriores."""

        return self.embed_text(text)
