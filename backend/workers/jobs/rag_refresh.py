"""Job programado para refrescar embeddings en pgvector."""

from __future__ import annotations

from pathlib import Path

from caria.config.settings import Settings
from caria.embeddings.generator import EmbeddingGenerator
from caria.retrieval.vector_store import VectorStore


def run(dataset_path: Path, settings: Settings) -> None:
    generator = EmbeddingGenerator(settings=settings)
    vector_store = VectorStore.from_settings(settings)
    records = generator.embed_file(dataset_path)
    vector_store.upsert(records)

