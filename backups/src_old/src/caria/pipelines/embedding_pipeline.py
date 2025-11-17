"""Pipeline de embeddings y carga a pgvector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from prefect import flow, task

from caria.config.settings import Settings
from caria.embeddings.generator import EmbeddingGenerator
from caria.retrieval.vector_store import VectorStore


LOGGER = logging.getLogger("caria.pipelines.embeddings")


@task(name="generate-embeddings")
def generate_embeddings(generator: EmbeddingGenerator, dataset_path: Path) -> list[dict[str, Any]]:
    return generator.embed_file(dataset_path)


@task(name="upsert-vector-store")
def upsert_vector_store(vector_store: VectorStore, records: list[dict[str, Any]]) -> None:
    vector_store.upsert(records)


@flow(name="caria-embedding-pipeline")
def embedding_flow(dataset_path: Path, settings: Settings) -> None:
    generator = EmbeddingGenerator(settings=settings)
    vector_store = VectorStore.from_settings(settings)
    records = generate_embeddings(generator, dataset_path)
    upsert_vector_store(vector_store, records)


def run(settings: Settings, pipeline_config_path: str | None = None) -> None:
    if not pipeline_config_path:
        raise ValueError("Se requiere pipeline_config_path para embeddings")
    dataset_path = Path(pipeline_config_path)
    embedding_flow(dataset_path=dataset_path, settings=settings)

