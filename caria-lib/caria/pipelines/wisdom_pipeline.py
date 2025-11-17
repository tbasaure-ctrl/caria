"""Pipeline de ingestión de textos de sabiduría hacia pgvector."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from prefect import flow, task

from caria.config.settings import Settings
from caria.embeddings.generator import EmbeddingGenerator
from caria.pipelines.embedding_pipeline import upsert_vector_store
from caria.retrieval.vector_store import VectorStore
from caria.wisdom.loader import WisdomLoader, discover_wisdom_files


LOGGER = logging.getLogger("caria.pipelines.wisdom")


@task(name="discover-wisdom-files")
def discover_task(raw_dir: Path) -> list[Path]:
    return discover_wisdom_files(raw_dir)


@task(name="load-wisdom-documents")
def load_task(
    root_dir: Path,
    files: list[Path],
    version: str,
    chunk_size: int,
    overlap: int,
) -> pd.DataFrame:
    loader = WisdomLoader(root_dir=root_dir, version=version, chunk_size=chunk_size, overlap=overlap)
    return loader.load(files)


@task(name="persist-wisdom-dataset")
def persist_task(frame: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "wisdom.parquet"
    frame.to_parquet(dataset_path, index=False)
    LOGGER.info("Dataset de sabiduría persistido en %s (%s registros)", dataset_path, len(frame))
    return dataset_path


@task(name="generate-wisdom-embeddings")
def embed_task(settings: Settings, dataset_path: Path) -> list[dict[str, object]]:
    generator = EmbeddingGenerator(settings=settings)
    return generator.embed_file(dataset_path)


@flow(name="caria-wisdom-ingest")
def wisdom_ingest_flow(
    settings: Settings,
    raw_dir: Path,
    version: str | None = None,
    chunk_size: int = 800,
    overlap: int = 120,
) -> None:
    version = version or datetime.utcnow().strftime("%Y%m%d")
    files = discover_task(raw_dir)
    frame = load_task(root_dir=raw_dir, files=files, version=version, chunk_size=chunk_size, overlap=overlap)

    silver_base = Path(settings.get("storage", "silver_path", default="data/silver"))
    dataset_path = persist_task(frame, silver_base / "wisdom" / version)

    embeddings = embed_task(settings, dataset_path)

    if not embeddings:
        raise RuntimeError("No se generaron embeddings para subir a pgvector")

    vector_store = VectorStore.from_settings(settings)
    upsert_vector_store(vector_store, embeddings)


def run(settings: Settings, raw_dir: Path | None = None, **kwargs: object) -> None:
    default_raw = Path(settings.get("storage", "raw_path", default="data/raw")) / "wisdom"
    wisdom_ingest_flow(
        settings=settings,
        raw_dir=raw_dir or default_raw,
        version=kwargs.get("version"),
        chunk_size=int(kwargs.get("chunk_size", 800)),
        overlap=int(kwargs.get("overlap", 120)),
    )


