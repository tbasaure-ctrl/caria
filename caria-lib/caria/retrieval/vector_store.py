"""Abstracción sobre pgvector para almacenamiento de embeddings."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, JSON, MetaData, String, Table, select, text

from caria.config.settings import Settings
from caria.data_access.db import factory


LOGGER = logging.getLogger("caria.retrieval.vector_store")


class VectorStore:
    def __init__(self, connection_uri: str, table_name: str, embedding_dim: int) -> None:
        self.connection_uri = connection_uri
        self.table_name = table_name
        self.metadata = MetaData()
        self.table = Table(
            table_name,
            self.metadata,
            Column("id", String(64), primary_key=True),
            Column("embedding", Vector(dim=embedding_dim)),
            Column("metadata", JSON, nullable=False),
        )
        # Intentar crear tabla, pero no fallar si hay problemas de conexión
        try:
            engine = factory.get_engine(connection_uri)
            
            # Ensure vector extension exists before creating tables
            with engine.begin() as connection:
                try:
                    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    LOGGER.info("Vector extension verified/created")
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Could not create vector extension (may already exist): %s", exc)
                    # Try to continue anyway - extension might already exist
            
            self.metadata.create_all(engine)
        except UnicodeDecodeError as exc:
            LOGGER.warning(
                "Error de encoding al conectar a PostgreSQL. "
                "Verifica que la connection string no tenga caracteres especiales mal codificados. "
                "Error: %s",
                exc,
            )
            raise
        except Exception as exc:
            LOGGER.warning("No se pudo crear tabla en PostgreSQL (esto es opcional): %s", exc)
            # Continuar sin tabla - se creará cuando se necesite
        self._embedding_dim = embedding_dim

    @classmethod
    def from_settings(cls, settings: Settings) -> "VectorStore":
        conn = settings.get("vector_store", "connection")
        table = settings.get("vector_store", "embedding_table", default="embeddings")
        # Dimensión por defecto actualizada para modelos locales (1024 para mxbai-embed-large-v1)
        embedding_dim = settings.get("retrieval", "embedding_dim", default=1024)
        if not conn:
            raise RuntimeError("vector_store.connection no configurado en settings")
        return cls(connection_uri=conn, table_name=table, embedding_dim=int(embedding_dim))

    def upsert(self, records: list[dict[str, Any]]) -> None:
        engine = factory.get_engine(self.connection_uri)
        with engine.begin() as connection:
            for record in records:
                embedding = np.array(record["embedding"], dtype=float)
                if embedding.shape[0] != self._embedding_dim:
                    raise ValueError(
                        f"Embedding recibido con dimensión {embedding.shape[0]} "
                        f"pero se esperaba {self._embedding_dim}"
                    )
                stmt = self.table.insert().values(
                    id=record["id"],
                    embedding=embedding,
                    metadata=record.get("metadata", {}),
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[self.table.c.id],
                    set_={"embedding": embedding, "metadata": record.get("metadata", {})},
                )
                connection.execute(stmt)

    def search(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        engine = factory.get_engine(self.connection_uri)
        with engine.connect() as connection:
            distance = self.table.c.embedding.l2_distance(embedding).label("distance")
            query = (
                select(
                    self.table.c.id,
                    self.table.c.metadata,
                    distance,
                )
                .order_by(distance)
                .limit(top_k)
            )
            result = connection.execute(query)
            rows: list[dict[str, Any]] = []
            for row in result:
                mapping = dict(row._mapping)
                distance_value = float(mapping.pop("distance", 0.0))
                mapping["score"] = 1.0 / (1.0 + distance_value)
                rows.append(mapping)
            return rows

