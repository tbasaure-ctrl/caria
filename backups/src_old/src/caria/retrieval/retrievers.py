"""Utilidades para búsquedas semánticas en Caria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from caria.retrieval.vector_store import VectorStore


@dataclass(slots=True)
class RetrievalResult:
    id: str
    score: float
    metadata: dict[str, Any]


class Retriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def query(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        rows = self.vector_store.search(embedding, top_k=top_k)
        results: list[RetrievalResult] = []
        for row in rows:
            results.append(
                RetrievalResult(
                    id=row["id"],
                    score=row.get("score", 0.0),
                    metadata=row.get("metadata", {}),
                )
            )
        return results

