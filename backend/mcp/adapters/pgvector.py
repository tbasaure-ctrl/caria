"""Adaptador utilitario para pgvector."""

from __future__ import annotations

from typing import Any

from caria.retrieval.vector_store import VectorStore


def get_vector_store(settings: Any) -> VectorStore:
    return VectorStore.from_settings(settings)

