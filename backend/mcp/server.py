"""Servidor MCP para búsquedas semánticas."""

from __future__ import annotations

from fastapi import FastAPI

from caria.config.settings import Settings
from caria.embeddings.generator import EmbeddingGenerator
from caria.retrieval.retrievers import Retriever
from caria.retrieval.vector_store import VectorStore


app = FastAPI(title="Caria MCP Server")
settings = Settings.from_yaml(base_path=__import__("pathlib").Path("configs/base.yaml"))
vector_store = VectorStore.from_settings(settings)
retriever = Retriever(vector_store=vector_store)
embedder = EmbeddingGenerator(settings=settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
def search(payload: dict[str, str]) -> dict[str, object]:
    query = payload.get("query")
    if not query:
        return {"results": []}
    embedding = embedder.embed_text(query)
    results = retriever.query(embedding, top_k=payload.get("top_k", 5))
    return {"results": [result.__dict__ for result in results]}

