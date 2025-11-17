"""Utilidades para cargar documentos de sabiduría y prepararlos para embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger("caria.wisdom.loader")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Divide un texto en fragmentos solapados."""

    words = text.split()
    if not words:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size debe ser mayor que overlap")

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
    return chunks


def _normalize_tags(tags: object) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, (list, tuple, set)):
        return [str(tag) for tag in tags]
    if isinstance(tags, str):
        separators = [",", ";", "|"]
        for sep in separators:
            if sep in tags:
                return [token.strip() for token in tags.split(sep) if token.strip()]
        return [tags.strip()]
    return [str(tags)]


@dataclass(slots=True)
class WisdomLoader:
    root_dir: Path
    version: str
    chunk_size: int = 800
    overlap: int = 120

    def load(self, files: Iterable[Path]) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for file_path in files:
            suffix = file_path.suffix.lower()
            if suffix == ".jsonl":
                records.extend(self._load_jsonl(file_path))
            elif suffix in {".md", ".txt"}:
                records.extend(self._load_text(file_path))
            else:
                LOGGER.warning("Formato no soportado en %s, se omite", file_path)
        if not records:
            raise ValueError("No se generaron registros de sabiduría; revisa los archivos de entrada")
        return pd.DataFrame(records)

    def _load_jsonl(self, path: Path) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.warning(
                        "Registro en %s (línea %s) con JSON inválido (%s); se omite",
                        path,
                        index + 1,
                        exc,
                    )
                    continue
                content = data.get("content")
                if not content:
                    LOGGER.warning("Registro en %s (línea %s) sin contenido; se omite", path, index + 1)
                    continue
                metadata = {k: v for k, v in data.items() if k != "content"}
                if "title" not in metadata:
                    metadata["title"] = metadata.get("id") or path.stem
                if self.root_dir in path.parents:
                    metadata.setdefault("source", str(path.relative_to(self.root_dir)))
                items.extend(self._materialize_chunks(content, metadata))
        return items

    def _load_text(self, path: Path) -> list[dict[str, object]]:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            LOGGER.warning("Archivo vacío detectado: %s", path)
            return []
        metadata = {
            "title": path.stem.replace("_", " ").title(),
            "source": str(path.relative_to(self.root_dir)),
        }
        return self._materialize_chunks(content, metadata)

    def _materialize_chunks(self, content: str, metadata: dict[str, object]) -> list[dict[str, object]]:
        chunks = chunk_text(content, chunk_size=self.chunk_size, overlap=self.overlap)
        if not chunks:
            return []
        raw_id = metadata.get("id") or self._hash_identifier(metadata.get("title", "document"))
        if isinstance(raw_id, str):
            document_id = raw_id
        else:
            document_id = str(raw_id)
        tags = _normalize_tags(metadata.get("tags"))
        records: list[dict[str, object]] = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{document_id}:{idx:03d}"
            record = {
                "id": chunk_id,
                "document_id": document_id,
                "version": self.version,
                "chunk_index": idx,
                "chunk_total": total,
                "content": chunk,
                "title": metadata.get("title"),
                "summary": metadata.get("summary"),
                "source": metadata.get("source"),
                "author": metadata.get("author"),
                "year": metadata.get("year"),
                "tags": tags,
            }
            records.append(record)
        return records

    def _hash_identifier(self, value: object) -> str:
        text = f"{self.version}:{value}"
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def discover_wisdom_files(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio de sabiduría en {root_dir}")
    files = [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jsonl", ".md", ".txt"}
    ]
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos de sabiduría en {root_dir}")
    return sorted(files)
