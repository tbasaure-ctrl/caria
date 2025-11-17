"""Utilidades para dividir documentos en chunks."""

from __future__ import annotations

from typing import Iterable


def sliding_window(text: str, chunk_size: int = 400, overlap: int = 50) -> Iterable[str]:
    words = text.split()
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if not chunk:
            break
        yield " ".join(chunk)

