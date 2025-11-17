"""Clases base para conectores de ingesta."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any


class IngestionSource(abc.ABC):
    """Interfaz para fuentes de datos externas."""

    name: str

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Ejecuta la extracción y retorna registros normalizados."""

    @abc.abstractmethod
    def persist(self, records: list[dict[str, Any]], **kwargs: Any) -> Path:
        """Guarda los registros extraídos en el almacenamiento bruto."""

    def run(self, **kwargs: Any) -> Path:
        records = self.extract(**kwargs)
        return self.persist(records, **kwargs)

