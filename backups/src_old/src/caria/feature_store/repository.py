"""Interfaz con Feast para publicar y consumir features."""

from __future__ import annotations

import logging
from pathlib import Path

from feast import FeatureStore

from caria.config.settings import Settings


LOGGER = logging.getLogger("caria.feature_store.repository")


class FeatureRepository:
    def __init__(self, settings: Settings) -> None:
        registry = Path(settings.get("feature_store", "registry_path", default="data/registry.db"))
        self.store = FeatureStore(repo_path=str(registry.parent))

    def materialize(self) -> None:
        LOGGER.info("Materializando feature views configuradas")
        self.store.materialize_incremental(end_date=None)

