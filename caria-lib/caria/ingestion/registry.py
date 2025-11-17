"""Registro de fuentes de ingesta."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from caria.ingestion.sources.base import IngestionSource
from caria.ingestion.sources.commodities import CommoditiesSource
from caria.ingestion.sources.fmp import FMPSource
from caria.ingestion.sources.fred import FREDSource
from caria.ingestion.sources.fx import FXSource
from caria.ingestion.sources.indices import IndicesSource
from caria.ingestion.sources.newsapi import NewsAPISource
from caria.ingestion.sources.reddit import RedditSource
from caria.ingestion.sources.twitter import TwitterSource


SOURCE_REGISTRY: dict[str, type[IngestionSource]] = {
    "fmp": FMPSource,
    "fred": FREDSource,
    "commodities": CommoditiesSource,
    "fx": FXSource,
    "indices": IndicesSource,
    "newsapi": NewsAPISource,
    "reddit": RedditSource,
    "twitter": TwitterSource,
}


def build_source(source_type: str, output_dir: Path, **kwargs: Any) -> IngestionSource:
    try:
        source_cls = SOURCE_REGISTRY[source_type]
    except KeyError as exc:  # noqa: BLE001
        raise ValueError(f"Fuente de ingesta no soportada: {source_type}") from exc
    return source_cls(output_dir=output_dir, **kwargs)

