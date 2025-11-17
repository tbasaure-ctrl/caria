"""Fuentes de ingesta soportadas."""

from caria.ingestion.sources.base import IngestionSource
from caria.ingestion.sources.fmp import FMPSource
from caria.ingestion.sources.fred import FREDSource
from caria.ingestion.sources.commodities import CommoditiesSource
from caria.ingestion.sources.fx import FXSource
from caria.ingestion.sources.indices import IndicesSource
from caria.ingestion.sources.newsapi import NewsAPISource
from caria.ingestion.sources.reddit import RedditSource
from caria.ingestion.sources.twitter import TwitterSource

__all__ = [
    "IngestionSource",
    "FMPSource",
    "FREDSource",
    "CommoditiesSource",
    "FXSource",
    "IndicesSource",
    "NewsAPISource",
    "RedditSource",
    "TwitterSource",
]

