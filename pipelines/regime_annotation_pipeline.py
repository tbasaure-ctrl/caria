"""Pipeline para anotar regímenes macro y agregar noticias históricas."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from prefect import flow, task

from caria.config.settings import Settings


LOGGER = logging.getLogger("caria.pipelines.regime_annotation")


FRED_SERIES_RECESSION = "USREC"
FRED_SERIES_DEPRESSION = "USRECD"


MANUAL_EVENTS = [
    {
        "name": "Pánico de 1907",
        "type": "crisis",
        "start_date": "1907-10-01",
        "end_date": "1908-06-30",
        "region": "US",
        "severity": 3,
        "description": "Crisis bancaria severe, intervención de JP Morgan.",
        "sources": ["https://www.federalreservehistory.org/essays/panic-of-1907"],
    },
    {
        "name": "Gran Depresión",
        "type": "depression",
        "start_date": "1929-08-01",
        "end_date": "1933-03-01",
        "region": "US",
        "severity": 5,
        "description": "Quiebra masiva post crash 1929.",
        "sources": ["https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions"],
    },
    {
        "name": "Burbuja Dotcom",
        "type": "mania",
        "start_date": "1997-01-01",
        "end_date": "2001-03-31",
        "region": "US",
        "severity": 4,
        "description": "Manía tecnológica previa al crash de Nasdaq.",
        "sources": ["https://www.imf.org/external/pubs/ft/wp/2001/wp0179.pdf"],
    },
    {
        "name": "Crisis Subprime",
        "type": "crisis",
        "start_date": "2007-08-01",
        "end_date": "2009-06-30",
        "region": "US",
        "severity": 4,
        "description": "Colapso del mercado hipotecario y sistema financiero.",
        "sources": ["https://www.federalreservehistory.org/essays/great-recession"],
    },
    {
        "name": "COVID Shock",
        "type": "crisis",
        "start_date": "2020-02-15",
        "end_date": "2020-06-30",
        "region": "Global",
        "severity": 3,
        "description": "Choque sanitario y económico global.",
        "sources": ["https://www.bis.org/publ/bisbull28.htm"],
    },
]


POSITIVE_TERMS = {"boom", "optimism", "growth", "record", "surge", "rally"}
NEGATIVE_TERMS = {"crash", "panic", "recession", "depression", "fear", "default", "bankruptcy", "bubble", "mania"}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fred_series(series_id: str, api_key: str, start_date: str | None = None) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start_date:
        params["observation_start"] = start_date
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    observations = resp.json().get("observations", [])
    df = pd.DataFrame(observations)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]]


@task
def build_recession_events(api_key: str, start_date: str | None) -> pd.DataFrame:
    rec = _fred_series(FRED_SERIES_RECESSION, api_key, start_date)
    if rec.empty:
        LOGGER.warning("Sin datos de recesión NBER")
        return pd.DataFrame()
    rec["flag"] = rec["value"].fillna(0.0) > 0
    rec["group"] = (rec["flag"] != rec["flag"].shift()).cumsum()
    events = []
    for _, group in rec.groupby("group"):
        if not group["flag"].iloc[0]:
            continue
        start = group["date"].min()
        end = group["date"].max()
        events.append(
            {
                "name": f"Recesión NBER {start.year}",
                "type": "recession",
                "start_date": start,
                "end_date": end,
                "region": "US",
                "severity": 3,
                "description": "Período marcado como recesión por NBER.",
                "sources": ["https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions"],
            }
        )
    return pd.DataFrame(events)


@task
def build_manual_events() -> pd.DataFrame:
    frame = pd.DataFrame(MANUAL_EVENTS)
    frame["start_date"] = pd.to_datetime(frame["start_date"])
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    frame["sources"] = frame["sources"].apply(lambda lst: json.dumps(lst) if isinstance(lst, list) else lst)
    return frame


@task
def aggregate_news_sentiment(raw_root: Path) -> pd.DataFrame:
    news_dir = raw_root / "newsapi"
    if not news_dir.exists():
        LOGGER.warning("No se encontraron datos raw de NewsAPI para agregación de sentimiento")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for parquet in news_dir.rglob("*.parquet"):
        df = pd.read_parquet(parquet)
        if df.empty:
            continue
        if "publishedAt" in df.columns:
            df["date"] = pd.to_datetime(df["publishedAt"]).dt.date
        else:
            df["date"] = pd.Timestamp(parquet.parent.name).date()
        frames.append(df)

    if not frames:
        LOGGER.warning("NewsAPI sin registros para sentimiento")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["headline"] = combined.get("title", combined.get("description", "")).fillna("")

    def score_text(text: str) -> float:
        if not text:
            return 0.0
        text_lower = text.lower()
        pos_hits = sum(term in text_lower for term in POSITIVE_TERMS)
        neg_hits = sum(term in text_lower for term in NEGATIVE_TERMS)
        total = pos_hits + neg_hits
        if total == 0:
            return 0.0
        return (pos_hits - neg_hits) / total

    combined["sentiment_score"] = combined["headline"].apply(score_text)
    combined["event_tags"] = combined["headline"].apply(
        lambda txt: [term for term in NEGATIVE_TERMS.union(POSITIVE_TERMS) if term in txt.lower()]
    )

    aggregated = (
        combined.groupby([pd.to_datetime(combined["date"]), combined.get("ticker", pd.Series([None] * len(combined)))])
        ["sentiment_score"]
        .mean()
        .reset_index()
    )
    aggregated.rename(columns={"date": "date", "sentiment_score": "sentiment_score", 0: "ticker"}, inplace=True)
    aggregated["source"] = "newsapi"
    aggregated["event_tags"] = aggregated["ticker"].apply(lambda _: json.dumps([]))
    aggregated["date"] = pd.to_datetime(aggregated["date"])
    return aggregated[["date", "ticker", "sentiment_score", "source", "event_tags"]]


@task
def persist_events(frame: pd.DataFrame, path: Path) -> Path:
    if frame.empty:
        LOGGER.warning("No se generaron eventos de régimen")
        return path
    frame = frame.copy()
    frame["start_date"] = pd.to_datetime(frame["start_date"])
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    _ensure_dir(path.parent)
    frame.to_parquet(path, index=False)
    LOGGER.info("Guardado eventos en %s", path)
    return path


@task
def persist_news_sentiment(frame: pd.DataFrame, path: Path) -> Path:
    if frame.empty:
        LOGGER.warning("No se generaron agregados de noticias")
        return path
    _ensure_dir(path.parent)
    frame.to_parquet(path, index=False)
    LOGGER.info("Guardado sentimiento noticias en %s", path)
    return path


@flow(name="caria-regime-annotation")
def regime_annotation_flow(settings: Settings, config_path: str) -> None:
    config = _load_yaml(Path(config_path))
    api_key = config.get("fred_api_key") or settings.get("database", "fred_api_key", default=None)
    if not api_key:
        api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Se requiere FRED_API_KEY para anotar regímenes")

    start_date = config.get("start_date")

    recession_df = build_recession_events(api_key, start_date)
    manual_df = build_manual_events()
    events_df = pd.concat([recession_df, manual_df], ignore_index=True, sort=False)

    raw_path = Path(settings.get("storage", "raw_path", default="data/raw"))
    news_sentiment_df = aggregate_news_sentiment(raw_path)

    silver_path = Path(settings.get("storage", "silver_path", default="data/silver"))
    persist_events.submit(events_df, silver_path / "events" / "regime_context.parquet")
    persist_news_sentiment.submit(news_sentiment_df, silver_path / "news_sentiment" / "daily_scores.parquet")


def run(settings: Settings, pipeline_config_path: str | None = None) -> None:
    if not pipeline_config_path:
        raise ValueError("pipeline_config_path es obligatorio para regime_annotation_flow")
    regime_annotation_flow(settings=settings, config_path=pipeline_config_path)

