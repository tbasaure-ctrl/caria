"""Cliente para Financial Modeling Prep (FMP)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from caria.ingestion.clients.fmp_client import FMPClient
from .base import IngestionSource


LOGGER = logging.getLogger("caria.ingestion.fmp")


class FMPSource(IngestionSource):
    """Descarga precios y estados financieros desde FMP."""

    base_url = "https://financialmodelingprep.com/api/v3"

    def __init__(self, output_dir: Path, api_key: str | None = None) -> None:
        super().__init__(output_dir)
        self.api_key = (api_key or os.getenv("FMP_API_KEY", "")).strip()
        if not self.api_key:
            raise RuntimeError("FMP_API_KEY no configurado en entorno")
        self.client = FMPClient(api_key=self.api_key)

    def _resolve_universe(
        self,
        tickers: list[str] | None,
        universe_strategy: str | None,
        selection_count: int | None,
    ) -> list[str]:
        if tickers:
            return tickers

        if not universe_strategy:
            raise ValueError("Debe especificarse 'tickers' o 'universe_strategy' para FMPSource")

        strategy = universe_strategy.lower()
        limit = selection_count or 50
        if strategy == "top_performers":
            return self.client.get_top_performers(limit=limit)
        if strategy in {"bankrupt", "delisted"}:
            return self.client.get_delisted_companies(limit=limit)
        if strategy in {"sp500", "sp500_constituents"}:
            constituents = self.client.get_sp500_constituents()
            return constituents[:limit] if selection_count else constituents

        raise ValueError(f"universe_strategy no soportado: {universe_strategy}")

    def extract(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        tickers: list[str] | None = None,
        universe_strategy: str | None = None,
        selection_count: int | None = 50,
        include_metrics: bool = True,
        **_: Any,
    ) -> list[dict[str, Any]]:
        symbols = self._resolve_universe(tickers, universe_strategy, selection_count)
        if not symbols:
            LOGGER.warning("No se encontraron tickers para FMP (%s)", universe_strategy)
            return []

        records: list[dict[str, Any]] = []
        for ticker in symbols:
            LOGGER.info("Descargando datos FMP para %s", ticker)
            try:
                prices = self.client.get_price_history(ticker, start_date=start_date, end_date=end_date)
            except requests.HTTPError as exc:  # noqa: BLE001
                LOGGER.error("Error descargando precios para %s: %s", ticker, exc)
                prices = []

            key_metrics: list[dict[str, Any]] = []
            financial_ratios: list[dict[str, Any]] = []
            financial_growth: list[dict[str, Any]] = []
            if include_metrics:
                try:
                    key_metrics = self.client.get_key_metrics(ticker)
                except requests.HTTPError as exc:  # noqa: BLE001
                    LOGGER.warning("Sin key metrics para %s: %s", ticker, exc)
                try:
                    financial_ratios = self.client.get_financial_ratios(ticker)
                except requests.HTTPError as exc:  # noqa: BLE001
                    LOGGER.warning("Sin financial ratios para %s: %s", ticker, exc)
                try:
                    financial_growth = self.client.get_financial_growth(ticker)
                except requests.HTTPError as exc:  # noqa: BLE001
                    LOGGER.warning("Sin financial growth para %s: %s", ticker, exc)

            records.append(
                {
                    "ticker": ticker,
                    "prices": prices,
                    "key_metrics": key_metrics,
                    "financial_ratios": financial_ratios,
                    "financial_growth": financial_growth,
                }
            )
        return records

    def persist(self, records: list[dict[str, Any]], partition: str | None = None, **_: Any) -> Path:
        partition = partition or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        output_path = self.output_dir / "fmp" / partition
        output_path.mkdir(parents=True, exist_ok=True)

        frames: list[pd.DataFrame] = []
        for record in records:
            ticker = record["ticker"]

            price_df = pd.DataFrame(record["prices"])
            if not price_df.empty:
                price_df["ticker"] = ticker
                price_df["dataset"] = "prices"
                frames.append(price_df)

            for dataset_name in ("key_metrics", "financial_ratios", "financial_growth"):
                data = record.get(dataset_name, []) or []
                df = pd.DataFrame(data)
                if df.empty:
                    continue
                df["ticker"] = ticker
                df["dataset"] = dataset_name
                frames.append(df)

        if not frames:
            LOGGER.warning("No se generaron datos FMP para persistir")
            return output_path / "fmp_data.parquet"

        combined = pd.concat(frames, ignore_index=True, sort=False)
        file_path = output_path / "fmp_data.parquet"
        combined.to_parquet(file_path, index=False)
        LOGGER.info("Guardado FMP en %s", file_path)
        return file_path

