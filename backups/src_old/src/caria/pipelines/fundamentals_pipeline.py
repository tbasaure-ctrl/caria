"""Pipeline Prefect para fundamentals y price action corporativo."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from prefect import flow, task

from caria.config.settings import Settings
from caria.ingestion.clients.fmp_client import FMPClient


LOGGER = logging.getLogger("caria.pipelines.fundamentals")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _prepare_statement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def _combine_frames(base: pd.DataFrame | None, manual: pd.DataFrame | None) -> pd.DataFrame | None:
    if manual is None or manual.empty:
        return base
    manual = manual.copy()
    manual["date"] = pd.to_datetime(manual["date"])
    if base is None or base.empty:
        return manual.reset_index(drop=True)
    base = base.copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.set_index(["ticker", "date"])
    manual = manual.set_index(["ticker", "date"])
    combined = base.combine_first(manual)
    return combined.reset_index()


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def _manual_metrics_from_statements(
    ticker: str,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash: pd.DataFrame,
    price_history: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    income = _prepare_statement(income)
    balance = _prepare_statement(balance)
    cash = _prepare_statement(cash)

    if income.empty or balance.empty:
        LOGGER.warning("Estados financieros insuficientes para %s", ticker)
        return None, None

    cols_income = {
        "date": "date",
        "period": "period",
        "revenue": "revenue",
        "grossProfit": "gross_profit",
        "operatingIncome": "operating_income",
        "ebit": "ebit",
        "incomeBeforeTax": "income_before_tax",
        "incomeTaxExpense": "income_tax",
        "netIncome": "net_income",
        "researchAndDevelopment": "research_and_development",
        "weightedAverageShsOutDil": "shares_diluted",
        "weightedAverageShsOut": "shares_basic",
    }
    cols_balance = {
        "date": "date",
        "totalAssets": "total_assets",
        "totalLiabilities": "total_liabilities",
        "totalDebt": "total_debt",
        "cashAndShortTermInvestments": "cash_equivalents",
        "totalStockholdersEquity": "total_equity",
        "commonStockSharesOutstanding": "shares_outstanding",
    }
    cols_cash = {
        "date": "date",
        "capitalExpenditure": "capital_expenditure",
        "freeCashFlow": "free_cash_flow",
        "operatingCashFlow": "operating_cash_flow",
    }

    income = income[[c for c in cols_income.keys() if c in income.columns]].rename(columns=cols_income)
    balance = balance[[c for c in cols_balance.keys() if c in balance.columns]].rename(columns=cols_balance)
    cash = cash[[c for c in cols_cash.keys() if c in cash.columns]].rename(columns=cols_cash)

    merged = income.merge(balance, on="date", how="left").merge(cash, on="date", how="left")
    merged["ticker"] = ticker

    numeric_cols = merged.columns.difference(["date", "period", "ticker"])
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.sort_values("date").reset_index(drop=True)

    def _safe_col(name: str, default: float = np.nan) -> pd.Series:
        if name in merged.columns:
            return pd.to_numeric(merged[name], errors="coerce")
        LOGGER.warning("Columna %s no disponible para %s", name, ticker)
        return pd.Series([default] * len(merged), index=merged.index, name=name)

    revenue_series = _safe_col("revenue")
    net_income_series = _safe_col("net_income")
    operating_income_series = _safe_col("operating_income")
    gross_profit_series = _safe_col("gross_profit", default=0)
    income_before_tax_series = _safe_col("income_before_tax")
    income_tax_series = _safe_col("income_tax")
    total_assets_series = _safe_col("total_assets")
    total_equity_series = _safe_col("total_equity")
    total_debt_series = _safe_col("total_debt", default=0)
    cash_equivalents_series = _safe_col("cash_equivalents", default=0)
    capital_expenditure_series = _safe_col("capital_expenditure")
    research_and_development_series = _safe_col("research_and_development")
    free_cash_flow_series = _safe_col("free_cash_flow")

    merged["revenue"] = revenue_series
    merged["net_income"] = net_income_series
    merged["operating_income"] = operating_income_series
    merged["gross_profit"] = gross_profit_series
    merged["income_before_tax"] = income_before_tax_series
    merged["income_tax"] = income_tax_series
    merged["total_assets"] = total_assets_series
    merged["total_equity"] = total_equity_series
    merged["total_debt"] = total_debt_series
    merged["cash_equivalents"] = cash_equivalents_series
    merged["capital_expenditure"] = capital_expenditure_series
    merged["research_and_development"] = research_and_development_series
    merged["free_cash_flow"] = free_cash_flow_series

    merged["revenue_growth"] = revenue_series.pct_change()
    merged["net_income_growth"] = net_income_series.pct_change()

    ebit_series = merged.get("ebit")
    if ebit_series is None:
        LOGGER.warning("EBIT no disponible para %s; usando operating_income/net_income", ticker)
        ebit_series = pd.Series([np.nan] * len(merged), index=merged.index)
    else:
        ebit_series = pd.to_numeric(ebit_series, errors="coerce")
    base_ebit = ebit_series.fillna(operating_income_series).fillna(net_income_series)
    tax_rate = np.where(
        income_before_tax_series.abs() > 1e-6,
        (income_tax_series.abs() / income_before_tax_series.abs()).clip(0, 1),
        0.21,
    )
    merged["nopat"] = base_ebit * (1 - tax_rate)

    merged["invested_capital"] = (
        total_debt_series.fillna(0)
        + total_equity_series.fillna(0)
        - cash_equivalents_series.fillna(0)
    )
    merged["roic_manual"] = np.where(
        merged["invested_capital"].abs() > 1e-6,
        merged["nopat"] / merged["invested_capital"],
        np.nan,
    )
    merged["roiic_manual"] = merged["nopat"].diff() / merged["invested_capital"].diff()

    merged["gross_profit_margin"] = np.where(
        merged["revenue"].abs() > 1e-6,
        merged["gross_profit"] / merged["revenue"],
        np.nan,
    )
    merged["net_profit_margin"] = np.where(
        merged["revenue"].abs() > 1e-6,
        merged["net_income"] / merged["revenue"],
        np.nan,
    )
    merged["return_on_assets"] = np.where(
        total_assets_series.abs() > 1e-6,
        net_income_series / total_assets_series,
        np.nan,
    )
    merged["return_on_equity"] = np.where(
        total_equity_series.abs() > 1e-6,
        net_income_series / total_equity_series,
        np.nan,
    )

    merged["net_debt"] = total_debt_series.fillna(0) - cash_equivalents_series.fillna(0)
    merged["capital_expenditure"] = capital_expenditure_series
    merged["research_and_development"] = research_and_development_series
    operating_cash_flow_series = _safe_col("operating_cash_flow")
    merged["free_cash_flow"] = free_cash_flow_series.fillna(
        operating_cash_flow_series + capital_expenditure_series
    )

    shares_diluted = _safe_col("shares_diluted")
    shares_outstanding = _safe_col("shares_outstanding")
    shares_basic = _safe_col("shares_basic")

    shares = shares_diluted.fillna(shares_outstanding).fillna(shares_basic)
    if shares.isna().all():
        LOGGER.warning("Sin datos de acciones para %s; usando 0", ticker)
        shares = pd.Series([0.0] * len(merged), index=merged.index)

    if price_history is not None and not price_history.empty:
        price_series = price_history.sort_values("date")["close"].astype(float)
        price_df = price_series.reset_index(drop=True)
        price_df = price_history[["date", "close"]].sort_values("date")
        merged = pd.merge_asof(
            merged.sort_values("date"),
            price_df,
            on="date",
            direction="backward",
        )
        merged.rename(columns={"close": "close_price"}, inplace=True)
    else:
        merged["close_price"] = np.nan

    merged["market_cap"] = merged["close_price"] * shares
    merged["enterprise_value_manual"] = merged["market_cap"] + merged["total_debt"].fillna(0) - \
        cash_equivalents_series.fillna(0)
    merged["free_cash_flow_yield_manual"] = np.where(
        merged["market_cap"].abs() > 1e-6,
        merged["free_cash_flow"] / merged["market_cap"],
        np.nan,
    )
    merged["price_to_book_manual"] = np.where(
        merged["total_equity"].abs() > 1e-6,
        merged["market_cap"] / merged["total_equity"],
        np.nan,
    )
    merged["price_to_sales_manual"] = np.where(
        merged["revenue"].abs() > 1e-6,
        merged["market_cap"] / merged["revenue"],
        np.nan,
    )

    quality = merged[
        [
            "date",
            "period",
            "ticker",
            "roic_manual",
            "roiic_manual",
            "gross_profit_margin",
            "net_profit_margin",
            "return_on_assets",
            "return_on_equity",
            "revenue_growth",
            "net_income_growth",
            "free_cash_flow_yield_manual",
            "net_debt",
            "capital_expenditure",
            "research_and_development",
        ]
    ].rename(
        columns={
            "roic_manual": "roic",
            "roiic_manual": "roiic",
            "gross_profit_margin": "grossProfitMargin",
            "net_profit_margin": "netProfitMargin",
            "return_on_assets": "returnOnAssets",
            "return_on_equity": "returnOnEquity",
            "revenue_growth": "revenueGrowth",
            "net_income_growth": "netIncomeGrowth",
            "free_cash_flow_yield_manual": "freeCashFlowYield",
            "capital_expenditure": "capitalExpenditures",
            "research_and_development": "r_and_d",
        }
    )

    value = merged[
        [
            "date",
            "period",
            "ticker",
            "free_cash_flow_yield_manual",
            "price_to_book_manual",
            "price_to_sales_manual",
            "enterprise_value_manual",
            "net_debt",
        ]
    ].rename(
        columns={
            "free_cash_flow_yield_manual": "freeCashFlowYield",
            "price_to_book_manual": "priceToBookRatio",
            "price_to_sales_manual": "priceToSalesRatio",
            "enterprise_value_manual": "enterpriseValue",
        }
    )

    return quality, value


@task
def load_fundamentals_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    return _load_yaml(config_path)


@task
def resolve_universe(config: dict[str, Any], client: FMPClient) -> dict[str, list[str]]:
    universe_cfg = config.get("universe", {})
    result: dict[str, list[str]] = {}

    top_cfg = universe_cfg.get("top_performers", {})
    if "tickers" in top_cfg:
        result["top_performers"] = top_cfg["tickers"]
    else:
        limit = top_cfg.get("limit", 50)
        result["top_performers"] = client.get_top_performers(limit)

    distressed_cfg = universe_cfg.get("distressed", {})
    if "tickers" in distressed_cfg:
        result["distressed"] = distressed_cfg["tickers"]
    else:
        limit = distressed_cfg.get("limit", 50)
        result["distressed"] = client.get_delisted_companies(limit)

    sp500 = universe_cfg.get("sp500", {}).get("include", False)
    if sp500:
        result["sp500"] = client.get_sp500_constituents()

    return result


@task
def combine_universe(universe_dict: dict[str, list[str]], limit: int | None = None) -> list[str]:
    merged: list[str] = []
    for tickers in universe_dict.values():
        merged.extend(tickers)
    unique = sorted(set(filter(None, merged)))
    if limit is not None:
        return unique[:limit]
    return unique


@task
def fetch_fundamental_frames(ticker: str, start_date: str | None, client: FMPClient) -> dict[str, pd.DataFrame]:
    LOGGER.info("Descargando fundamentals para %s", ticker)
    income_stmt = pd.DataFrame(client.get_income_statement(ticker))
    balance_stmt = pd.DataFrame(client.get_balance_sheet(ticker))
    cash_stmt = pd.DataFrame(client.get_cash_flow(ticker))
    key_metrics = pd.DataFrame(client.get_key_metrics(ticker))
    growth = pd.DataFrame(client.get_financial_growth(ticker))
    ratios = pd.DataFrame(client.get_financial_ratios(ticker))

    if key_metrics.empty:
        LOGGER.warning("Sin key metrics para %s", ticker)
    if growth.empty:
        LOGGER.warning("Sin financial growth para %s", ticker)
    if ratios.empty:
        LOGGER.warning("Sin financial ratios para %s", ticker)
    if income_stmt.empty:
        LOGGER.warning("Sin income statement para %s", ticker)
    if balance_stmt.empty:
        LOGGER.warning("Sin balance sheet para %s", ticker)
    if cash_stmt.empty:
        LOGGER.warning("Sin cash flow statement para %s", ticker)

    frames: dict[str, pd.DataFrame] = {}

    if not key_metrics.empty:
        cols_quality = [
            "date",
            "period",
            "roic",
            "roiic",
            "grossProfitMargin",
            "netProfitMargin",
            "returnOnAssets",
            "returnOnEquity",
            "freeCashFlowYield",
        ]
        quality = key_metrics[[c for c in cols_quality if c in key_metrics.columns]].copy()
        if not growth.empty:
            growth_cols = ["date", "period", "revenueGrowth", "netIncomeGrowth"]
            quality = quality.merge(
                growth[[c for c in growth_cols if c in growth.columns]],
                on=[col for col in ["date", "period"] if col in quality.columns and col in growth.columns],
                how="left",
            )
        quality["ticker"] = ticker
        frames["quality"] = quality

        cols_value = [
            "date",
            "period",
            "priceToBookRatio",
            "priceToSalesRatio",
            "evToEbitda",
            "enterpriseValue",
            "freeCashFlowPerShare",
            "freeCashFlowYield",
        ]
        value = key_metrics[[c for c in cols_value if c in key_metrics.columns]].copy()
        value["ticker"] = ticker
        frames["value"] = value

    price_history = pd.DataFrame(client.get_price_history(ticker, start_date=start_date))
    if price_history.empty:
        LOGGER.warning("Sin price history para %s", ticker)
        return frames

    price_history["date"] = pd.to_datetime(price_history["date"])
    price_history = price_history.sort_values("date").reset_index(drop=True)
    price_history["ticker"] = ticker

    close = price_history["close"].astype(float)
    price_history["sma_20"] = close.rolling(window=20).mean()
    price_history["sma_50"] = close.rolling(window=50).mean()
    price_history["sma_200"] = close.rolling(window=200).mean()
    price_history["ema_20"] = close.ewm(span=20, adjust=False).mean()
    price_history["ema_50"] = close.ewm(span=50, adjust=False).mean()
    price_history["ema_200"] = close.ewm(span=200, adjust=False).mean()
    price_history["rsi_14"] = _compute_rsi(close, 14)
    macd, signal = _compute_macd(close)
    price_history["macd"] = macd
    price_history["macd_signal"] = signal

    momentum_cols = [
        "date",
        "ticker",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "macd",
        "macd_signal",
    ]
    frames["momentum"] = price_history[momentum_cols].copy()

    price_history["returns_20d"] = close.pct_change(20)
    price_history["returns_60d"] = close.pct_change(60)
    price_history["returns_120d"] = close.pct_change(120)
    price_history["volatility_30d"] = close.pct_change().rolling(30).std() * np.sqrt(252)
    price_history["rolling_max"] = close.rolling(window=252, min_periods=1).max()
    price_history["drawdown"] = (close / price_history["rolling_max"]) - 1
    if {"high", "low"}.issubset(price_history.columns):
        price_history["atr_14"] = _compute_atr(price_history, 14)
    else:
        price_history["atr_14"] = np.nan

    risk_cols = [
        "date",
        "ticker",
        "close",
        "returns_20d",
        "returns_60d",
        "returns_120d",
        "volatility_30d",
        "drawdown",
        "atr_14",
    ]
    frames["risk"] = price_history[risk_cols].copy()

    manual_quality, manual_value = _manual_metrics_from_statements(
        ticker,
        income_stmt,
        balance_stmt,
        cash_stmt,
        price_history[["date", "close"]],
    )

    frames["quality"] = _combine_frames(frames.get("quality"), manual_quality)
    frames["value"] = _combine_frames(frames.get("value"), manual_value)

    return frames


@task
def persist_frames(frames: Iterable[pd.DataFrame], path: Path) -> Path:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        LOGGER.warning("No hay datos para persistir en %s", path)
        return path
    combined = pd.concat(frames, ignore_index=True)
    _ensure_dir(path.parent)
    combined.to_parquet(path, index=False)
    LOGGER.info("Guardado %s (%d filas)", path, len(combined))
    return path


@flow(name="caria-fundamentals-pipeline")
def fundamentals_flow(settings: Settings, config_path: str) -> None:
    config = load_fundamentals_config(config_path)
    client = FMPClient()

    universe_dict = resolve_universe(config, client)
    ticker_limit = config.get("max_tickers")
    tickers = combine_universe(universe_dict, ticker_limit)
    LOGGER.info("Universo final (%d tickers)", len(tickers))

    start_date = config.get("start_date")

    bronze_path = Path(settings.get("storage", "bronze_path", default="data/bronze"))
    silver_path = Path(settings.get("storage", "silver_path", default="data/silver"))

    quality_frames: list[pd.DataFrame] = []
    value_frames: list[pd.DataFrame] = []
    momentum_frames: list[pd.DataFrame] = []
    risk_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        frames = fetch_fundamental_frames(ticker, start_date, client)
        if "quality" in frames:
            quality_frames.append(frames["quality"])
        if "value" in frames:
            value_frames.append(frames["value"])
        if "momentum" in frames:
            momentum_frames.append(frames["momentum"])
        if "risk" in frames:
            risk_frames.append(frames["risk"])

    bronze_manifest = {
        "universe": universe_dict,
        "tickers": tickers,
        "config": config,
    }
    bronze_meta_path = bronze_path / "fundamentals" / "manifest.json"
    _ensure_dir(bronze_meta_path.parent)
    bronze_meta_path.write_text(json.dumps(bronze_manifest, indent=2), encoding="utf-8")

    persist_frames.submit(
        quality_frames,
        silver_path / "fundamentals" / "quality_signals.parquet",
    )
    persist_frames.submit(
        value_frames,
        silver_path / "fundamentals" / "value_signals.parquet",
    )
    persist_frames.submit(
        momentum_frames,
        silver_path / "technicals" / "momentum_signals.parquet",
    )
    persist_frames.submit(
        risk_frames,
        silver_path / "technicals" / "risk_signals.parquet",
    )


def run(settings: Settings, pipeline_config_path: str | None = None) -> None:
    if not pipeline_config_path:
        raise ValueError("pipeline_config_path es obligatorio para fundamentals_flow")
    fundamentals_flow(settings=settings, config_path=pipeline_config_path)

