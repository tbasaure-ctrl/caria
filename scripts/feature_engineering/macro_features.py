"""Feature engineering para datos macro y commodities: crear features cíclicas y relativas."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR / "src"))


def calculate_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features macro cíclicas desde datos FRED."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Yield curve slope (10Y - 2Y) - indicador de recesión
    if "DGS10" in df.columns and "DGS2" in df.columns:
        df["yield_curve_slope"] = df["DGS10"] - df["DGS2"]
        df["yield_curve_inverted"] = (df["yield_curve_slope"] < 0).astype(int)

    # Credit spreads (si están disponibles)
    # BAA - AAA spread indica riesgo crediticio
    if "BAA" in df.columns and "AAA" in df.columns:
        df["credit_spread"] = df["BAA"] - df["AAA"]
    elif "DGS10" in df.columns:
        # Usar spread aproximado basado en treasury
        df["credit_spread"] = df["DGS10"] * 0.02  # Aproximación

    # Economic regime indicators
    # Recession probability basado en múltiples indicadores
    df["recession_probability"] = 0.0

    # PMI < 50 indica contracción
    if "MANPMI" in df.columns:
        df["pmi_below_50"] = (df["MANPMI"] < 50).astype(int)
        df["recession_probability"] += df["pmi_below_50"] * 0.3

    # Unemployment rising
    if "UNRATE" in df.columns:
        df["unemployment_change"] = df["UNRATE"].diff()
        df["unemployment_rising"] = (df["unemployment_change"] > 0.5).astype(int)
        df["recession_probability"] += df["unemployment_rising"] * 0.3

    # Yield curve inverted
    if "yield_curve_inverted" in df.columns:
        df["recession_probability"] += df["yield_curve_inverted"] * 0.4

    df["recession_probability"] = np.clip(df["recession_probability"], 0, 1)

    # Macro regime classification
    df["macro_regime"] = "expansion"
    df.loc[df["recession_probability"] > 0.5, "macro_regime"] = "recession"
    df.loc[
        (df["recession_probability"] > 0.3) & (df["recession_probability"] <= 0.5),
        "macro_regime",
    ] = "slowdown"

    # Commodity momentum
    commodity_cols = [
        "GOLDAMGBD228NLBM",  # Gold
        "PSLVAMUSD",  # Silver
        "DCOILWTICO",  # Oil
        "PCOPPUSDM",  # Copper
        "PNICKUSDM",  # Nickel
        "PALUMUSDM",  # Aluminum
    ]

    for col in commodity_cols:
        if col in df.columns:
            # Momentum de 3 meses
            df[f"{col}_momentum_3m"] = df[col].pct_change(periods=63)
            # Momentum de 12 meses
            df[f"{col}_momentum_12m"] = df[col].pct_change(periods=252)

    # Commodity ratios (indicadores de riesgo/crecimiento)
    if "GOLDAMGBD228NLBM" in df.columns and "DCOILWTICO" in df.columns:
        df["gold_oil_ratio"] = df["GOLDAMGBD228NLBM"] / (df["DCOILWTICO"] + 1e-6)
        # Ratio alto indica aversión al riesgo
        df["risk_aversion_indicator"] = (
            df["gold_oil_ratio"] > df["gold_oil_ratio"].rolling(252).mean()
        ).astype(int)

    if "PCOPPUSDM" in df.columns and "GOLDAMGBD228NLBM" in df.columns:
        df["copper_gold_ratio"] = df["PCOPPUSDM"] / (df["GOLDAMGBD228NLBM"] + 1e-6)
        # Ratio alto indica crecimiento económico
        df["growth_indicator"] = (
            df["copper_gold_ratio"] > df["copper_gold_ratio"].rolling(252).mean()
        ).astype(int)

    # Currency features (si están disponibles)
    currency_cols = [
        "DEXUSEU",  # Euro vs USD
        "DEXCHUS",  # Yuan vs USD
        "DEXJPUS",  # Yen vs USD
        "DEXUSUK",  # GBP vs USD
        "DEXCAUS",  # CAD vs USD
        "DEXMXUS",  # MXN vs USD
        "DTWEXBGS",  # Dollar Index Broad
        "DTWEXEMEGS",  # Dollar Index Emerging Markets
    ]

    for col in currency_cols:
        if col in df.columns:
            # Momentum de moneda
            df[f"{col}_momentum_3m"] = df[col].pct_change(periods=63)
            # Strength vs histórico
            df[f"{col}_strength"] = (
                df[col] / df[col].rolling(252).mean() - 1
            )  # % sobre/bajo promedio

    # Z-scores para normalización histórica
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["date"]:
            rolling_mean = df[col].rolling(window=252, min_periods=1).mean()
            rolling_std = df[col].rolling(window=252, min_periods=1).std()
            rolling_std = rolling_std.replace(0, np.nan)
            df[f"{col}_zscore"] = (df[col] - rolling_mean) / rolling_std

    return df


def calculate_relative_features(
    stock_df: pd.DataFrame, macro_df: pd.DataFrame
) -> pd.DataFrame:
    """Calcula features relativas históricas para stocks (percentiles, z-scores)."""
    stock_df = stock_df.copy()

    # Percentiles históricos de múltiplos (5 años = ~1260 trading days)
    valuation_cols = [
        "priceToBookRatio",
        "priceToSalesRatio",
        "priceToEarningsRatio",
        "enterpriseValue",
        "freeCashFlowYield",
    ]

    for col in valuation_cols:
        if col in stock_df.columns:
            # Percentil histórico por ticker
            stock_df[f"{col}_percentile_5y"] = stock_df.groupby("ticker")[
                col
            ].transform(
                lambda x: x.rolling(window=1260, min_periods=252).apply(
                    lambda y: (y.iloc[-1] > y.iloc[:-1]).sum() / len(y.iloc[:-1])
                    if len(y.iloc[:-1]) > 0
                    else 0.5
                )
            )

            # Z-score histórico
            rolling_mean = stock_df.groupby("ticker")[col].transform(
                lambda x: x.rolling(window=1260, min_periods=252).mean()
            )
            rolling_std = stock_df.groupby("ticker")[col].transform(
                lambda x: x.rolling(window=1260, min_periods=252).std()
            )
            rolling_std = rolling_std.replace(0, np.nan)
            stock_df[f"{col}_zscore_5y"] = (stock_df[col] - rolling_mean) / rolling_std

    # ROIC/ROE históricos
    quality_cols = ["roic", "returnOnEquity", "returnOnAssets"]

    for col in quality_cols:
        if col in stock_df.columns:
            # Percentil histórico
            stock_df[f"{col}_percentile_5y"] = stock_df.groupby("ticker")[
                col
            ].transform(
                lambda x: x.rolling(window=1260, min_periods=252).apply(
                    lambda y: (y.iloc[-1] > y.iloc[:-1]).sum() / len(y.iloc[:-1])
                    if len(y.iloc[:-1]) > 0
                    else 0.5
                )
            )

            # Promedio histórico (3 años)
            stock_df[f"{col}_3y_avg"] = stock_df.groupby("ticker")[col].transform(
                lambda x: x.rolling(window=756, min_periods=252).mean()
            )

            # Lag quarters (para evitar leakage)
            stock_df[f"{col}_lag_1q"] = stock_df.groupby("ticker")[col].shift(63)
            stock_df[f"{col}_lag_2q"] = stock_df.groupby("ticker")[col].shift(126)

    # Merge con macro para contexto económico
    if not macro_df.empty and "date" in macro_df.columns:
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        macro_df["date"] = pd.to_datetime(macro_df["date"])

        # Merge usando merge_asof para forward-fill macro data
        stock_df = stock_df.sort_values(["ticker", "date"])
        macro_df = macro_df.sort_values("date")

        # Seleccionar solo columnas macro relevantes
        macro_cols_to_merge = [
            "yield_curve_slope",
            "credit_spread",
            "recession_probability",
            "macro_regime",
            "gold_oil_ratio",
            "copper_gold_ratio",
            "risk_aversion_indicator",
            "growth_indicator",
        ]
        macro_cols_to_merge = [
            col for col in macro_cols_to_merge if col in macro_df.columns
        ]

        if macro_cols_to_merge:
            macro_subset = macro_df[["date"] + macro_cols_to_merge]
            stock_df = pd.merge_asof(
                stock_df,
                macro_subset,
                on="date",
                direction="backward",
                suffixes=("", "_macro"),
            )

    return stock_df


def main() -> None:
    """Ejemplo de uso."""
    print("Feature engineering macro - funciones disponibles:")
    print("  - calculate_macro_features(df): Calcula features cíclicas desde FRED")
    print("  - calculate_relative_features(stock_df, macro_df): Features relativas históricas")


if __name__ == "__main__":
    main()

