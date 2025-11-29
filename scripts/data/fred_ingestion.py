"""Ingesta de datos macro y commodities desde FRED API."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    from fredapi import Fred
    HAS_FREDAPI = True
except ImportError:
    HAS_FREDAPI = False
    print("[WARNING] fredapi no está instalado. Instala con: pip install fredapi")

BASE_DIR = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingesta de datos macro/commodities desde FRED")
    parser.add_argument(
        "--start-date",
        default="1900-01-01",
        help="Fecha de inicio (YYYY-MM-DD). Default: 1900-01-01",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "data" / "silver" / "macro"),
        help="Directorio de salida",
    )
    parser.add_argument(
        "--api-key",
        help="FRED API key (o usar FRED_API_KEY en .env)",
    )
    return parser.parse_args()


# Series FRED a descargar
FRED_SERIES = {
    # Macro
    "GDPC1": "GDP Real (Billions of Chained 2017 Dollars)",
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items",
    "UNRATE": "Unemployment Rate",
    "FEDFUNDS": "Effective Federal Funds Rate",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "VIXCLS": "CBOE Volatility Index: VIX",
    "MANPMI": "ISM Manufacturing: PMI Composite Index",
    "UMCSENT": "University of Michigan: Consumer Sentiment",
    # Commodities - Metales
    "GOLDAMGBD228NLBM": "Gold Fixing Price (London)",
    "PSLVAMUSD": "Silver Price (USD per Troy Ounce)",
    "PCOPPUSDM": "Global Price of Copper",
    "PNICKUSDM": "Global Price of Nickel",
    "PALUMUSDM": "Global Price of Aluminum",
    # Commodities - Energía
    "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate (WTI)",
    "PNRGINDEXM": "Natural Gas Price Index",
    "DHOILNYH": "No. 2 Heating Oil New York Harbor",
    # Commodities - Agrícolas
    "PWHEAMTUSDM": "Global Price of Wheat",
    "PSOYABUSDM": "Global Price of Soybeans",
    "PCOFFOTMUSDM": "Global Price of Coffee",
    "PSUGAISAUSDM": "Global Price of Sugar",
    # Commodities - Índices
    "PALLFNFINDEXM": "Global Price of All Commodities",
    "PALLFNFINDEXQ": "Global Price of All Commodities (Quarterly)",
    # Currencies - Dólar vs otras monedas
    "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
    "DEXCHUS": "China / U.S. Foreign Exchange Rate",
    "DEXJPUS": "Japan / U.S. Foreign Exchange Rate",
    "DEXUSUK": "U.S. / U.K. Foreign Exchange Rate",
    "DEXCAUS": "Canada / U.S. Foreign Exchange Rate",
    "DEXMXUS": "Mexico / U.S. Foreign Exchange Rate",
    "DTWEXBGS": "Trade Weighted U.S. Dollar Index: Broad",
    "DTWEXEMEGS": "Trade Weighted U.S. Dollar Index: Emerging Markets",
    # Credit spreads (si están disponibles)
    "BAA10Y": "Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury",
    "AAA": "Moody's Seasoned Aaa Corporate Bond Yield",
    "BAA": "Moody's Seasoned Baa Corporate Bond Yield",
}


def download_fred_series(
    fred: Fred, series_id: str, start_date: str, description: str
) -> pd.DataFrame:
    """Descarga una serie de FRED."""
    try:
        data = fred.get_series(series_id, observation_start=start_date)
        if data.empty:
            print(f"  [WARNING] {series_id} ({description}): Sin datos")
            return pd.DataFrame()

        df = pd.DataFrame({"date": data.index, series_id: data.values})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        print(f"  [OK] {series_id}: {len(df)} observaciones desde {df['date'].min()} hasta {df['date'].max()}")
        return df
    except Exception as e:
        print(f"  [ERROR] {series_id}: {e}")
        return pd.DataFrame()


def calculate_derived_features(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Calcula features derivados de una serie macro/commodity."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # YoY changes (year-over-year)
    df[f"{series_id}_yoy"] = df[series_id].pct_change(periods=252)  # ~1 año (252 trading days)

    # 3-month changes
    df[f"{series_id}_3m"] = df[series_id].pct_change(periods=63)  # ~3 meses

    # Rolling statistics (252 días = ~1 año)
    window = 252
    df[f"{series_id}_ma_20"] = df[series_id].rolling(window=20, min_periods=1).mean()
    df[f"{series_id}_ma_50"] = df[series_id].rolling(window=50, min_periods=1).mean()
    df[f"{series_id}_ma_200"] = df[series_id].rolling(window=200, min_periods=1).mean()

    # Z-scores (normalización histórica)
    rolling_mean = df[series_id].rolling(window=window, min_periods=1).mean()
    rolling_std = df[series_id].rolling(window=window, min_periods=1).std()
    rolling_std = rolling_std.replace(0, np.nan)  # Evitar división por cero
    df[f"{series_id}_zscore"] = (df[series_id] - rolling_mean) / rolling_std

    # Volatilidad (rolling std)
    df[f"{series_id}_volatility"] = df[series_id].rolling(window=window, min_periods=1).std()

    return df


def resample_to_daily(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Resamplea series mensuales/trimestrales a frecuencia diaria usando forward-fill."""
    df = df.copy()
    df = df.set_index("date")

    # Crear rango diario completo
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df_daily = pd.DataFrame(index=date_range)

    # Reindexar y forward-fill
    df_daily = df_daily.join(df, how="left")
    df_daily = df_daily.fillna(method="ffill")  # Forward fill para series macro

    df_daily = df_daily.reset_index()
    df_daily = df_daily.rename(columns={"index": "date"})

    return df_daily


def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ratios entre commodities (gold/oil, copper/gold, etc.)."""
    df = df.copy()

    # Gold/Oil ratio
    if "GOLDAMGBD228NLBM" in df.columns and "DCOILWTICO" in df.columns:
        df["gold_oil_ratio"] = df["GOLDAMGBD228NLBM"] / (df["DCOILWTICO"] + 1e-6)

    # Copper/Gold ratio
    if "PCOPPUSDM" in df.columns and "GOLDAMGBD228NLBM" in df.columns:
        df["copper_gold_ratio"] = df["PCOPPUSDM"] / (df["GOLDAMGBD228NLBM"] + 1e-6)

    # Yield curve slope (10Y - 2Y)
    if "DGS10" in df.columns and "DGS2" in df.columns:
        df["yield_curve_slope"] = df["DGS10"] - df["DGS2"]

    return df


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not HAS_FREDAPI:
        print("[ERROR] Instala fredapi: pip install fredapi")
        return

    # Obtener API key
    api_key = args.api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        print("[ERROR] FRED_API_KEY no configurado. Agrega a .env o usa --api-key")
        return

    fred = Fred(api_key=api_key)

    print("=" * 60)
    print("INGESTA DE DATOS MACRO/COMMODITIES DESDE FRED")
    print("=" * 60)
    print(f"\nFecha inicio: {args.start_date}")
    print(f"Series a descargar: {len(FRED_SERIES)}")

    # Descargar todas las series
    all_dataframes: list[pd.DataFrame] = []

    print("\n[1/4] Descargando series desde FRED...")
    for series_id, description in FRED_SERIES.items():
        df = download_fred_series(fred, series_id, args.start_date, description)
        if not df.empty:
            all_dataframes.append(df)

    if not all_dataframes:
        print("[ERROR] No se descargó ninguna serie")
        return

    # Merge todas las series por fecha
    print("\n[2/4] Combinando series...")
    merged = all_dataframes[0]
    for df in all_dataframes[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    print(f"  Total observaciones: {len(merged)}")
    print(f"  Rango de fechas: {merged['date'].min()} a {merged['date'].max()}")

    # Resamplear a frecuencia diaria (forward-fill para series mensuales/trimestrales)
    print("\n[3/4] Resampleando a frecuencia diaria...")
    merged_daily = resample_to_daily(merged, "all_series")
    print(f"  Observaciones diarias: {len(merged_daily)}")

    # Calcular features derivados para cada serie
    print("\n[4/4] Calculando features derivados...")
    for series_id in FRED_SERIES.keys():
        if series_id in merged_daily.columns:
            merged_daily = calculate_derived_features(merged_daily, series_id)

    # Calcular ratios entre commodities
    merged_daily = calculate_ratios(merged_daily)

    # Limpiar valores infinitos y NaN
    merged_daily = merged_daily.replace([np.inf, -np.inf], np.nan)
    merged_daily = merged_daily.fillna(method="ffill").fillna(method="bfill")

    # Guardar
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"fred_series_{timestamp_str}.parquet"

    merged_daily.to_parquet(output_path, index=False)
    print(f"\n[OK] Datos guardados en: {output_path}")
    print(f"  Columnas: {len(merged_daily.columns)}")
    print(f"  Filas: {len(merged_daily)}")
    print(f"  Rango: {merged_daily['date'].min()} a {merged_daily['date'].max()}")

    # También guardar versión sin timestamp para uso en pipeline
    latest_path = output_dir / "fred_series_latest.parquet"
    merged_daily.to_parquet(latest_path, index=False)
    print(f"  También guardado como: {latest_path}")

    print("\n" + "=" * 60)
    print("[COMPLETADO] INGESTA FRED")
    print("=" * 60)


if __name__ == "__main__":
    main()



