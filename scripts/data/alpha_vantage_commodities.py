"""Descarga de datos de commodities desde Alpha Vantage API."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[3]

# Mapeo de commodities a símbolos Alpha Vantage
# Alpha Vantage usa símbolos de futuros de commodities
COMMODITY_SYMBOLS = {
    "WTI": "CL",  # Crude Oil WTI
    "BRENT": "BZ",  # Brent Crude
    "GOLD": "GC",  # Gold
    "SILVER": "SI",  # Silver
    "COPPER": "HG",  # Copper
    "NATURAL_GAS": "NG",  # Natural Gas
    "CORN": "ZC",  # Corn
    "WHEAT": "ZW",  # Wheat
    "SOYBEAN": "ZS",  # Soybeans
    "SUGAR": "SB",  # Sugar
    "COFFEE": "KC",  # Coffee
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga commodities desde Alpha Vantage")
    parser.add_argument(
        "--api-key",
        help="Alpha Vantage API key (o usar ALPHA_VANTAGE_API_KEY en .env)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "silver" / "macro"),
        help="Directorio de salida",
    )
    parser.add_argument(
        "--start-date",
        default="1900-01-01",
        help="Fecha de inicio (YYYY-MM-DD)",
    )
    return parser.parse_args()


def download_commodity_alpha_vantage(
    api_key: str, symbol: str, commodity_name: str
) -> pd.DataFrame:
    """Descarga datos de commodity desde Alpha Vantage.
    
    Nota: Alpha Vantage tiene límites de rate (5 calls/min, 500 calls/day).
    Para datos históricos extensos, es mejor usar FRED cuando esté disponible.
    """
    base_url = "https://www.alphavantage.co/query"
    
    # Alpha Vantage tiene endpoints limitados para commodities
    # Usamos el endpoint de commodities que está disponible
    params = {
        "function": "COMMODITY_CHANNELS",
        "symbol": symbol,
        "interval": "daily",
        "apikey": api_key,
        "datatype": "csv"
    }
    
    try:
        print(f"  Descargando {commodity_name} ({symbol})...")
        response = requests.get(base_url, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"  [ERROR] {commodity_name}: HTTP {response.status_code}")
            return pd.DataFrame()
        
        # Alpha Vantage puede devolver JSON con error
        if "Error Message" in response.text or "Note" in response.text:
            print(f"  [WARNING] {commodity_name}: API limit o error. Usando FRED en su lugar.")
            return pd.DataFrame()
        
        # Intentar parsear como CSV
        try:
            df = pd.read_csv(pd.StringIO(response.text))
            if df.empty:
                return pd.DataFrame()
            
            # Normalizar columnas
            if "time" in df.columns:
                df.rename(columns={"time": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            
            print(f"  [OK] {commodity_name}: {len(df)} observaciones")
            return df
            
        except Exception as e:
            print(f"  [WARNING] {commodity_name}: Error parseando respuesta: {e}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"  [ERROR] {commodity_name}: {e}")
        return pd.DataFrame()
    
    finally:
        # Rate limiting: Alpha Vantage permite 5 calls/min
        time.sleep(12)  # Esperar 12 segundos entre llamadas


def main() -> None:
    load_dotenv()
    args = parse_args()
    
    api_key = args.api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("[ERROR] ALPHA_VANTAGE_API_KEY no configurado. Agrega a .env o usa --api-key")
        print("[INFO] Alpha Vantage tiene límites estrictos (5 calls/min).")
        print("[INFO] Para datos históricos extensos, usa FRED cuando esté disponible.")
        return
    
    print("=" * 60)
    print("DESCARGA DE COMMODITIES DESDE ALPHA VANTAGE")
    print("=" * 60)
    print("\n[NOTA] Alpha Vantage tiene límites de rate (5 calls/min, 500 calls/day).")
    print("Para datos históricos extensos desde 1900, se recomienda usar FRED.")
    print(f"\nDescargando {len(COMMODITY_SYMBOLS)} commodities...")
    
    all_dataframes = []
    
    for commodity_name, symbol in COMMODITY_SYMBOLS.items():
        df = download_commodity_alpha_vantage(api_key, symbol, commodity_name)
        if not df.empty:
            # Renombrar columnas para incluir el nombre del commodity
            price_col = [c for c in df.columns if c not in ["date"] and df[c].dtype in [float, int]]
            if price_col:
                df.rename(columns={price_col[0]: f"{commodity_name}_price"}, inplace=True)
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("\n[WARNING] No se descargaron datos. Verifica tu API key y límites de rate.")
        return
    
    # Merge todas las series
    print("\nCombinando series...")
    merged = all_dataframes[0]
    for df in all_dataframes[1:]:
        merged = merged.merge(df, on="date", how="outer")
    
    merged = merged.sort_values("date").reset_index(drop=True)
    
    # Guardar
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "alpha_vantage_commodities.parquet"
    merged.to_parquet(output_path, index=False)
    
    print(f"\n[OK] Datos guardados en: {output_path}")
    print(f"  Columnas: {len(merged.columns)}")
    print(f"  Filas: {len(merged)}")
    print(f"  Rango: {merged['date'].min()} a {merged['date'].max()}")
    print("\n" + "=" * 60)
    print("[COMPLETADO]")
    print("=" * 60)


if __name__ == "__main__":
    main()

