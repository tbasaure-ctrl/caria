"""
Script to compute regime risk statistics by ticker and regime.

This script calculates distribution statistics (mean, skewness, kurtosis, VaR, CVaR)
for forward returns grouped by ticker and regime.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import os
import sys
from pathlib import Path

def compute_regime_risk_stats(
    df: pd.DataFrame,
    ret_col: str = "future_ret_10d",   # nombre columna de retornos futuros
    regime_col: str = "regime",        # nombre columna del régimen
    ticker_col: str = "ticker",        # nombre columna del ticker
    alpha: float = 0.05                # nivel de cola para VaR/CVaR
) -> pd.DataFrame:
    """
    Calcula estadísticas de distribución de retornos por ticker y régimen:
    - mean
    - skewness
    - excess kurtosis
    - VaR (alpha)
    - CVaR (Expected Shortfall, alpha)
    
    df debe contener:
        - una fila por día (o por observación de inicio de ventana),
        - retornos futuros ya calculados (ej: 10-day forward return),
        - etiqueta de régimen (ej: "Normal" / "Fragile" o 0/1),
        - ticker.
    """
    
    rows = []

    for ticker, df_ticker in df.groupby(ticker_col):
        for regime, df_regime in df_ticker.groupby(regime_col):
            r = df_regime[ret_col].dropna().values
            
            # Evitar stats basura con muy pocos datos
            if len(r) < 30:
                continue
            
            mean_r = np.mean(r)
            skew_r = skew(r, bias=False)
            # kurtosis de SciPy con fisher=True da "excess kurtosis" (0 = normal)
            kurt_r = kurtosis(r, fisher=True, bias=False)
            
            # VaR alpha (ej: 5%) como cuantil de la distribución de retornos
            var_alpha = np.quantile(r, alpha)
            
            # CVaR = Expected Shortfall = media de los retornos ≤ VaR
            tail = r[r <= var_alpha]
            cvar_alpha = np.mean(tail) if len(tail) > 0 else np.nan
            
            rows.append({
                "Ticker": ticker,
                "Regime": regime,
                "N_obs": len(r),
                "Mean": mean_r,
                "Skewness": skew_r,
                "ExcessKurtosis": kurt_r,
                f"VaR_{int(alpha*100)}": var_alpha,
                f"CVaR_{int(alpha*100)}": cvar_alpha
            })
    
    stats_df = pd.DataFrame(rows)
    
    # Ordenar para que salga Normal primero y Fragile después si usas esos nombres
    if stats_df["Regime"].dtype == object:
        order = ["Normal", "Fragile"]
        stats_df["Regime"] = pd.Categorical(stats_df["Regime"], categories=order, ordered=True)
        stats_df = stats_df.sort_values(["Ticker", "Regime"])
    else:
        stats_df = stats_df.sort_values(["Ticker", "Regime"])
    
    return stats_df


def load_data_from_sources():
    """Try to load data from existing data sources in the workspace."""
    df_all = None
    
    # Try loading from silver/market data
    possible_paths = [
        "silver/market/stock_prices_daily.parquet",
        "data/silver/market/stock_prices_daily.parquet",
        "caria_data/data/silver/market/stock_prices_daily.parquet",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found data at: {path}")
            try:
                df_prices = pd.read_parquet(path)
                print(f"  Loaded {len(df_prices)} price records")
                print(f"  Columns: {df_prices.columns.tolist()}")
                
                # Check if we have the required columns
                if 'ticker' not in df_prices.columns and 'symbol' in df_prices.columns:
                    df_prices['ticker'] = df_prices['symbol']
                
                # Try to load regime data
                regime_paths = [
                    "silver/macro/fred_us.parquet",
                    "data/silver/macro/fred_us.parquet",
                ]
                
                for rpath in regime_paths:
                    if os.path.exists(rpath):
                        try:
                            df_regime = pd.read_parquet(rpath)
                            print(f"  Found regime data at: {rpath}")
                            
                            # Merge if we have dates
                            if 'date' in df_prices.columns and 'date' in df_regime.columns:
                                # Try to merge
                                if 'regime' in df_regime.columns:
                                    df_all = pd.merge(df_prices, df_regime[['date', 'regime']], on='date', how='inner')
                                    print(f"  Merged data: {len(df_all)} records")
                                    
                                    # Calculate forward returns if we have close prices
                                    if 'close' in df_all.columns:
                                        df_all = df_all.sort_values(['ticker', 'date'])
                                        df_all['future_ret_10d'] = df_all.groupby('ticker')['close'].pct_change(10).shift(-10)
                                        print(f"  Calculated future_ret_10d")
                                        
                                        # Filter to required columns
                                        required_cols = ['ticker', 'regime', 'future_ret_10d']
                                        if all(col in df_all.columns for col in required_cols):
                                            df_all = df_all[required_cols].dropna()
                                            print(f"  Final dataset: {len(df_all)} records with all required columns")
                                            return df_all
                        except Exception as e:
                            print(f"  Error loading regime data: {e}")
                            continue
            except Exception as e:
                print(f"  Error loading price data: {e}")
                continue
    
    return None


def create_sample_data():
    """Create sample data for testing if real data is not available."""
    print("Creating sample data for demonstration...")
    
    np.random.seed(42)
    n_tickers = 5
    n_days = 500
    tickers = [f"TICKER_{i}" for i in range(1, n_tickers + 1)]
    regimes = ["Normal", "Fragile"]
    
    rows = []
    for ticker in tickers:
        for day in range(n_days):
            # Randomly assign regime
            regime = np.random.choice(regimes, p=[0.7, 0.3])
            
            # Generate returns with different distributions per regime
            if regime == "Normal":
                # Normal regime: lower volatility, positive mean
                ret = np.random.normal(0.001, 0.02)
            else:
                # Fragile regime: higher volatility, negative skew
                ret = np.random.normal(-0.002, 0.04)
                # Add some negative tail events
                if np.random.random() < 0.1:
                    ret -= 0.05
            
            rows.append({
                'ticker': ticker,
                'regime': regime,
                'future_ret_10d': ret
            })
    
    df_all = pd.DataFrame(rows)
    print(f"Created sample dataset: {len(df_all)} records")
    print(f"  Tickers: {df_all['ticker'].unique()}")
    print(f"  Regimes: {df_all['regime'].unique()}")
    
    return df_all


if __name__ == "__main__":
    print("=" * 60)
    print("Computing Regime Risk Statistics")
    print("=" * 60)
    
    # Try to load real data
    df_all = load_data_from_sources()
    
    # If no real data, create sample data
    if df_all is None or len(df_all) == 0:
        print("\nNo suitable data found in workspace. Using sample data.")
        df_all = create_sample_data()
    else:
        print("\nUsing data from workspace.")
    
    # Check required columns
    required_cols = ['ticker', 'regime', 'future_ret_10d']
    missing_cols = [col for col in required_cols if col not in df_all.columns]
    
    if missing_cols:
        print(f"\nERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {df_all.columns.tolist()}")
        print("\nPlease ensure your dataframe has:")
        print("  - 'ticker' (or 'symbol' which will be renamed)")
        print("  - 'regime' (regime labels)")
        print("  - 'future_ret_10d' (10-day forward returns)")
        sys.exit(1)
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_regime_risk_stats(
        df_all,
        ret_col="future_ret_10d",
        regime_col="regime",
        ticker_col="ticker",
        alpha=0.05   # VaR/CVaR al 5%
    )
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(stats)
    
    # Save results
    output_path = "regime_risk_stats.csv"
    stats.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary by Regime:")
    print("=" * 60)
    if len(stats) > 0:
        summary = stats.groupby('Regime', observed=True).agg({
            'Mean': 'mean',
            'Skewness': 'mean',
            'ExcessKurtosis': 'mean',
            'VaR_5': 'mean',
            'CVaR_5': 'mean',
            'N_obs': 'sum'
        })
        print(summary)
