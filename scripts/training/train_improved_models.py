"""Script local para entrenar modelos mejorados (versión sin Colab)."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR / "src"))

# Importar funciones de feature engineering
try:
    from scripts.feature_engineering.macro_features import (
        calculate_macro_features,
        calculate_relative_features,
    )
except ImportError:
    print("[WARNING] No se pudieron importar funciones de feature engineering")
    calculate_macro_features = None
    calculate_relative_features = None

print("=" * 60)
print("ENTRENAMIENTO MODELOS MEJORADOS (LOCAL)")
print("=" * 60)

# Cargar datos
print("\n[1/6] Cargando datos...")
train_df = pd.read_parquet(BASE_DIR / "data" / "gold" / "train.parquet")
val_df = pd.read_parquet(BASE_DIR / "data" / "gold" / "val.parquet")
test_df = pd.read_parquet(BASE_DIR / "data" / "gold" / "test.parquet")

print(f"  Train: {len(train_df)} rows")
print(f"  Val: {len(val_df)} rows")
print(f"  Test: {len(test_df)} rows")

# Cargar datos macro si existen
print("\n[2/6] Cargando datos macro...")
macro_path = BASE_DIR / "data" / "silver" / "macro" / "macro_features.parquet"
if macro_path.exists():
    macro_df = pd.read_parquet(macro_path)
    print(f"  ✓ Datos macro cargados: {len(macro_df)} rows")
    
    # Combinar con stocks
    train_df["date"] = pd.to_datetime(train_df["date"])
    val_df["date"] = pd.to_datetime(val_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    
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
    macro_cols_to_merge = [c for c in macro_cols_to_merge if c in macro_df.columns]
    macro_subset = macro_df[["date"] + macro_cols_to_merge].sort_values("date")
    
    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df_sorted = (
            df.sort_values(["ticker", "date"])
            if "ticker" in df.columns
            else df.sort_values("date")
        )
        df_merged = pd.merge_asof(df_sorted, macro_subset, on="date", direction="backward")
        if df_name == "train":
            train_df = df_merged
        elif df_name == "val":
            val_df = df_merged
        else:
            test_df = df_merged
    print(f"  ✓ Datos combinados con macro")
else:
    print("  ⚠ No se encontraron datos macro (continuando sin ellos)")

# Feature engineering histórico
print("\n[3/6] Calculando features históricas...")

def add_historical_features(df):
    """Agrega features históricas (percentiles, lags, promedios)."""
    df = df.copy()
    if "ticker" in df.columns:
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    else:
        df = df.sort_values("date").reset_index(drop=True)
    
    # Percentiles históricos
    valuation_cols = ["priceToBookRatio", "priceToSalesRatio", "enterpriseValue", "freeCashFlowYield"]
    for col in valuation_cols:
        if col in df.columns:
            if "ticker" in df.columns:
                df[f"{col}_percentile_5y"] = df.groupby("ticker")[col].transform(
                    lambda x: x.rolling(window=1260, min_periods=252).apply(
                        lambda y: (y.iloc[-1] > y.iloc[:-1]).sum() / len(y.iloc[:-1])
                        if len(y.iloc[:-1]) > 0
                        else 0.5
                    )
                )
            else:
                df[f"{col}_percentile_5y"] = df[col].rolling(window=1260, min_periods=252).apply(
                    lambda y: (y.iloc[-1] > y.iloc[:-1]).sum() / len(y.iloc[:-1])
                    if len(y.iloc[:-1]) > 0
                    else 0.5
                )
    
    # Lags para quality
    quality_cols = ["roic", "returnOnEquity", "returnOnAssets"]
    for col in quality_cols:
        if col in df.columns:
            if "ticker" in df.columns:
                df[f"{col}_3y_avg"] = df.groupby("ticker")[col].transform(
                    lambda x: x.rolling(window=756, min_periods=252).mean()
                )
                df[f"{col}_lag_1q"] = df.groupby("ticker")[col].shift(63)
                df[f"{col}_lag_2q"] = df.groupby("ticker")[col].shift(126)
            else:
                df[f"{col}_3y_avg"] = df[col].rolling(window=756, min_periods=252).mean()
                df[f"{col}_lag_1q"] = df[col].shift(63)
                df[f"{col}_lag_2q"] = df[col].shift(126)
    
    return df

train_df = add_historical_features(train_df)
val_df = add_historical_features(val_df)
test_df = add_historical_features(test_df)
print("  ✓ Features históricas calculadas")

# Entrenar modelos (similar al notebook de Colab)
print("\n[4/6] Entrenando modelos...")

# Quality Model
print("\n  Entrenando Quality Model...")
quality_features = [
    "roic_lag_1q",
    "roic_lag_2q",
    "roic_3y_avg",
    "returnOnEquity",
    "returnOnAssets",
    "grossProfitMargin",
    "netProfitMargin",
    "freeCashFlowYield",
    "freeCashFlowPerShare",
    "revenueGrowth",
    "netIncomeGrowth",
]
macro_quality_features = ["recession_probability", "macro_regime", "credit_spread"]
quality_features.extend([f for f in macro_quality_features if f in train_df.columns])
quality_features = [f for f in quality_features if f in train_df.columns]

# Codificar variables categóricas antes de convertir a float32
def encode_categorical_features(df_train, df_val, df_test, features):
    """Codifica variables categóricas usando label encoding."""
    from sklearn.preprocessing import LabelEncoder
    
    df_train = df_train[features].copy()
    df_val = df_val[features].copy()
    df_test = df_test[features].copy()
    
    # Identificar columnas categóricas (object o string)
    categorical_cols = []
    for col in features:
        if col in df_train.columns:
            if df_train[col].dtype == 'object' or df_train[col].dtype.name == 'category':
                categorical_cols.append(col)
    
    # Codificar variables categóricas
    for col in categorical_cols:
        le = LabelEncoder()
        # Entrenar con train + val + test para tener todos los valores posibles
        all_values = pd.concat([df_train[col], df_val[col], df_test[col]], axis=0).dropna().astype(str)
        le.fit(all_values.unique())
        
        # Transformar cada dataset
        df_train[col] = df_train[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        df_val[col] = df_val[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        df_test[col] = df_test[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    if categorical_cols:
        print(f"  Variables categóricas codificadas: {categorical_cols}")
    
    return df_train, df_val, df_test

X_train_quality, X_val_quality, X_test_quality = encode_categorical_features(
    train_df, val_df, test_df, quality_features
)

X_train_quality = X_train_quality.fillna(0).astype("float32")
X_val_quality = X_val_quality.fillna(0).astype("float32")
X_test_quality = X_test_quality.fillna(0).astype("float32")

train_df["roic_percentile_by_date"] = train_df.groupby("date")["roic"].rank(pct=True)
train_df["is_quality"] = (train_df["roic_percentile_by_date"] > 0.80).astype(int)
val_df["roic_percentile_by_date"] = val_df.groupby("date")["roic"].rank(pct=True)
val_df["is_quality"] = (val_df["roic_percentile_by_date"] > 0.80).astype(int)
test_df["roic_percentile_by_date"] = test_df.groupby("date")["roic"].rank(pct=True)
test_df["is_quality"] = (test_df["roic_percentile_by_date"] > 0.80).astype(int)

y_train_quality = train_df["is_quality"]
y_val_quality = val_df["is_quality"]
y_test_quality = test_df["is_quality"]

scale_pos_weight = (len(y_train_quality) - y_train_quality.sum()) / max(y_train_quality.sum(), 1)

quality_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.01,
    reg_alpha=2.0,
    reg_lambda=3.0,
    subsample=0.75,
    colsample_bytree=0.75,
    min_child_weight=10,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
)

quality_model.fit(
    X_train_quality,
    y_train_quality,
    eval_set=[(X_val_quality, y_val_quality)],
    early_stopping_rounds=50,
    verbose=False,
)

y_pred_proba_val = quality_model.predict_proba(X_val_quality)[:, 1]
print(f"    Val AUC: {roc_auc_score(y_val_quality, y_pred_proba_val):.4f}")

# Valuation Model
print("\n  Entrenando Valuation Model...")
valuation_features = [
    "priceToBookRatio_percentile_5y",
    "priceToSalesRatio_percentile_5y",
    "freeCashFlowYield_percentile_5y",
    "priceToBookRatio",
    "priceToSalesRatio",
    "enterpriseValue",
    "returnOnEquity",
    "roic",
    "grossProfitMargin",
    "netProfitMargin",
    "freeCashFlowYield",
    "revenueGrowth",
    "netIncomeGrowth",
    "yield_curve_slope",
    "credit_spread",
    "recession_probability",
    "gold_oil_ratio",
    "copper_gold_ratio",
]
valuation_features = [f for f in valuation_features if f in train_df.columns]

# Codificar variables categóricas antes de convertir a float32
X_train_val, X_val_val, X_test_val = encode_categorical_features(
    train_df, val_df, test_df, valuation_features
)

X_train_val = X_train_val.fillna(0).astype("float32")
X_val_val = X_val_val.fillna(0).astype("float32")
X_test_val = X_test_val.fillna(0).astype("float32")

y_train_val = train_df["target"].fillna(0)
y_val_val = val_df["target"].fillna(0)
y_test_val = test_df["target"].fillna(0)

valuation_model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.01,
    reg_alpha=1.5,
    reg_lambda=2.5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    eval_metric="rmse",
)

valuation_model.fit(
    X_train_val,
    y_train_val,
    eval_set=[(X_val_val, y_val_val)],
    early_stopping_rounds=50,
    verbose=False,
)

y_pred_val = valuation_model.predict(X_val_val)
print(f"    Val R²: {r2_score(y_val_val, y_pred_val):.4f}")

# Momentum Model
print("\n  Entrenando Momentum Model...")
momentum_features = [
    "volume",
    "volume_sma_20",
    "sma_200",
    "sma_50",
    "sma_20",
    "ema_20",
    "ema_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "atr_14",
    "volatility_30d",
]
momentum_features = [f for f in momentum_features if f in train_df.columns]

# Agregar features calculadas
if "sma_200" in train_df.columns:
    price_proxy = train_df.get("price", train_df.get("enterpriseValue", pd.Series(1.0, index=train_df.index)))
    train_df["price_above_sma200"] = (price_proxy > train_df["sma_200"]).astype(int)
    train_df["price_above_sma50"] = (price_proxy > train_df["sma_50"]).astype(int)
    val_df["price_above_sma200"] = (val_df.get("price", val_df.get("enterpriseValue", 1.0)) > val_df["sma_200"]).astype(int)
    val_df["price_above_sma50"] = (val_df.get("price", val_df.get("enterpriseValue", 1.0)) > val_df["sma_50"]).astype(int)
    test_df["price_above_sma200"] = (test_df.get("price", test_df.get("enterpriseValue", 1.0)) > test_df["sma_200"]).astype(int)
    test_df["price_above_sma50"] = (test_df.get("price", test_df.get("enterpriseValue", 1.0)) > test_df["sma_50"]).astype(int)
    momentum_features.extend(["price_above_sma200", "price_above_sma50"])

if "volume" in train_df.columns and "volume_sma_20" in train_df.columns:
    train_df["volume_ratio_20d"] = train_df["volume"] / (train_df["volume_sma_20"] + 1e-6)
    val_df["volume_ratio_20d"] = val_df["volume"] / (val_df["volume_sma_20"] + 1e-6)
    test_df["volume_ratio_20d"] = test_df["volume"] / (test_df["volume_sma_20"] + 1e-6)
    momentum_features.append("volume_ratio_20d")

momentum_features = [f for f in momentum_features if f in train_df.columns]

# Codificar variables categóricas antes de convertir a float32
X_train_momentum, X_val_momentum, X_test_momentum = encode_categorical_features(
    train_df, val_df, test_df, momentum_features
)

X_train_momentum = X_train_momentum.fillna(0).astype("float32")
X_val_momentum = X_val_momentum.fillna(0).astype("float32")
X_test_momentum = X_test_momentum.fillna(0).astype("float32")

y_train_momentum = (train_df["target"] > 0).astype(int)
y_val_momentum = (val_df["target"] > 0).astype(int)
y_test_momentum = (test_df["target"] > 0).astype(int)

momentum_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.01,
    reg_alpha=1.0,
    reg_lambda=2.0,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    eval_metric="auc",
)

momentum_model.fit(
    X_train_momentum,
    y_train_momentum,
    eval_set=[(X_val_momentum, y_val_momentum)],
    early_stopping_rounds=50,
    verbose=False,
)

y_pred_proba_val = momentum_model.predict_proba(X_val_momentum)[:, 1]
print(f"    Val AUC: {roc_auc_score(y_val_momentum, y_pred_proba_val):.4f}")

# Guardar modelos
print("\n[5/6] Guardando modelos...")
output_dir = BASE_DIR / "models"
output_dir.mkdir(exist_ok=True)

joblib.dump(quality_model, output_dir / "improved_quality_model.pkl")
joblib.dump(valuation_model, output_dir / "improved_valuation_model.pkl")
joblib.dump(momentum_model, output_dir / "improved_momentum_model.pkl")

feature_config = {
    "quality_features": quality_features,
    "valuation_features": valuation_features,
    "momentum_features": momentum_features,
}
joblib.dump(feature_config, output_dir / "improved_feature_config.pkl")

print(f"  ✓ Modelos guardados en: {output_dir}")

# Resumen
print("\n[6/6] Resumen:")
print("  ✓ Quality Model: Percentiles por fecha")
print("  ✓ Valuation Model: DCF/Múltiplos con macro")
print("  ✓ Momentum Model: Features mejoradas (volumen, SMAs, RSI)")

print("\n" + "=" * 60)
print("[COMPLETADO] ENTRENAMIENTO LOCAL")
print("=" * 60)


