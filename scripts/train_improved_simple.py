"""
Training mejorado SIMPLIFICADO para memoria limitada
Sin rolling windows complejos - usa datos existentes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[1]

def log(msg):
    print(msg, flush=True)

log("=" * 60)
log("ENTRENAMIENTO MEJORADO SIMPLE")
log("=" * 60)

# Cargar solo columnas necesarias
log("\n[1/4] Cargando datos...")

needed_cols = [
    'date', 'ticker', 'target',
    'roic', 'returnOnEquity', 'returnOnAssets',
    'grossProfitMargin', 'netProfitMargin',
    'freeCashFlowYield', 'freeCashFlowPerShare',
    'priceToBookRatio', 'priceToSalesRatio', 'enterpriseValue',
    'volume', 'volume_sma_20',
    'sma_200', 'sma_50', 'sma_20',
    'ema_20', 'ema_50',
    'rsi_14', 'macd', 'macd_signal',
    'atr_14', 'volatility_30d',
]

train_df = pd.read_parquet(BASE_DIR / 'data/gold/train.parquet', columns=needed_cols)
val_df = pd.read_parquet(BASE_DIR / 'data/gold/val.parquet', columns=needed_cols)
test_df = pd.read_parquet(BASE_DIR / 'data/gold/test.parquet', columns=needed_cols)

log(f"  train: {len(train_df)} filas, {len(train_df.columns)} cols")
log(f"  val: {len(val_df)} filas")
log(f"  test: {len(test_df)} filas")

# Entrenar Quality Model (SIN features históricas complejas)
log("\n[2/4] Entrenando QUALITY MODEL (sin rolling)...")

import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

# Features simples disponibles
quality_features = [
    'returnOnEquity', 'returnOnAssets',
    'grossProfitMargin', 'netProfitMargin',
    'freeCashFlowYield', 'freeCashFlowPerShare',
]

quality_features = [f for f in quality_features if f in train_df.columns]
log(f"  Features: {quality_features}")

X_train_quality = train_df[quality_features].fillna(0).astype('float32')
X_val_quality = val_df[quality_features].fillna(0).astype('float32')
X_test_quality = test_df[quality_features].fillna(0).astype('float32')

# Labels: Top 20% ROIC por fecha
if 'roic' not in train_df.columns:
    log("  [ERROR] No hay 'roic'")
    exit(1)

log("  Calculando labels por fecha...")
train_df['roic_pct'] = train_df.groupby('date')['roic'].rank(pct=True)
train_df['is_quality'] = (train_df['roic_pct'] > 0.80).astype(int)

val_df['roic_pct'] = val_df.groupby('date')['roic'].rank(pct=True)
val_df['is_quality'] = (val_df['roic_pct'] > 0.80).astype(int)

test_df['roic_pct'] = test_df.groupby('date')['roic'].rank(pct=True)
test_df['is_quality'] = (test_df['roic_pct'] > 0.80).astype(int)

y_train_quality = train_df['is_quality']
y_val_quality = val_df['is_quality']
y_test_quality = test_df['is_quality']

log(f"  Train: {y_train_quality.sum()}/{len(y_train_quality)} ({y_train_quality.mean():.2%})")

scale_pos_weight = max((len(y_train_quality) - y_train_quality.sum()) / max(y_train_quality.sum(), 1), 1.0)

log("  Training XGBoost...")
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
    eval_metric='auc',
)

quality_model.fit(
    X_train_quality, y_train_quality,
    eval_set=[(X_val_quality, y_val_quality)],
    verbose=False,
)

y_pred_proba_test = quality_model.predict_proba(X_test_quality)[:, 1]
y_pred_test = quality_model.predict(X_test_quality)

log(f"  Test AUC: {roc_auc_score(y_test_quality, y_pred_proba_test):.4f}")
log(f"  Test Acc: {accuracy_score(y_test_quality, y_pred_test):.4f}")

joblib.dump(quality_model, BASE_DIR / 'models' / 'improved_quality_simple.pkl')
log("  [OK] Guardado")

# Valuation Model
log("\n[3/4] Entrenando VALUATION MODEL...")

from sklearn.metrics import mean_squared_error, r2_score

valuation_features = [
    'priceToBookRatio', 'priceToSalesRatio', 'enterpriseValue',
    'returnOnEquity', 'roic', 'grossProfitMargin', 'netProfitMargin',
    'freeCashFlowYield',
]

valuation_features = [f for f in valuation_features if f in train_df.columns]
log(f"  Features: {valuation_features}")

X_train_val = train_df[valuation_features].fillna(0).astype('float32')
X_val_val = val_df[valuation_features].fillna(0).astype('float32')
X_test_val = test_df[valuation_features].fillna(0).astype('float32')

y_train_val = train_df['target'].fillna(0)
y_val_val = val_df['target'].fillna(0)
y_test_val = test_df['target'].fillna(0)

log("  Training XGBoost...")
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
    eval_metric='rmse',
)

valuation_model.fit(
    X_train_val, y_train_val,
    eval_set=[(X_val_val, y_val_val)],
    verbose=False,
)

y_pred_test = valuation_model.predict(X_test_val)

log(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test_val, y_pred_test)):.4f}")
log(f"  Test R²: {r2_score(y_test_val, y_pred_test):.4f}")

joblib.dump(valuation_model, BASE_DIR / 'models' / 'improved_valuation_simple.pkl')
log("  [OK] Guardado")

# Momentum Model
log("\n[4/4] Entrenando MOMENTUM MODEL...")

momentum_features = [
    'volume', 'volume_sma_20',
    'sma_200', 'sma_50', 'sma_20',
    'ema_20', 'ema_50',
    'rsi_14', 'macd', 'macd_signal',
    'atr_14', 'volatility_30d',
]

momentum_features = [f for f in momentum_features if f in train_df.columns]
log(f"  Features: {momentum_features}")

X_train_momentum = train_df[momentum_features].fillna(0).astype('float32')
X_val_momentum = val_df[momentum_features].fillna(0).astype('float32')
X_test_momentum = test_df[momentum_features].fillna(0).astype('float32')

y_train_momentum = (train_df['target'] > 0).astype(int)
y_val_momentum = (val_df['target'] > 0).astype(int)
y_test_momentum = (test_df['target'] > 0).astype(int)

log("  Training XGBoost...")
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
    eval_metric='auc',
)

momentum_model.fit(
    X_train_momentum, y_train_momentum,
    eval_set=[(X_val_momentum, y_val_momentum)],
    verbose=False,
)

y_pred_proba_test = momentum_model.predict_proba(X_test_momentum)[:, 1]
y_pred_test = momentum_model.predict(X_test_momentum)

log(f"  Test AUC: {roc_auc_score(y_test_momentum, y_pred_proba_test):.4f}")
log(f"  Test Acc: {accuracy_score(y_test_momentum, y_pred_test):.4f}")

joblib.dump(momentum_model, BASE_DIR / 'models' / 'improved_momentum_simple.pkl')
log("  [OK] Guardado")

# Feature config
feature_config = {
    'quality_features': quality_features,
    'valuation_features': valuation_features,
    'momentum_features': momentum_features,
}

joblib.dump(feature_config, BASE_DIR / 'models' / 'improved_feature_config_simple.pkl')

log("\n" + "=" * 60)
log("COMPLETADO")
log("=" * 60)
log("\nNOTA: Versión simplificada sin rolling windows complejos")
log("Para el notebook completo con features históricas, ejecutar en Colab con más RAM")
