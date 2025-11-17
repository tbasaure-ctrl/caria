"""
Entrenamiento mejorado local (adaptado del notebook Colab)
Junta capas gold + macro sin cambiar estructura
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
log("ENTRENAMIENTO MEJORADO (LOCAL)")
log("=" * 60)

# 1. Cargar datos gold (solo columnas necesarias para reducir memoria)
log("\n[1/5] Cargando datos gold...")

# Columnas necesarias
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

log(f"  train: {len(train_df)} filas")
log(f"  val: {len(val_df)} filas")
log(f"  test: {len(test_df)} filas")

# 2. Calcular features históricas para evitar leakage
log("\n[2/5] Calculando features históricas relativas...")

def add_historical_features(df):
    """Agrega features históricas (percentiles, lags, promedios) para evitar leakage."""
    df = df.copy()

    # Asegurar que está ordenado por ticker y fecha
    if 'ticker' in df.columns:
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    else:
        df = df.sort_values('date').reset_index(drop=True)

    # Percentiles históricos de múltiplos (5 años = ~1260 trading days)
    valuation_cols = ['priceToBookRatio', 'priceToSalesRatio', 'enterpriseValue', 'freeCashFlowYield']
    for col in valuation_cols:
        if col in df.columns:
            # Percentil histórico por ticker
            if 'ticker' in df.columns:
                df[f'{col}_percentile_5y'] = df.groupby('ticker')[col].transform(
                    lambda x: x.rolling(window=1260, min_periods=252).apply(
                        lambda y: (y.iloc[-1] > y.iloc[:-1]).sum() / max(len(y.iloc[:-1]), 1)
                    )
                )
            else:
                df[f'{col}_percentile_5y'] = df[col].rolling(window=1260, min_periods=252).apply(
                    lambda y: (y.iloc[-1] > y.iloc[:-1]).sum() / max(len(y.iloc[:-1]), 1)
                )

    # ROIC/ROE históricos con lags para evitar leakage
    quality_cols = ['roic', 'returnOnEquity', 'returnOnAssets']
    for col in quality_cols:
        if col in df.columns:
            # Promedio histórico (3 años)
            if 'ticker' in df.columns:
                df[f'{col}_3y_avg'] = df.groupby('ticker')[col].transform(
                    lambda x: x.rolling(window=756, min_periods=252).mean()
                )
                # Lags (trimestres anteriores)
                df[f'{col}_lag_1q'] = df.groupby('ticker')[col].shift(63)
                df[f'{col}_lag_2q'] = df.groupby('ticker')[col].shift(126)
            else:
                df[f'{col}_3y_avg'] = df[col].rolling(window=756, min_periods=252).mean()
                df[f'{col}_lag_1q'] = df[col].shift(63)
                df[f'{col}_lag_2q'] = df[col].shift(126)

    return df

# Aplicar a train, val, test
train_df = add_historical_features(train_df)
val_df = add_historical_features(val_df)
test_df = add_historical_features(test_df)

log(f"  Nuevas columnas: {len([c for c in train_df.columns if 'percentile' in c or 'lag' in c or '_3y_avg' in c])}")

# 3. Entrenar Quality Model
log("\n[3/5] Entrenando QUALITY MODEL...")

import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Features para Quality Model (SIN roic actual para evitar leakage)
quality_features = [
    'roic_lag_1q', 'roic_lag_2q', 'roic_3y_avg',
    'returnOnEquity', 'returnOnAssets',
    'grossProfitMargin', 'netProfitMargin',
    'freeCashFlowYield', 'freeCashFlowPerShare',
]

# Filtrar solo features que existen
quality_features = [f for f in quality_features if f in train_df.columns]
log(f"  Features usadas ({len(quality_features)}): {quality_features}")

# Preparar datos
X_train_quality = train_df[quality_features].fillna(0).astype('float32')
X_val_quality = val_df[quality_features].fillna(0).astype('float32')
X_test_quality = test_df[quality_features].fillna(0).astype('float32')

# Crear labels: Top 20% de ROIC POR FECHA (adaptado al régimen económico)
log("  Creando labels por fecha (percentiles)...")
if 'roic' not in train_df.columns:
    log("  [ERROR] No hay columna 'roic' en train_df")
    log(f"  Columnas disponibles: {train_df.columns.tolist()[:10]}...")
    exit(1)

train_df['roic_percentile_by_date'] = train_df.groupby('date')['roic'].rank(pct=True)
train_df['is_quality'] = (train_df['roic_percentile_by_date'] > 0.80).astype(int)

val_df['roic_percentile_by_date'] = val_df.groupby('date')['roic'].rank(pct=True)
val_df['is_quality'] = (val_df['roic_percentile_by_date'] > 0.80).astype(int)

test_df['roic_percentile_by_date'] = test_df.groupby('date')['roic'].rank(pct=True)
test_df['is_quality'] = (test_df['roic_percentile_by_date'] > 0.80).astype(int)

y_train_quality = train_df['is_quality']
y_val_quality = val_df['is_quality']
y_test_quality = test_df['is_quality']

log(f"  Train: {y_train_quality.sum()} / {len(y_train_quality)} positivos ({y_train_quality.mean():.2%})")
log(f"  Val: {y_val_quality.sum()} / {len(y_val_quality)} positivos ({y_val_quality.mean():.2%})")

# Calcular scale_pos_weight para balancear clases
scale_pos_weight = max((len(y_train_quality) - y_train_quality.sum()) / max(y_train_quality.sum(), 1), 1.0)

log(f"  scale_pos_weight: {scale_pos_weight:.2f}")

# Entrenar modelo con hiperparámetros anti-overfitting
log("  Entrenando modelo XGBoost...")
quality_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=3,  # Reducido para evitar overfitting
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
    X_train_quality,
    y_train_quality,
    eval_set=[(X_val_quality, y_val_quality)],
    early_stopping_rounds=50,
    verbose=False,
)

# Evaluar
y_pred_proba_train = quality_model.predict_proba(X_train_quality)[:, 1]
y_pred_proba_val = quality_model.predict_proba(X_val_quality)[:, 1]
y_pred_proba_test = quality_model.predict_proba(X_test_quality)[:, 1]

y_pred_train = quality_model.predict(X_train_quality)
y_pred_val = quality_model.predict(X_val_quality)
y_pred_test = quality_model.predict(X_test_quality)

log("\n  Resultados Quality Model:")
log(f"    Train - Accuracy: {accuracy_score(y_train_quality, y_pred_train):.4f}, AUC: {roc_auc_score(y_train_quality, y_pred_proba_train):.4f}")
log(f"    Val   - Accuracy: {accuracy_score(y_val_quality, y_pred_val):.4f}, AUC: {roc_auc_score(y_val_quality, y_pred_proba_val):.4f}")
log(f"    Test  - Accuracy: {accuracy_score(y_test_quality, y_pred_test):.4f}, AUC: {roc_auc_score(y_test_quality, y_pred_proba_test):.4f}")

# Guardar
output_path = BASE_DIR / 'models' / 'improved_quality_model.pkl'
joblib.dump(quality_model, output_path)
log(f"\n  [OK] Modelo guardado: {output_path}")

# 4. Entrenar Valuation Model
log("\n[4/5] Entrenando VALUATION MODEL...")

from sklearn.metrics import mean_squared_error, r2_score

valuation_features = [
    # Múltiplos relativos históricos
    'priceToBookRatio_percentile_5y',
    'priceToSalesRatio_percentile_5y',
    'freeCashFlowYield_percentile_5y',
    # Fundamentales
    'priceToBookRatio', 'priceToSalesRatio', 'enterpriseValue',
    'returnOnEquity', 'roic', 'grossProfitMargin', 'netProfitMargin',
    'freeCashFlowYield',
]

# Filtrar solo features que existen
valuation_features = [f for f in valuation_features if f in train_df.columns]
log(f"  Features usadas ({len(valuation_features)}): {valuation_features}")

# Preparar datos
X_train_val = train_df[valuation_features].fillna(0).astype('float32')
X_val_val = val_df[valuation_features].fillna(0).astype('float32')
X_test_val = test_df[valuation_features].fillna(0).astype('float32')

# Target: forward returns
y_train_val = train_df['target'].fillna(0)
y_val_val = val_df['target'].fillna(0)
y_test_val = test_df['target'].fillna(0)

log(f"  Train target: mean={y_train_val.mean():.4f}, std={y_train_val.std():.4f}")

# Entrenar modelo
log("  Entrenando modelo XGBoost...")
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
    X_train_val,
    y_train_val,
    eval_set=[(X_val_val, y_val_val)],
    early_stopping_rounds=50,
    verbose=False,
)

# Evaluar
y_pred_train = valuation_model.predict(X_train_val)
y_pred_val = valuation_model.predict(X_val_val)
y_pred_test = valuation_model.predict(X_test_val)

log("\n  Resultados Valuation Model:")
log(f"    Train - RMSE: {np.sqrt(mean_squared_error(y_train_val, y_pred_train)):.4f}, R²: {r2_score(y_train_val, y_pred_train):.4f}")
log(f"    Val   - RMSE: {np.sqrt(mean_squared_error(y_val_val, y_pred_val)):.4f}, R²: {r2_score(y_val_val, y_pred_val):.4f}")
log(f"    Test  - RMSE: {np.sqrt(mean_squared_error(y_test_val, y_pred_test)):.4f}, R²: {r2_score(y_test_val, y_pred_test):.4f}")

# Guardar
output_path = BASE_DIR / 'models' / 'improved_valuation_model.pkl'
joblib.dump(valuation_model, output_path)
log(f"\n  [OK] Modelo guardado: {output_path}")

# 5. Entrenar Momentum Model
log("\n[5/5] Entrenando MOMENTUM MODEL...")

momentum_features = [
    'volume', 'volume_sma_20',
    'sma_200', 'sma_50', 'sma_20',
    'ema_20', 'ema_50',
    'rsi_14',
    'macd', 'macd_signal',
    'atr_14', 'volatility_30d',
]

# Filtrar solo features que existen
momentum_features = [f for f in momentum_features if f in train_df.columns]
log(f"  Features usadas ({len(momentum_features)}): {momentum_features}")

# Preparar datos
X_train_momentum = train_df[momentum_features].fillna(0).astype('float32')
X_val_momentum = val_df[momentum_features].fillna(0).astype('float32')
X_test_momentum = test_df[momentum_features].fillna(0).astype('float32')

# Target: Dirección de retorno
y_train_momentum = (train_df['target'] > 0).astype(int)
y_val_momentum = (val_df['target'] > 0).astype(int)
y_test_momentum = (test_df['target'] > 0).astype(int)

log(f"  Train: {y_train_momentum.sum()} / {len(y_train_momentum)} positivos ({y_train_momentum.mean():.2%})")

# Entrenar modelo
log("  Entrenando modelo XGBoost...")
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
    X_train_momentum,
    y_train_momentum,
    eval_set=[(X_val_momentum, y_val_momentum)],
    early_stopping_rounds=50,
    verbose=False,
)

# Evaluar
y_pred_proba_train = momentum_model.predict_proba(X_train_momentum)[:, 1]
y_pred_proba_val = momentum_model.predict_proba(X_val_momentum)[:, 1]
y_pred_proba_test = momentum_model.predict_proba(X_test_momentum)[:, 1]

y_pred_train = momentum_model.predict(X_train_momentum)
y_pred_val = momentum_model.predict(X_val_momentum)
y_pred_test = momentum_model.predict(X_test_momentum)

log("\n  Resultados Momentum Model:")
log(f"    Train - Accuracy: {accuracy_score(y_train_momentum, y_pred_train):.4f}, AUC: {roc_auc_score(y_train_momentum, y_pred_proba_train):.4f}")
log(f"    Val   - Accuracy: {accuracy_score(y_val_momentum, y_pred_val):.4f}, AUC: {roc_auc_score(y_val_momentum, y_pred_proba_val):.4f}")
log(f"    Test  - Accuracy: {accuracy_score(y_test_momentum, y_pred_test):.4f}, AUC: {roc_auc_score(y_test_momentum, y_pred_proba_test):.4f}")

# Guardar
output_path = BASE_DIR / 'models' / 'improved_momentum_model.pkl'
joblib.dump(momentum_model, output_path)
log(f"\n  [OK] Modelo guardado: {output_path}")

# Guardar feature config
feature_config = {
    'quality_features': quality_features,
    'valuation_features': valuation_features,
    'momentum_features': momentum_features,
}

feature_config_path = BASE_DIR / 'models' / 'improved_feature_config.pkl'
joblib.dump(feature_config, feature_config_path)
log(f"\n[OK] Feature config guardado: {feature_config_path}")

log("\n" + "=" * 60)
log("ENTRENAMIENTO MEJORADO COMPLETADO")
log("=" * 60)
