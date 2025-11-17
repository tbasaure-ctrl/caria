"""Entrenar modelos correctamente separados: Quality (fundamentals), Valuation (intrinsic value), Momentum (technicals)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

print("=" * 60)
print("ENTRENAMIENTO MODELOS CORREGIDOS")
print("=" * 60)

# Load data
print("\n[1/6] Cargando datos...")
train_df = pd.read_parquet(BASE_DIR / "data/gold/train.parquet")
val_df = pd.read_parquet(BASE_DIR / "data/gold/val.parquet")

print(f"  Train: {len(train_df)} rows")
print(f"  Val: {len(val_df)} rows")

# Define feature groups
print("\n[2/6] Definiendo feature groups...")

# QUALITY features: profitability + cash generation (NO technicals, NO price)
quality_features = [
    'roic', 'roiic', 'returnOnEquity', 'returnOnAssets',
    'grossProfitMargin', 'netProfitMargin',
    'freeCashFlowYield', 'freeCashFlowPerShare',
    'capitalExpenditures', 'r_and_d'
]
quality_features = [f for f in quality_features if f in train_df.columns]
print(f"  Quality features ({len(quality_features)}): {quality_features}")

# VALUATION features: fundamentals + valuation multiples (NO technicals, NO returns)
valuation_features = [
    'priceToBookRatio', 'priceToSalesRatio', 'enterpriseValue',
    'returnOnEquity', 'roic', 'roiic',
    'grossProfitMargin', 'netProfitMargin', 'freeCashFlowYield',
    'revenueGrowth', 'netIncomeGrowth',
    'net_debt'
]
valuation_features = [f for f in valuation_features if f in train_df.columns]
print(f"  Valuation features ({len(valuation_features)}): {valuation_features}")

# MOMENTUM features: technicals + volume
momentum_features = [
    'sma_20', 'sma_50', 'sma_200',
    'ema_20', 'ema_50', 'ema_200',
    'rsi_14', 'macd', 'macd_signal',
    'atr_14', 'volatility_30d',
    'volume', 'volume_sma_20', 'volume_ratio', 'volume_change'
]
momentum_features = [f for f in momentum_features if f in train_df.columns]
print(f"  Momentum features ({len(momentum_features)}): {momentum_features}")

# Prepare data
X_train_quality = train_df[quality_features].fillna(0)
X_train_valuation = train_df[valuation_features].fillna(0)
X_train_momentum = train_df[momentum_features].fillna(0)

X_val_quality = val_df[quality_features].fillna(0)
X_val_valuation = val_df[valuation_features].fillna(0)
X_val_momentum = val_df[momentum_features].fillna(0)

y_train = train_df['target']
y_val = val_df['target']

# ===========================
# QUALITY MODEL (corrected with relative percentiles)
# ===========================
print("\n[3/6] Entrenando QUALITY MODEL...")
print("  Labels: Top 20% ROIC by date (regime-adjusted)")

# Create quality labels using RELATIVE percentiles by date (not global threshold)
# This adapts to different regimes (1970s vs 2010s had different ROIC levels)
train_df['roic_percentile'] = train_df.groupby('date')['roic'].rank(pct=True)
train_df['is_quality'] = (train_df['roic_percentile'] > 0.80).astype(int)

# If not enough samples, also use ROE percentile
if train_df['is_quality'].sum() < 1000:
    print("  [WARNING] Pocos samples de ROIC, añadiendo ROE percentile")
    train_df['roe_percentile'] = train_df.groupby('date')['returnOnEquity'].rank(pct=True)
    train_df.loc[train_df['roe_percentile'] > 0.80, 'is_quality'] = 1

print(f"  Quality samples: {train_df['is_quality'].sum()} / {len(train_df)} ({train_df['is_quality'].mean():.1%})")

quality_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=(len(train_df) - train_df['is_quality'].sum()) / train_df['is_quality'].sum()  # Handle imbalance
)
quality_model.fit(X_train_quality, train_df['is_quality'])

output_dir = BASE_DIR / "models"
output_dir.mkdir(exist_ok=True)
joblib.dump(quality_model, output_dir / "quality_model.pkl")
print("  [OK] Guardado: models/quality_model.pkl")

# ===========================
# VALUATION MODEL (corrected with relative percentiles)
# ===========================
print("\n[4/6] Entrenando VALUATION MODEL...")
print("  Target: Relatively cheap (bottom 30% P/B by date)")

# Create valuation target using RELATIVE percentiles (not absolute thresholds)
# This adapts to market regimes (bull/bear) and finds "cheap" relative to history

train_df['is_undervalued'] = 0

# Method 1: Low P/B relative to cross-section at that date
if 'priceToBookRatio' in train_df.columns:
    # Group by date and find bottom 30% P/B at each date
    train_df['pb_rank'] = train_df.groupby('date')['priceToBookRatio'].rank(pct=True)
    train_df.loc[train_df['pb_rank'] < 0.30, 'is_undervalued'] = 1
    print(f"    Using P/B percentile (bottom 30% at each date)")

# Method 2 (optional): High FCF yield relative to cross-section
if 'freeCashFlowYield' in train_df.columns:
    train_df['fcf_rank'] = train_df.groupby('date')['freeCashFlowYield'].rank(pct=True)
    # Top 30% FCF yield also counts as undervalued
    train_df.loc[train_df['fcf_rank'] > 0.70, 'is_undervalued'] = 1
    print(f"    + FCF yield percentile (top 30% at each date)")

# Method 3: P/B low relative to own ROE
if 'priceToBookRatio' in train_df.columns and 'returnOnEquity' in train_df.columns:
    # P/B / ROE ratio (lower = more attractive)
    train_df['pb_roe_ratio'] = train_df['priceToBookRatio'] / (train_df['returnOnEquity'] + 0.01)
    train_df['pb_roe_rank'] = train_df.groupby('date')['pb_roe_ratio'].rank(pct=True)
    train_df.loc[train_df['pb_roe_rank'] < 0.30, 'is_undervalued'] = 1
    print(f"    + P/B vs ROE ratio (bottom 30%)")

print(f"  Undervalued samples: {train_df['is_undervalued'].sum()} / {len(train_df)} ({train_df['is_undervalued'].mean():.1%})")

valuation_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    scale_pos_weight=(len(train_df) - train_df['is_undervalued'].sum()) / max(train_df['is_undervalued'].sum(), 1)
)
valuation_model.fit(X_train_valuation, train_df['is_undervalued'])
joblib.dump(valuation_model, output_dir / "valuation_model.pkl")
print("  [OK] Guardado: models/valuation_model.pkl")

# ===========================
# MOMENTUM MODEL (unchanged, already correct)
# ===========================
print("\n[5/6] Entrenando MOMENTUM MODEL...")
print("  Target: Positive vs negative return")

momentum_model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
momentum_model.fit(X_train_momentum, (y_train > 0).astype(int))
joblib.dump(momentum_model, output_dir / "momentum_model.pkl")
print("  [OK] Guardado: models/momentum_model.pkl")

# ===========================
# SAVE FEATURE LISTS
# ===========================
print("\n[6/6] Guardando feature lists...")
feature_config = {
    'quality_features': quality_features,
    'valuation_features': valuation_features,
    'momentum_features': momentum_features
}
joblib.dump(feature_config, output_dir / "feature_config.pkl")
print("  [OK] Guardado: models/feature_config.pkl")

print("\n" + "=" * 60)
print("[COMPLETADO] ENTRENAMIENTO CORREGIDO")
print("=" * 60)
print(f"\nModelos guardados en: {output_dir}")
print("\nCambios clave:")
print("  1. Quality: Solo profitability, labels por ROIC (no leakage)")
print("  2. Valuation: Predice undervalued (P/B vs ROE), no returns")
print("  3. Momentum: Solo technicals (correcto)")
print("\nProximo: Combinar los 3 para identificar undervalued quality + momentum")
