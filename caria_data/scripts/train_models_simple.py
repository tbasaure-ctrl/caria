"""Script simple para entrenar modelos sin Prefect."""

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
print("ENTRENAMIENTO DE MODELOS CARIA")
print("=" * 60)

# Load gold data
print("\n[1/5] Cargando datos...")
train_df = pd.read_parquet(BASE_DIR / "data/gold/train.parquet")
val_df = pd.read_parquet(BASE_DIR / "data/gold/val.parquet")

print(f"  Train: {len(train_df)} rows")
print(f"  Val: {len(val_df)} rows")

# Extract features
print("\n[2/5] Extrayendo features...")
feature_cols = [col for col in train_df.columns if col not in [
    'ticker', 'date', 'target', 'target_return_20d', 'target_drawdown_prob',
    'features', 'feature_columns', 'period'
]]
print(f"  Features: {len(feature_cols)}")

X_train = train_df[feature_cols].fillna(0)
y_train = train_df['target']

X_val = val_df[feature_cols].fillna(0)
y_val = val_df['target']

# Train Quality Model (predice si es outlier)
print("\n[3/5] Entrenando Quality Model...")
# Create quality labels (top 20% performers)
train_df['is_quality'] = (train_df.groupby('date')['target'].rank(pct=True) > 0.8).astype(int)

quality_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
quality_model.fit(X_train, train_df['is_quality'])

# Save
output_dir = BASE_DIR / "models"
output_dir.mkdir(exist_ok=True)
joblib.dump(quality_model, output_dir / "quality_model.pkl")
print("  [OK] Guardado: models/quality_model.pkl")

# Train Valuation Model (predice si está barata)
print("\n[4/5] Entrenando Valuation Model...")
# Valuation = future return (target ya existe)
val_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
val_model.fit(X_train, y_train)
joblib.dump(val_model, output_dir / "valuation_model.pkl")
print("  [OK] Guardado: models/valuation_model.pkl")

# Train Momentum Model (basado en technical features)
print("\n[5/5] Entrenando Momentum Model...")
momentum_features = [col for col in feature_cols if any(x in col.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'volume'])]
if momentum_features:
    X_train_mom = X_train[momentum_features]
    X_val_mom = X_val[momentum_features]

    momentum_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    # Momentum = positive vs negative return
    momentum_model.fit(X_train_mom, (y_train > 0).astype(int))
    joblib.dump(momentum_model, output_dir / "momentum_model.pkl")
    print("  [OK] Guardado: models/momentum_model.pkl")
else:
    print("  [WARNING] No momentum features found, skipping")

print("\n" + "=" * 60)
print("[COMPLETADO] ENTRENAMIENTO EXITOSO")
print("=" * 60)
print(f"\nModelos guardados en: {output_dir}")
print("\nPróximo paso: Crear API con FastAPI")
