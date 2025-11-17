"""Evaluación rápida de los 3 modelos simples sobre test.parquet"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]

print("=" * 60)
print("EVALUACION MODELOS SIMPLES")
print("=" * 60)

# Load test data
print("\n[1/4] Cargando test data...")
test_df = pd.read_parquet(BASE_DIR / "data/gold/test.parquet")
print(f"  Test: {len(test_df)} rows")

# Extract features
feature_cols = [col for col in test_df.columns if col not in [
    'ticker', 'date', 'target', 'target_return_20d', 'target_drawdown_prob',
    'features', 'feature_columns', 'period'
]]

X_test = test_df[feature_cols].fillna(0)
y_test = test_df['target']

# Load models
print("\n[2/4] Cargando modelos...")
quality_model = joblib.load(BASE_DIR / "models/quality_model.pkl")
valuation_model = joblib.load(BASE_DIR / "models/valuation_model.pkl")
momentum_model = joblib.load(BASE_DIR / "models/momentum_model.pkl")

# Evaluate Valuation Model (Regressor)
print("\n[3/4] Evaluando VALUATION MODEL (Regressor)...")
y_pred_val = valuation_model.predict(X_test)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred_val))
mae_val = mean_absolute_error(y_test, y_pred_val)
r2_val = r2_score(y_test, y_pred_val)

print(f"  RMSE: {rmse_val:.4f}")
print(f"  MAE:  {mae_val:.4f}")
print(f"  R2:   {r2_val:.4f}")

# Baseline: predict mean
baseline_pred = np.full_like(y_test, y_test.mean())
rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_pred))
mae_baseline = mean_absolute_error(y_test, baseline_pred)
r2_baseline = r2_score(y_test, baseline_pred)

print(f"\n  BASELINE (mean):")
print(f"  RMSE: {rmse_baseline:.4f}")
print(f"  MAE:  {mae_baseline:.4f}")
print(f"  R2:   {r2_baseline:.4f}")

improvement = (rmse_baseline - rmse_val) / rmse_baseline * 100
print(f"\n  Mejora vs baseline: {improvement:.1f}%")

# Evaluate Quality Model (Classifier)
print("\n[4/4] Evaluando QUALITY MODEL (Classifier)...")
# Recreate quality labels on test (WITH LEAKAGE WARNING)
test_df['is_quality'] = (test_df.groupby('date')['target'].rank(pct=True) > 0.8).astype(int)
y_quality_test = test_df['is_quality']

y_pred_quality = quality_model.predict(X_test)
y_pred_quality_proba = quality_model.predict_proba(X_test)[:, 1]

acc_quality = accuracy_score(y_quality_test, y_pred_quality)
auc_quality = roc_auc_score(y_quality_test, y_pred_quality_proba)

print(f"  Accuracy: {acc_quality:.3f}")
print(f"  AUC-ROC:  {auc_quality:.3f}")
print(f"  [WARNING] Leakage: labels usan target futuro")

cm = confusion_matrix(y_quality_test, y_pred_quality)
print(f"\n  Confusion Matrix:")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# Evaluate Momentum Model (Classifier)
print("\n[5/4] Evaluando MOMENTUM MODEL (Classifier)...")
momentum_features = [col for col in feature_cols if any(x in col.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'volume'])]
print(f"  Momentum features encontrados: {len(momentum_features)}")

if momentum_features:
    X_test_mom = X_test[momentum_features]
    y_momentum_test = (y_test > 0).astype(int)

    y_pred_momentum = momentum_model.predict(X_test_mom)
    y_pred_momentum_proba = momentum_model.predict_proba(X_test_mom)[:, 1]

    acc_momentum = accuracy_score(y_momentum_test, y_pred_momentum)
    auc_momentum = roc_auc_score(y_momentum_test, y_pred_momentum_proba)

    print(f"  Accuracy: {acc_momentum:.3f}")
    print(f"  AUC-ROC:  {auc_momentum:.3f}")

    cm_mom = confusion_matrix(y_momentum_test, y_pred_momentum)
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm_mom[0,0]}, FP={cm_mom[0,1]}")
    print(f"  FN={cm_mom[1,0]}, TP={cm_mom[1,1]}")
else:
    print("  [ERROR] No momentum features found!")

print("\n" + "=" * 60)
print("[COMPLETADO] EVALUACION")
print("=" * 60)
