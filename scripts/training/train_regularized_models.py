"""Entrenar modelos XGBoost/LightGBM con hiperparámetros anti-overfitting."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

print("=" * 60)
print("ENTRENAMIENTO MODELOS REGULARIZADOS")
print("=" * 60)

# Load data
print("\n[1/7] Cargando datos...")
train_df = pd.read_parquet(BASE_DIR / "data/gold/train.parquet")
val_df = pd.read_parquet(BASE_DIR / "data/gold/val.parquet")

print(f"  Train: {len(train_df)} rows")
print(f"  Val: {len(val_df)} rows")

# Load feature config if exists
feature_config_path = BASE_DIR / "models" / "feature_config.pkl"
if feature_config_path.exists():
    feature_config = joblib.load(feature_config_path)
    quality_features = feature_config.get("quality_features", [])
    valuation_features = feature_config.get("valuation_features", [])
    momentum_features = feature_config.get("momentum_features", [])
else:
    # Fallback feature definitions
    quality_features = [
        "roic",
        "roiic",
        "returnOnEquity",
        "returnOnAssets",
        "grossProfitMargin",
        "netProfitMargin",
        "freeCashFlowYield",
        "freeCashFlowPerShare",
    ]
    valuation_features = [
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
    ]
    momentum_features = [
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_20",
        "ema_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "atr_14",
        "volatility_30d",
        "volume",
        "volume_sma_20",
    ]

# Filter features that exist in data
quality_features = [f for f in quality_features if f in train_df.columns]
valuation_features = [f for f in valuation_features if f in train_df.columns]
momentum_features = [f for f in momentum_features if f in train_df.columns]

print(f"\n  Quality features: {len(quality_features)}")
print(f"  Valuation features: {len(valuation_features)}")
print(f"  Momentum features: {len(momentum_features)}")

# Prepare data
X_train_quality = train_df[quality_features].fillna(0)
X_train_valuation = train_df[valuation_features].fillna(0)
X_train_momentum = train_df[momentum_features].fillna(0)

X_val_quality = val_df[quality_features].fillna(0)
X_val_valuation = val_df[valuation_features].fillna(0)
X_val_momentum = val_df[momentum_features].fillna(0)

y_train = train_df["target"]
y_val = val_df["target"]

# ===========================
# QUALITY MODEL (Classifier)
# ===========================
print("\n[2/7] Entrenando QUALITY MODEL (regularizado)...")

train_df["roic_percentile"] = train_df.groupby("date")["roic"].rank(pct=True)
train_df["is_quality"] = (train_df["roic_percentile"] > 0.80).astype(int)

if train_df["is_quality"].sum() < 100:
    train_df["roe_percentile"] = train_df.groupby("date")["returnOnEquity"].rank(pct=True)
    train_df.loc[train_df["roe_percentile"] > 0.80, "is_quality"] = 1

y_train_quality = train_df["is_quality"]
y_val_quality = val_df.copy()
y_val_quality["roic_percentile"] = val_df.groupby("date")["roic"].rank(pct=True)
y_val_quality["is_quality"] = (y_val_quality["roic_percentile"] > 0.80).astype(int)
y_val_quality = y_val_quality["is_quality"]

scale_pos_weight = (
    (len(train_df) - y_train_quality.sum()) / max(y_train_quality.sum(), 1)
    if y_train_quality.sum() > 0
    else 1.0
)

if HAS_XGBOOST:
    quality_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        min_child_weight=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
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
elif HAS_LIGHTGBM:
    quality_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        min_child_samples=20,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    quality_model.fit(
        X_train_quality,
        y_train_quality,
        eval_set=[(X_val_quality, y_val_quality)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
else:
    raise ImportError("Se requiere XGBoost o LightGBM")

output_dir = BASE_DIR / "models"
output_dir.mkdir(exist_ok=True)
joblib.dump(quality_model, output_dir / "regularized_quality_model.pkl")

# Evaluate
y_pred_quality = quality_model.predict(X_val_quality)
y_pred_quality_proba = quality_model.predict_proba(X_val_quality)[:, 1]
acc = accuracy_score(y_val_quality, y_pred_quality)
auc = roc_auc_score(y_val_quality, y_pred_quality_proba)
print(f"  Val Accuracy: {acc:.4f}, AUC: {auc:.4f}")

# ===========================
# VALUATION MODEL (Classifier)
# ===========================
print("\n[3/7] Entrenando VALUATION MODEL (regularizado)...")

train_df["is_undervalued"] = 0
if "priceToBookRatio" in train_df.columns:
    train_df["pb_rank"] = train_df.groupby("date")["priceToBookRatio"].rank(pct=True)
    train_df.loc[train_df["pb_rank"] < 0.30, "is_undervalued"] = 1

if "freeCashFlowYield" in train_df.columns:
    train_df["fcf_rank"] = train_df.groupby("date")["freeCashFlowYield"].rank(pct=True)
    train_df.loc[train_df["fcf_rank"] > 0.70, "is_undervalued"] = 1

if "priceToBookRatio" in train_df.columns and "returnOnEquity" in train_df.columns:
    train_df["pb_roe_ratio"] = train_df["priceToBookRatio"] / (train_df["returnOnEquity"] + 0.01)
    train_df["pb_roe_rank"] = train_df.groupby("date")["pb_roe_ratio"].rank(pct=True)
    train_df.loc[train_df["pb_roe_rank"] < 0.30, "is_undervalued"] = 1

y_train_valuation = train_df["is_undervalued"]
y_val_valuation = val_df.copy()
y_val_valuation["is_undervalued"] = 0
if "priceToBookRatio" in val_df.columns:
    y_val_valuation["pb_rank"] = val_df.groupby("date")["priceToBookRatio"].rank(pct=True)
    y_val_valuation.loc[y_val_valuation["pb_rank"] < 0.30, "is_undervalued"] = 1
y_val_valuation = y_val_valuation["is_undervalued"]

scale_pos_weight = (
    (len(train_df) - y_train_valuation.sum()) / max(y_train_valuation.sum(), 1)
    if y_train_valuation.sum() > 0
    else 1.0
)

if HAS_XGBOOST:
    valuation_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        min_child_weight=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
    )
    valuation_model.fit(
        X_train_valuation,
        y_train_valuation,
        eval_set=[(X_val_valuation, y_val_valuation)],
        early_stopping_rounds=50,
        verbose=False,
    )
elif HAS_LIGHTGBM:
    valuation_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        min_child_samples=20,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    valuation_model.fit(
        X_train_valuation,
        y_train_valuation,
        eval_set=[(X_val_valuation, y_val_valuation)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

joblib.dump(valuation_model, output_dir / "regularized_valuation_model.pkl")

# Evaluate
y_pred_valuation = valuation_model.predict(X_val_valuation)
y_pred_valuation_proba = valuation_model.predict_proba(X_val_valuation)[:, 1]
acc = accuracy_score(y_val_valuation, y_pred_valuation)
auc = roc_auc_score(y_val_valuation, y_pred_valuation_proba)
print(f"  Val Accuracy: {acc:.4f}, AUC: {auc:.4f}")

# ===========================
# MOMENTUM MODEL (Classifier)
# ===========================
print("\n[4/7] Entrenando MOMENTUM MODEL (regularizado)...")

y_train_momentum = (y_train > 0).astype(int)
y_val_momentum = (y_val > 0).astype(int)

if HAS_XGBOOST:
    momentum_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        min_child_weight=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
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
elif HAS_LIGHTGBM:
    momentum_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        min_child_samples=20,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
    )
    momentum_model.fit(
        X_train_momentum,
        y_train_momentum,
        eval_set=[(X_val_momentum, y_val_momentum)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

joblib.dump(momentum_model, output_dir / "regularized_momentum_model.pkl")

# Evaluate
y_pred_momentum = momentum_model.predict(X_val_momentum)
y_pred_momentum_proba = momentum_model.predict_proba(X_val_momentum)[:, 1]
acc = accuracy_score(y_val_momentum, y_pred_momentum)
auc = roc_auc_score(y_val_momentum, y_pred_momentum_proba)
print(f"  Val Accuracy: {acc:.4f}, AUC: {auc:.4f}")

# ===========================
# CROSS-VALIDATION TEMPORAL
# ===========================
print("\n[5/7] Ejecutando cross-validation temporal...")

# Combine train and val for CV
combined_df = pd.concat([train_df, val_df]).sort_values("date")
X_combined_quality = pd.concat([X_train_quality, X_val_quality])
y_combined_quality = pd.concat([y_train_quality, y_val_quality])

tscv = TimeSeriesSplit(n_splits=3)
cv_scores_quality = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_combined_quality)):
    X_cv_train = X_combined_quality.iloc[train_idx]
    X_cv_val = X_combined_quality.iloc[val_idx]
    y_cv_train = y_combined_quality.iloc[train_idx]
    y_cv_val = y_combined_quality.iloc[val_idx]

    if HAS_XGBOOST:
        cv_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            min_child_weight=5,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )
    else:
        cv_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            min_child_samples=20,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )

    cv_model.fit(X_cv_train, y_cv_train)
    y_cv_pred_proba = cv_model.predict_proba(X_cv_val)[:, 1]
    cv_auc = roc_auc_score(y_cv_val, y_cv_pred_proba)
    cv_scores_quality.append(cv_auc)
    print(f"  Fold {fold + 1} AUC: {cv_auc:.4f}")

print(f"  CV Mean AUC: {np.mean(cv_scores_quality):.4f} (+/- {np.std(cv_scores_quality):.4f})")

# ===========================
# SAVE FEATURE CONFIG
# ===========================
print("\n[6/7] Guardando feature config...")
feature_config = {
    "quality_features": quality_features,
    "valuation_features": valuation_features,
    "momentum_features": momentum_features,
}
joblib.dump(feature_config, output_dir / "feature_config.pkl")

# ===========================
# SUMMARY
# ===========================
print("\n[7/7] Resumen de modelos regularizados:")
print(f"\n  Modelos guardados en: {output_dir}")
print("  - regularized_quality_model.pkl")
print("  - regularized_valuation_model.pkl")
print("  - regularized_momentum_model.pkl")
print("\n  Hiperparámetros anti-overfitting:")
print("    - reg_alpha=1.0, reg_lambda=2.0")
print("    - max_depth=4 (reducido)")
print("    - learning_rate=0.01 (bajo)")
print("    - subsample=0.8, colsample_bytree=0.8")
print("    - early_stopping_rounds=50")
print("\n" + "=" * 60)
print("[COMPLETADO] ENTRENAMIENTO REGULARIZADO")
print("=" * 60)





