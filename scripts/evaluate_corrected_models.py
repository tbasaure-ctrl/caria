"""Evaluate corrected models and create ensemble scoring"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parents[1]

print("=" * 60)
print("EVALUACION MODELOS CORREGIDOS")
print("=" * 60)

# Load test data
print("\n[1/5] Cargando test data...")
test_df = pd.read_parquet(BASE_DIR / "data/gold/test.parquet")
print(f"  Test: {len(test_df)} rows")

# Load models and feature config
quality_model = joblib.load(BASE_DIR / "models/quality_model.pkl")
valuation_model = joblib.load(BASE_DIR / "models/valuation_model.pkl")
momentum_model = joblib.load(BASE_DIR / "models/momentum_model.pkl")
feature_config = joblib.load(BASE_DIR / "models/feature_config.pkl")

quality_features = feature_config['quality_features']
valuation_features = feature_config['valuation_features']
momentum_features = feature_config['momentum_features']

# Prepare test data
X_test_quality = test_df[quality_features].fillna(0)
X_test_valuation = test_df[valuation_features].fillna(0)
X_test_momentum = test_df[momentum_features].fillna(0)

y_test = test_df['target']

# ===========================
# EVALUATE QUALITY MODEL
# ===========================
print("\n[2/5] Evaluando QUALITY MODEL...")
# Use same relative percentile logic as training
test_df['roic_percentile'] = test_df.groupby('date')['roic'].rank(pct=True)
test_df['is_quality'] = (test_df['roic_percentile'] > 0.80).astype(int)

if test_df['is_quality'].sum() < 100:
    test_df['roe_percentile'] = test_df.groupby('date')['returnOnEquity'].rank(pct=True)
    test_df.loc[test_df['roe_percentile'] > 0.80, 'is_quality'] = 1

y_quality_test = test_df['is_quality']
y_pred_quality = quality_model.predict(X_test_quality)
y_pred_quality_proba = quality_model.predict_proba(X_test_quality)[:, 1]

acc_quality = accuracy_score(y_quality_test, y_pred_quality)
auc_quality = roc_auc_score(y_quality_test, y_pred_quality_proba)

print(f"  Accuracy: {acc_quality:.3f}")
print(f"  AUC-ROC:  {auc_quality:.3f}")
print(f"  Labels: High ROIC (no leakage)")

cm = confusion_matrix(y_quality_test, y_pred_quality)
print(f"\n  Confusion Matrix:")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")

# ===========================
# EVALUATE VALUATION MODEL
# ===========================
print("\n[3/5] Evaluando VALUATION MODEL...")
test_df['is_undervalued'] = 0

# Use same relative percentile logic as training
if 'priceToBookRatio' in test_df.columns:
    test_df['pb_rank'] = test_df.groupby('date')['priceToBookRatio'].rank(pct=True)
    test_df.loc[test_df['pb_rank'] < 0.30, 'is_undervalued'] = 1

if 'freeCashFlowYield' in test_df.columns:
    test_df['fcf_rank'] = test_df.groupby('date')['freeCashFlowYield'].rank(pct=True)
    test_df.loc[test_df['fcf_rank'] > 0.70, 'is_undervalued'] = 1

if 'priceToBookRatio' in test_df.columns and 'returnOnEquity' in test_df.columns:
    test_df['pb_roe_ratio'] = test_df['priceToBookRatio'] / (test_df['returnOnEquity'] + 0.01)
    test_df['pb_roe_rank'] = test_df.groupby('date')['pb_roe_ratio'].rank(pct=True)
    test_df.loc[test_df['pb_roe_rank'] < 0.30, 'is_undervalued'] = 1

y_valuation_test = test_df['is_undervalued']
print(f"  Undervalued in test: {y_valuation_test.sum()} / {len(test_df)} ({y_valuation_test.mean():.1%})")

if y_valuation_test.sum() > 10:
    y_pred_valuation = valuation_model.predict(X_test_valuation)
    y_pred_valuation_proba = valuation_model.predict_proba(X_test_valuation)[:, 1]

    acc_valuation = accuracy_score(y_valuation_test, y_pred_valuation)
    auc_valuation = roc_auc_score(y_valuation_test, y_pred_valuation_proba)

    print(f"  Accuracy: {acc_valuation:.3f}")
    print(f"  AUC-ROC:  {auc_valuation:.3f}")

    cm_val = confusion_matrix(y_valuation_test, y_pred_valuation)
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm_val[0,0]}, FP={cm_val[0,1]}")
    print(f"  FN={cm_val[1,0]}, TP={cm_val[1,1]}")

    precision_val = cm_val[1,1] / (cm_val[1,1] + cm_val[0,1]) if (cm_val[1,1] + cm_val[0,1]) > 0 else 0
    recall_val = cm_val[1,1] / (cm_val[1,1] + cm_val[1,0]) if (cm_val[1,1] + cm_val[1,0]) > 0 else 0
    print(f"  Precision: {precision_val:.3f}, Recall: {recall_val:.3f}")
else:
    print("  [WARNING] Not enough undervalued samples in test")
    y_pred_valuation_proba = valuation_model.predict(X_test_valuation)  # Regressor fallback

# ===========================
# EVALUATE MOMENTUM MODEL
# ===========================
print("\n[4/5] Evaluando MOMENTUM MODEL...")
y_momentum_test = (y_test > 0).astype(int)
y_pred_momentum = momentum_model.predict(X_test_momentum)
y_pred_momentum_proba = momentum_model.predict_proba(X_test_momentum)[:, 1]

acc_momentum = accuracy_score(y_momentum_test, y_pred_momentum)
auc_momentum = roc_auc_score(y_momentum_test, y_pred_momentum_proba)

print(f"  Accuracy: {acc_momentum:.3f}")
print(f"  AUC-ROC:  {auc_momentum:.3f}")

cm_mom = confusion_matrix(y_momentum_test, y_pred_momentum)
print(f"\n  Confusion Matrix:")
print(f"  TN={cm_mom[0,0]}, FP={cm_mom[0,1]}")
print(f"  FN={cm_mom[1,0]}, TP={cm_mom[1,1]}")

# ===========================
# ENSEMBLE: Quality + Undervalued + Momentum
# ===========================
print("\n[5/5] ENSEMBLE: Combinando los 3 scores...")

test_df['quality_score'] = y_pred_quality_proba
test_df['valuation_score'] = y_pred_valuation_proba if isinstance(y_pred_valuation_proba, np.ndarray) else 0
test_df['momentum_score'] = y_pred_momentum_proba

# Composite score: prioritize valuation (best AUC ~0.92), then quality, minimal momentum
test_df['composite_score'] = (
    test_df['valuation_score'] * 0.5 +  # Valuation tiene mejor AUC y mejora con más data
    test_df['quality_score'] * 0.4 +
    test_df['momentum_score'] * 0.1     # Momentum AUC ~0.5 = random
)

# Identify "best opportunities": high quality + undervalued + positive momentum
test_df['is_opportunity'] = (
    (test_df['quality_score'] > 0.7) &
    (test_df['valuation_score'] > 0.5) &
    (test_df['momentum_score'] > 0.5)
).astype(int)

opportunities = test_df[test_df['is_opportunity'] == 1]
print(f"\n  Opportunities identified: {len(opportunities)} / {len(test_df)} ({len(opportunities)/len(test_df):.1%})")

if len(opportunities) > 0:
    avg_return_opportunities = opportunities['target'].mean()
    avg_return_all = test_df['target'].mean()
    print(f"\n  Avg return (opportunities): {avg_return_opportunities:.2%}")
    print(f"  Avg return (all): {avg_return_all:.2%}")
    print(f"  Outperformance: {(avg_return_opportunities - avg_return_all):.2%}")

    # Top 10 opportunities by composite score
    print(f"\n  Top 10 opportunities:")
    top10 = test_df.nlargest(10, 'composite_score')[['ticker', 'date', 'quality_score', 'valuation_score', 'momentum_score', 'composite_score', 'target']]
    print(top10.to_string(index=False))

print("\n" + "=" * 60)
print("[COMPLETADO] EVALUACION")
print("=" * 60)
