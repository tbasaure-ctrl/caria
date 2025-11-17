"""Analyze feature importance for each model"""

import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Load models
quality_model = joblib.load(BASE_DIR / "models/quality_model.pkl")
valuation_model = joblib.load(BASE_DIR / "models/valuation_model.pkl")
momentum_model = joblib.load(BASE_DIR / "models/momentum_model.pkl")

# Load train data to get feature names
train = pd.read_parquet(BASE_DIR / "data/gold/train.parquet")

# All feature columns
feature_cols = [col for col in train.columns if col not in [
    'ticker', 'date', 'target', 'target_return_20d', 'target_drawdown_prob',
    'target_regime', 'features', 'feature_columns', 'regime_name', 'period'
]]

# Momentum features
momentum_features = [col for col in feature_cols if any(x in col.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'volume'])]

print("=" * 60)
print("ANALISIS DE FEATURE IMPORTANCE")
print("=" * 60)

# Quality Model
print("\n[1/3] QUALITY MODEL (usa TODAS las features)")
print(f"  Features: {len(feature_cols)}")
quality_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': quality_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Top 10 features:")
for idx, row in quality_importance.head(10).iterrows():
    print(f"    {row['feature']:<30} {row['importance']:.4f}")

# Categorize features
technicals = [f for f in quality_importance['feature'].values if f in ['sma_20', 'sma_50', 'sma_200', 'ema_20', 'ema_50', 'ema_200', 'rsi_14', 'macd', 'macd_signal', 'atr_14', 'volatility_30d', 'drawdown', 'returns_20d', 'returns_60d', 'returns_120d']]
fundamentals = [f for f in quality_importance['feature'].values if f in ['capitalExpenditures', 'freeCashFlowYield', 'grossProfitMargin', 'netIncomeGrowth', 'netProfitMargin', 'net_debt', 'r_and_d', 'returnOnAssets', 'returnOnEquity', 'revenueGrowth', 'roic', 'roiic']]
valuation_feats = [f for f in quality_importance['feature'].values if f in ['enterpriseValue', 'freeCashFlowPerShare', 'priceToBookRatio', 'priceToSalesRatio', 'close']]

print(f"\n  Importancia por categoria:")
print(f"    Technicals: {quality_importance[quality_importance['feature'].isin(technicals)]['importance'].sum():.3f}")
print(f"    Fundamentals: {quality_importance[quality_importance['feature'].isin(fundamentals)]['importance'].sum():.3f}")
print(f"    Valuation: {quality_importance[quality_importance['feature'].isin(valuation_feats)]['importance'].sum():.3f}")

# Valuation Model
print("\n[2/3] VALUATION MODEL (usa TODAS las features)")
print(f"  Features: {len(feature_cols)}")
valuation_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': valuation_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Top 10 features:")
for idx, row in valuation_importance.head(10).iterrows():
    print(f"    {row['feature']:<30} {row['importance']:.4f}")

print(f"\n  Importancia por categoria:")
print(f"    Technicals: {valuation_importance[valuation_importance['feature'].isin(technicals)]['importance'].sum():.3f}")
print(f"    Fundamentals: {valuation_importance[valuation_importance['feature'].isin(fundamentals)]['importance'].sum():.3f}")
print(f"    Valuation: {valuation_importance[valuation_importance['feature'].isin(valuation_feats)]['importance'].sum():.3f}")

# Momentum Model
print("\n[3/3] MOMENTUM MODEL (solo technicals)")
print(f"  Features: {len(momentum_features)}")
print(f"  Features usados: {momentum_features}")

momentum_importance = pd.DataFrame({
    'feature': momentum_features,
    'importance': momentum_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Feature importances:")
for idx, row in momentum_importance.iterrows():
    print(f"    {row['feature']:<30} {row['importance']:.4f}")

print("\n" + "=" * 60)
print("PROBLEMA IDENTIFICADO:")
print("=" * 60)
print("\nQuality Model:")
print("  - Usa target FUTURO para crear labels (leakage)")
print("  - Deberia usar solo fundamentals HISTORICOS")
print("  - Ejemplo: roic > percentile(80), returnOnEquity > 15%, etc")

print("\nValuation Model:")
print("  - Mezcla technicals + fundamentals + valuation")
print("  - Deberia usar SOLO fundamentals + valuation")
print("  - Technicals NO predicen value, predicen momentum")
