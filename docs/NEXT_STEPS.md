# PRÓXIMOS PASOS CRÍTICOS - Caria ML Project

**Fecha**: 2025-11-09
**Estado**: Pipeline funcional, modelo limitado por features

---

## ✅ LO QUE SE LOGRÓ HOY

### 1. Pipeline de Datos Corregido
- ✅ Eliminados joins problem\u00e1ticos (FX/commodities/sentiment)
- ✅ Splits temporales actualizados (train 1985-2019, test 2023-2024)
- ✅ Dataset ampliado: 754 → 1,139 filas (+51%)
- ✅ 11 tickers funcionando (de 11 disponibles)
- ✅ 0% features nulos (antes: 98%)

### 2. Target Realista
- ✅ Cambiado de 20 quarters (5 años) → 4 quarters (1 año)
- ✅ Target mean: 25.6% anual (razonable)
- ✅ Target std: 77% (alta volatilidad esperada)

### 3. Vector Store Configurado
- ✅ `configs/base.yaml` tiene sección `vector_store`
- ✅ URI construida desde credenciales .env
- ⚠️ Requiere Postgres + pgvector instalado

### 4. Modelo Entrenado
- ✅ Version 18 completada (best: epoch 1, val_loss=0.165)
- ✅ Val_loss 10x mejor que version 17
- ❌ Val_r2 sigue negativo (-1.0)

---

## 🔴 PROBLEMA CRÍTICO: Features Sin Poder Predictivo

**Diagnóstico:**
El modelo no puede predecir returns a 1 año mejor que la media simple (val_r2=-1.0).

**Causas Raíz:**
1. **Features trimestrales** no capturan dinámica de precios intra-trimestre
2. **Falta momentum/technical features** de alta frecuencia
3. **Sin features macro** (tasas, inflación, sentiment)
4. **Dataset pequeño**: 930 filas de train es insuficiente
5. **Target ruidoso**: Returns a 1 año son inherentemente difíciles de predecir

---

## 🎯 ROADMAP RECOMENDADO

### 🚨 CRÍTICO (Próxima semana)

#### 1. Expandir Universe de Tickers (50-100)
**Por qué:** 930 filas → 3,000+ filas para reducir overfitting

**Acción:**
```python
# Agregar S&P 500 top 50 tickers
TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "AVGO", "AMD", "ORCL",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT",
    # Consumer
    "AMZN", "COST", "WMT", "HD", "MCD", "NKE", "SBUX",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Industrial
    "CAT", "BA", "GE", "UNP",
    # ... hasta 50-100
]
```

**Comando:**
```bash
# Modificar configs/pipelines/ingestion.yaml
# Agregar tickers a la lista
# Ejecutar ingesta completa
poetry run python scripts/orchestration/run_ingestion.py
```

#### 2. Agregar Technical Features de Alta Frecuencia
**Por qué:** Capturar momentum y volatility intra-trimestre

**Features a agregar:**
- Momentum: RSI, MACD, Bollinger Bands, Stochastic
- Volatility: ATR, Historical Volatility, VIX correlation
- Volume: OBV, Volume Price Trend, Chaikin Money Flow
- Market: Sector relative strength, Market beta

**Acción:**
Modificar `src/caria/feature_engineering/technical_indicators.py`

#### 3. Agregar Macro Features
**Por qué:** Returns correlacionan fuertemente con ciclo macro

**Features críticas:**
- Tasas de interés (Fed Funds Rate, 10Y Treasury)
- Inflación (CPI, PCE)
- Spreads de crédito (IG, HY)
- Dólar Index (DXY)
- VIX
- Economic surprise indexes

**Fuentes:**
- FRED API (ya configurado en .env)
- Yahoo Finance (gratis)

#### 4. Cambiar Arquitectura: Ensemble Model
**Por qué:** Neural Net solo no es óptimo para tabular data

**Propuesta:**
```python
# Crear ensemble de 3 modelos
1. XGBoost (mejor para tabular features)
2. LSTM (para secuencias temporales)
3. Transformer (para attention sobre quarters)

# Meta-learner: promedio ponderado
final_pred = 0.5*xgb + 0.3*lstm + 0.2*transformer
```

**Resultado esperado:** val_r2 > 0.1

---

### ⚙️ IMPORTANTE (2-4 semanas)

#### 5. Configurar Postgres + pgvector para RAG
**Por qué:** Desbloquear ingesta de wisdom corpus (29MB de textos)

**Instrucciones Windows:**

```powershell
# 1. Descargar Postgres 16 con pgvector
# https://www.postgresql.org/download/windows/
# O usar Docker:
docker run -d \
  --name caria-postgres \
  -e POSTGRES_PASSWORD=Theolucas7 \
  -e POSTGRES_USER=caria_user \
  -e POSTGRES_DB=caria \
  -p 5432:5432 \
  ankane/pgvector

# 2. Conectar y crear schema
psql -U caria_user -d caria -c "CREATE EXTENSION vector;"
psql -U caria_user -d caria -c "CREATE SCHEMA rag;"

# 3. Ejecutar wisdom pipeline
cd C:/key/wise_adviser_cursor_context/notebooks
poetry run python scripts/orchestration/run_wisdom_pipeline.py --version 2025-11-08

# 4. Verificar ingesta
psql -U caria_user -d caria -c "SELECT COUNT(*) FROM rag.embeddings;"
```

#### 6. Implementar Multi-Target Prediction
**Por qué:** Predecir múltiples horizontes mejora robustez

**Targets a predecir:**
- `target_return_1q` (1 quarter = 3 meses)
- `target_return_2q` (6 meses)
- `target_return_4q` (1 año) ← actual
- `target_regime` (clasificación: bull/bear/crash)

**Beneficio:** Modelo aprende patrones a diferentes escalas

#### 7. Feature Engineering Avanzado
**Lag features:**
```python
# Crear ventanas temporales
for lag in [1, 2, 4, 8]:
    df[f'roic_lag{lag}q'] = df.groupby('ticker')['roic'].shift(lag)
    df[f'revenue_growth_lag{lag}q'] = df.groupby('ticker')['revenueGrowth'].shift(lag)
```

**Rolling aggregations:**
```python
# Promedios móviles de fundamentals
for window in [4, 8, 12]:  # 1, 2, 3 años
    df[f'avg_roic_{window}q'] = df.groupby('ticker')['roic'].rolling(window).mean()
```

**Cross-sectional features:**
```python
# Rank relativo vs universo
df['roic_percentile'] = df.groupby('date')['roic'].rank(pct=True)
df['valuation_vs_sector'] = df['priceToBookRatio'] / df.groupby(['date', 'sector'])['priceToBookRatio'].transform('mean')
```

---

### 📚 NICE-TO-HAVE (1-2 meses)

#### 8. Sentiment from News/Earnings Calls
- Scrape earnings call transcripts
- NLP sentiment scores con FinBERT
- Topic modeling (optimism, risk, growth, etc.)

#### 9. Alternative Data
- Web traffic (SimilarWeb API)
- App downloads (Apptopia)
- Satellite imagery (parking lots, shipping)

#### 10. Backtesting Framework
- Walk-forward validation
- Transaction costs
- Portfolio construction
- Sharpe, max drawdown, turnover

---

## 📂 ARCHIVOS CLAVE

### Configs
- `configs/base.yaml` → Vector store configurado
- `configs/pipelines/gold_builder.yaml` → Splits + datasets

### Pipelines
- `src/caria/pipelines/gold_builder_pipeline.py` → **Target cambiado a 4q**
- `src/caria/pipelines/wisdom_pipeline.py` → RAG ingestion (bloqueado por Postgres)

### Data
- `data/gold/{train,val,test}.parquet` → Dataset regenerado (1,139 rows)
- `data/raw/wisdom/2025-11-08/` → 29MB corpus de libros de inversión

### Models
- `lightning_logs/caria/version_18/` → Best checkpoint: epoch 1, val_loss=0.165
- Eval command:
  ```bash
  poetry run python scripts/orchestration/run_evaluate.py \
    --checkpoint lightning_logs/caria/version_18/checkpoints/epoch=01-val_loss=0.1648.ckpt
  ```

---

## 🧪 EXPERIMENTOS A PROBAR

### A. Cambiar Target a Clasificación
En vez de regresión (predecir return exacto), clasificar:
- **Clase 0**: Return < -10% (avoid)
- **Clase 1**: Return entre -10% y +20% (neutral)
- **Clase 2**: Return > +20% (buy)

**Ventaja:** Más fácil de aprender, más útil para portfolio construction

### B. Predecir Ranking en vez de Absoluto
En vez de predecir `return = 25%`, predecir:
- `percentile_rank = 0.85` (top 15% del universo)

**Ventaja:** Normaliza por régimen de mercado

### C. Transfer Learning
Pre-entrenar en S&P 500, fine-tune en tickers seleccionados

---

## 💡 INSIGHTS CLAVE

### Por Qué val_r2 es Negativo
1. **Fundamentals son lagging indicators**
   - ROE/ROIC del Q1 2024 NO predice precio en Q1 2025
   - Precio ya incorpora expectativas

2. **Falta Forward-Looking Data**
   - Necesitas: consensus estimates, guidance, analyst sentiment
   - APIs de pago: Bloomberg, FactSet, S&P Capital IQ

3. **Dataset Demasiado Pequeño**
   - 930 filas x 40 features = 37,200 parámetros
   - Modelo tiene 281K parámetros → 7.5x overparameterized

### Solución Realista
**Opción 1:** Pivot a **Portfolio Optimization** en vez de Return Prediction
- Input: Features actuales
- Output: Stock ranking/scoring
- Objetivo: Construir portfolio con Sharpe > S&P 500

**Opción 2:** **Factor Investing Approach**
- Extraer factors (value, quality, momentum, size)
- Predecir factor exposures
- Combinar con Fama-French framework

---

## 📞 RECURSOS

- **Checkpoint actual**: `lightning_logs/caria/version_18/`
- **Config repo**: `configs/pipelines/gold_builder.yaml`
- **Data audit**: `docs/session_progress_report.md`
- **Wisdom corpus**: `data/raw/wisdom/2025-11-08/` (29MB, 35 libros)

---

## ✉️ CONTACTO PARA DUDAS

Si tienes preguntas sobre:
- **Data pipeline**: Revisar `src/caria/pipelines/gold_builder_pipeline.py:66-68` (target computation)
- **Model architecture**: `src/caria/models/financial_forecaster.py`
- **RAG setup**: Revisar sección "Configurar Postgres + pgvector" arriba
- **Feature engineering**: Crear issue en repo o revisar `src/caria/feature_engineering/`

---

**Última actualización**: 2025-11-09 16:35 UTC
**Próxima acción recomendada**: Expandir universe a 50 tickers + agregar momentum features
