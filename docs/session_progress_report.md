# Sesión de Debugging y Mejoras - Caria ML Pipeline

**Fecha**: 2025-11-09
**Objetivo**: Resolver test_r2=-1.0 y desbloquear ingesta RAG con pgvector

---

## 🔍 PROBLEMAS IDENTIFICADOS

### 1. Dataset Estrecho y Degradado
**Antes:**
- Solo 6 tickers en gold (de 11 disponibles en silver)
- 744 filas en train
- Fecha máxima: 2005-12-31 ❌
- 98% features nulos por joins fallidos

**Causa Raíz:**
- Splits configurados para train hasta 2005
- Joins problemáticos: FX usa `["date", "pair"]`, commodities usa solo `["date"]`
- Sentiment data con `ticker=None`

### 2. Configuración de Vector Store Faltante
- `vector_store.connection` no definido en base.yaml
- Wisdom pipeline bloqueado por falta de conexión pgvector

### 3. Target Demasiado Largo
- `target_return_20d` usa `periods=20` (quarters)
- 20 quarters = 5 años hacia adelante
- Imposible de predecir con features trimestrales

---

## ✅ CORRECCIONES IMPLEMENTADAS

### 1. Configuración de Vector Store (configs/base.yaml)
```yaml
vector_store:
  connection: postgresql://${database.user}:${database.password}@${database.host}:${database.port}/${database.db}
  embedding_table: ${oc.env:PGVECTOR_TABLE,embeddings}
  schema: ${database.schema}
```

### 2. Splits Temporales Corregidos (configs/pipelines/gold_builder.yaml)
**Antes:**
```yaml
splits:
  train: ["1950-01-01", "2005-12-31"]  # ❌ Muy antiguo
  val: ["2006-01-01", "2014-12-31"]
  test: ["2015-01-01", "2025-12-31"]
```

**Después:**
```yaml
splits:
  train: ["1985-01-01", "2015-12-31"]  # ✅ 30 años recientes
  val: ["2016-01-01", "2018-12-31"]    # ✅ 3 años
  test: ["2019-01-01", "2020-12-31"]   # ✅ 2 años
```

### 3. Joins Limpios
**Removidos temporalmente:**
- `commodities/commodities.parquet` (sin ticker causa duplicados)
- `fx/fx_rates.parquet` (pair no existe en stocks)
- `news_sentiment/daily_scores.parquet` (ticker=None)

**Mantenidos:**
- fundamentals (quality + value)
- technicals (momentum + risk)
- events (regimes)

---

## 📊 RESULTADOS POST-CORRECCIÓN

### Gold Layer Regenerado
| Métrica | Antes | Después | Delta |
|---------|-------|---------|-------|
| **Train rows** | 744 | 754 | +1.3% |
| **Val rows** | 0 | 132 | ✅ |
| **Test rows** | 0 | 77 | ✅ |
| **Tickers** | 6 | 11 | +83% |
| **Features nulos** | 98% | 0% | ✅ |
| **Fecha max** | 2005 | 2020 | +15 años |

### Tickers Recuperados
- ✅ AAPL, AMZN, COST, GOOGL, LLY, MSFT (ya existían)
- ✅ AVGO, GM, MA, NVDA, V (recuperados)

---

## ⚠️ PROBLEMA ACTUAL: Modelo Divergente

### Training Metrics (version_17)
```
Epoch 0: val_loss=2.08, val_r2=-0.36
Epoch 1: val_loss=2.18, val_r2=-0.43
Epoch 2: val_loss=2.35, val_r2=-0.54
Epoch 3: val_loss=2.56, val_r2=-0.67
Epoch 4: val_loss=2.74, val_r2=-0.79
Epoch 5: val_loss=2.94, val_r2=-0.92
Epoch 6: val_loss=3.05, val_r2=-1.00  ⬇️ EMPEORANDO
```

**Interpretación:**
- R² negativo significa que el modelo es **peor que predecir la media**
- El modelo está aprendiendo train pero fallando en val/test
- Las features actuales **no pueden predecir** returns a 5 años

---

## 🎯 PRÓXIMOS PASOS RECOMENDADOS

### CRÍTICO: Cambiar Horizonte de Predicción
```python
# En gold_builder_pipeline.py línea 66-68
# ACTUAL (5 años):
df["target_return_20d"] = (
    df.groupby("ticker")["close"].pct_change(periods=20).shift(-20)
)

# PROPUESTO (1 año):
df["target_return_4q"] = (
    df.groupby("ticker")["close"].pct_change(periods=4).shift(-4)
)
```

**Beneficios:**
- Horizonte de 1 año es más realista
- Más datos disponibles para train/val/test
- Features fundamentales tienen más poder predictivo a 1 año

### 1. Reconfigurar Target (URGENTE)
- Cambiar de 20 quarters → 4 quarters (1 año)
- Regenerar gold con nuevo target
- Reentrenar modelo

### 2. Expandir Features
**Agregar back (con fix):**
- Commodities: Hacer broadcast correcto por fecha
- FX: Crear features específicas (USD index, etc.)
- Sentiment: Limpiar datos upstream

**Nuevas fuentes:**
- Macro indicators más granulares (inflation, rates)
- Sector/industry features
- Relative valuation vs peers

### 3. Configurar Postgres/pgvector
```bash
# 1. Instalar Postgres con pgvector
# 2. Crear base de datos
psql -U postgres -c "CREATE DATABASE caria;"
psql -U caria_user -d caria -c "CREATE EXTENSION vector;"

# 3. Inicializar schema
psql -U caria_user -d caria -c "CREATE SCHEMA rag;"

# 4. Ejecutar wisdom pipeline
poetry run python scripts/orchestration/run_wisdom_pipeline.py --version 2025-11-08
```

### 4. Ampliar Universo de Tickers
**Actual:** 11 tickers
**Target:** 50-100 tickers

**Estrategia:**
- S&P 500 top 50 por market cap
- Diversificar sectores (tech, finance, healthcare, consumer, energy)
- Verificar calidad de fundamentals data

---

## 📝 ARCHIVOS MODIFICADOS

1. ✅ `configs/base.yaml` - Agregada sección vector_store
2. ✅ `configs/pipelines/gold_builder.yaml` - Splits corregidos, joins limpiados
3. ✅ `data/gold/*.parquet` - Regenerados con 11 tickers

---

## 🚧 BLOQUEANTES ACTUALES

### 1. Postgres/pgvector No Instalado
- **Impacto**: No se puede ejecutar wisdom pipeline
- **Solución**: Instalar Postgres + pgvector extension
- **Comando**: Ver sección "Configurar Postgres/pgvector"

### 2. Target Demasiado Largo
- **Impacto**: Modelo no puede aprender (val_r2 negativo)
- **Solución**: Cambiar a target de 1 año (4 quarters)
- **Archivo**: `src/caria/pipelines/gold_builder_pipeline.py:66-68`

### 3. Features Limitadas
- **Impacto**: Solo fundamentals + technicals
- **Solución**: Re-agregar commodities/FX con broadcast correcto

---

## 💡 RECOMENDACIONES ARQUITECTÓNICAS

### Corto Plazo (Esta Semana)
1. ✅ Cambiar target a 4 quarters
2. ✅ Instalar Postgres + pgvector
3. ✅ Ejecutar wisdom pipeline
4. ✅ Reentrenar modelo con target corto

### Mediano Plazo (2-4 Semanas)
1. Implementar broadcast correcto para market-level features
2. Expandir a 50+ tickers
3. Agregar macro features (tasas, inflación)
4. Implementar ensemble (XGBoost + Neural Net)

### Largo Plazo (1-2 Meses)
1. Multi-horizon prediction (1q, 4q, 8q)
2. Attention mechanism para wisdom retrieval
3. Regime-aware training
4. Backtesting pipeline automatizado

---

## 🔗 RECURSOS

- **Model checkpoint**: `lightning_logs/caria/version_17/`
- **Config files**: `configs/pipelines/gold_builder.yaml`
- **Data**: `data/gold/{train,val,test}.parquet`
- **Wisdom corpus**: `data/raw/wisdom/2025-11-08/` (29MB unified corpus)

---

## ✉️ CONTACTO / PREGUNTAS

Si tienes dudas sobre:
- **Data pipeline**: Revisar `src/caria/pipelines/gold_builder_pipeline.py`
- **Model training**: Revisar `src/caria/pipelines/training_pipeline.py`
- **Wisdom RAG**: Revisar `src/caria/pipelines/wisdom_pipeline.py`
