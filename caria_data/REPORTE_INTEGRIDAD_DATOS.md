# Reporte de Integridad de Datos - Proyecto Caria
**Fecha**: 2025-11-11
**Estado**: Datos mucho más completos de lo esperado

---

## RESUMEN EJECUTIVO

⚠️ **El diagnóstico original estaba desactualizado**. El dataset ya tiene:
- ✅ **476-498 tickers** (no 11 como indicaba el diagnóstico)
- ✅ **Features técnicos completamente implementados** (RSI, MACD, ATR, volume, etc.)
- ✅ **Features fundamentales completos** (ROIC, ROE, margins, FCF, debt, etc.)
- ✅ **3+ millones de observaciones** (suficiente para reducir overfitting)

### FASES YA COMPLETADAS (sin saberlo)
- ✅ **FASE 2.1**: Expandir universe de tickers (11 → 50-100) → **HECHO: 476-498 tickers**
- ✅ **FASE 2.2**: Implementar features técnicos → **HECHO: RSI, MACD, ATR, SMA, EMA, etc.**

### PENDIENTES CRÍTICOS
- ⚠️ **Features macro NO están en data/gold** (yield curve, VIX, credit spreads, etc.)
- ⚠️ **Regime predictions tienen date range incorrecto** (1919-1968 vs 1990-2019)
- ⚠️ **Warming period con valores 0.0** en primeras observaciones

---

## DATA GOLD: DATASETS PROCESADOS

### 1. Train Dataset (`data/gold/train.parquet`)

**Dimensiones**:
- **Filas**: 2,853,509
- **Columnas**: 37
- **Tickers**: 476 únicos
- **Period**: 1990-01-02 a 2019-12-31 (30 años)

**Features Disponibles** (37 total):

#### a) Identificadores (2)
1. `date` - Fecha de observación
2. `ticker` - Símbolo de la empresa

#### b) Features Técnicos - Tendencia (8)
3. `sma_20` - Simple Moving Average 20 períodos
4. `sma_50` - Simple Moving Average 50 períodos
5. `sma_200` - Simple Moving Average 200 períodos
6. `ema_20` - Exponential Moving Average 20 períodos
7. `ema_50` - Exponential Moving Average 50 períodos
8. `ema_200` - Exponential Moving Average 200 períodos
9. `macd` - MACD indicator
10. `macd_signal` - MACD signal line

#### c) Features Técnicos - Momentum (1)
11. `rsi_14` - Relative Strength Index 14 períodos

#### d) Features Técnicos - Volatilidad (2)
12. `atr_14` - Average True Range 14 períodos
13. `volatility_30d` - Volatilidad histórica 30 días

#### e) Features Técnicos - Volume (4)
14. `volume` - Volumen de trading
15. `volume_sma_20` - Volume SMA 20 períodos
16. `volume_ratio` - Ratio volume actual vs promedio
17. `volume_change` - Cambio de volumen

#### f) Features Fundamentales - Rentabilidad (4)
18. `roic` - Return on Invested Capital
19. `returnOnEquity` - ROE
20. `returnOnAssets` - ROA
21. `grossProfitMargin` - Margen bruto
22. `netProfitMargin` - Margen neto

#### g) Features Fundamentales - Valuación (5)
23. `freeCashFlowPerShare` - FCF por acción
24. `priceToBookRatio` - P/B ratio
25. `priceToSalesRatio` - P/S ratio
26. `freeCashFlowYield` - FCF yield
27. `marketCap` - Capitalización de mercado
28. `enterpriseValue` - Enterprise Value

#### h) Features Fundamentales - Solvencia (3)
29. `totalDebt` - Deuda total
30. `cashAndCashEquivalents` - Efectivo
31. `net_debt` - Deuda neta (calculated)

#### i) Features de Precio/Returns (5)
32. `close` - Precio de cierre
33. `returns_20d` - Retorno 20 días
34. `returns_60d` - Retorno 60 días
35. `returns_120d` - Retorno 120 días
36. `drawdown` - Drawdown actual

#### j) Target (1)
37. `target` - Variable objetivo (probablemente returns futuros)

### 2. Validation Dataset (`data/gold/val.parquet`)
- **Período estimado**: 2020-2022 (no verificado aún)
- **Columnas**: 37 (misma estructura que train)

### 3. Test Dataset (`data/gold/test.parquet`)

**Dimensiones**:
- **Filas**: 231,178
- **Columnas**: 37
- **Tickers**: 498 únicos (más que train - nuevas IPOs)
- **Período**: 2023-01-03 a 2024-11-07 (2 años recientes)

**Observación**: Test tiene 498 tickers vs 476 en train (22 tickers adicionales, probablemente IPOs recientes)

---

## DATA SILVER: DATOS PROCESADOS INTERMEDIOS

### 1. Regime Predictions (`data/silver/regime/hmm_regime_predictions.parquet`)

**Dimensiones**:
- **Filas**: 18,057
- **Columnas**: 7
- **Período**: 1919-01-01 a 1968-06-08 ⚠️ **PROBLEMA CRÍTICO**

**Columns**:
1. `date`
2. `regime` - Régimen detectado (expansion/slowdown/recession/stress)
3. `expansion_prob` - Probabilidad de expansión
4. `slowdown_prob` - Probabilidad de desaceleración
5. `recession_prob` - Probabilidad de recesión
6. `stress_prob` - Probabilidad de estrés
7. `confidence` - Confianza en la clasificación

**⚠️ PROBLEMAS IDENTIFICADOS**:

**P-REGIME-1: Date Range Incorrecto**
- **Problema**: Regime predictions cubren 1919-1968, pero train data es 1990-2019
- **Causa probable**:
  - Modelo entrenado con datos macro históricos (FRED tiene datos desde 1919)
  - No filtrado para coincidir con período de train data
- **Impacto**: No se pueden hacer joins con train/val/test data
- **Solución**: Re-entrenar HMM con período 1990-2024 o extender train data

**P-REGIME-2: Probabilidades Uniformes**
- **Problema**: Primeras observaciones tienen probabilidades ~0.175 para todos los estados
- **Causa probable**:
  - Warming period del HMM
  - Modelo no convergió correctamente
  - Inicialización aleatoria sin suficiente información
- **Solución**:
  - Analizar convergencia del modelo (log-likelihood)
  - Verificar que EM algorithm convergió
  - Aumentar n_iter si es necesario

### 2. Macro Data (`data/silver/macro/`)
- **Estado**: Directorio existe pero no inspeccionado aún
- **Contenido esperado**: FRED data (yield curve, VIX, credit spreads, unemployment, etc.)
- ⚠️ **PENDIENTE VERIFICAR**

---

## MODELOS ENTRENADOS

### 1. Modelo HMM (`models/regime_hmm_model.pkl`)
- ✅ **Existe**: Sí
- ⚠️ **Validación**: Pendiente (verificar convergencia y date range)

### 2. Modelos Legacy (DEPRECATED)
Los siguientes modelos existen pero están marcados como **DEPRECATED** según el diagnóstico:
- `quality_model.pkl`
- `momentum_model.pkl`
- `valuation_model.pkl`
- `improved_*.pkl` (versiones mejoradas)
- `feature_config.pkl`

**Acción recomendada**: Mantener solo para referencia histórica, no usar en producción

---

## PROBLEMAS CRÍTICOS IDENTIFICADOS

### 🔴 P-DATA-1: Mismatch de Período entre Regime y Train
- **Regime predictions**: 1919-1968
- **Train data**: 1990-2019
- **Test data**: 2023-2024
- **Impacto**: Imposible integrar Sistema I (HMM) con Sistema III (Factores) y IV (Valuación)
- **Prioridad**: CRÍTICA
- **Tiempo estimado de fix**: 1 hora (re-entrenar HMM)

### 🟡 P-DATA-2: Warming Period con Valores 0.0
- **Ubicación**: Primeras observaciones de ticker 'A' en 1999-11
- **Problema**: roic=0, returnOnEquity=0, rsi_14=0
- **Causa probable**: Warming period para indicadores técnicos (SMA necesita 200 días)
- **Impacto**: Bias en primeras observaciones de cada ticker
- **Solución**: Filtrar primeras N observaciones por ticker (N=200 días)
- **Prioridad**: MEDIA
- **Tiempo estimado de fix**: 30 minutos

### 🟡 P-DATA-3: Features Macro NO están en Gold Data
- **Problema**: data/gold/* no incluye features macro (yield_curve_slope, vix, sentiment, etc.)
- **Impacto**:
  - Modelo ML no puede usar contexto macro directamente
  - Sistema III (Factores) no puede ajustar por régimen sin joins complejos
- **Solución**:
  - Opción A: Agregar features macro a data/gold mediante join
  - Opción B: Sistema I (HMM) genera features de régimen separadamente, se consumen via API
- **Prioridad**: MEDIA (depende de arquitectura de integración)

### 🟢 P-DATA-4: Test Dataset tiene 22 Tickers Nuevos
- **Problema**: Test (498 tickers) > Train (476 tickers)
- **Causa**: IPOs recientes entre 2020-2023
- **Impacto**: Modelo no tiene historia de estos tickers (cold start problem)
- **Solución**:
  - Opción A: Excluir tickers nuevos de test (no realista)
  - Opción B: Cross-sectional features solo (no usar lags personalizados)
  - Opción C: Feature engineering agnóstico a ticker (percentiles, ranks)
- **Prioridad**: BAJA (modelado cross-sectional maneja esto)

---

## CALIDAD DE DATOS

### ✅ POSITIVO
1. **Sin missing values**: Todas las columnas verificadas tienen 0 nulls
2. **Dimensiones correctas**: 2.8M+ observaciones suficientes para ML
3. **Features completos**: 37 features bien balanceados (técnicos + fundamentales)
4. **Período largo**: 30 años de historia en train (múltiples ciclos económicos)
5. **Universe amplio**: 476-498 tickers (suficiente para cross-sectional)

### ⚠️ POR VERIFICAR
1. **Outliers**: No verificado (necesita EDA completo)
2. **Data leakage**: No verificado (verificar que target no tenga look-ahead bias)
3. **Survival bias**: No verificado (¿incluye empresas que quebraron?)
4. **Splits correctos**: No verificado (purged k-fold implementation)

---

## ACCIONES INMEDIATAS RECOMENDADAS

### 1. 🔴 CRÍTICO: Re-entrenar HMM con Período Correcto
```bash
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data
python scripts/orchestration/run_regime_hmm.py --start-date 1990-01-01 --end-date 2024-11-30
```
**Tiempo estimado**: 1 hora

### 2. 🟡 IMPORTANTE: Filtrar Warming Period
```python
# Filtrar primeras 200 observaciones por ticker
df_train = df_train.groupby('ticker').apply(lambda x: x.iloc[200:]).reset_index(drop=True)
```
**Tiempo estimado**: 30 minutos

### 3. 🟡 IMPORTANTE: Verificar Data Leakage en Target
```python
# Verificar que target no tenga look-ahead bias
# Target debería ser return_4q FUTURO, no contemporáneo
df_check = df_train[['date', 'ticker', 'target', 'returns_120d']]
```
**Tiempo estimado**: 30 minutos

### 4. 🟢 BUENO A TENER: EDA Completo
- Distribuciones de features
- Correlaciones
- Outliers
- Missing patterns
**Tiempo estimado**: 2 horas

---

## CONCLUSIÓN

**Estado General**: ✅✅✅ EXCELENTE - Datos mucho más completos de lo esperado

**Sorpresas Positivas**:
1. 476-498 tickers (FASE 2.1 completada)
2. Features técnicos completos (FASE 2.2 completada)
3. 3M+ observaciones (suficiente para ML robusto)
4. 30 años de historia (múltiples ciclos)

**Problemas Críticos**:
1. Regime predictions con date range incorrecto (re-entrenar HMM)
2. Warming period con valores 0.0 (filtrar primeras observaciones)
3. Features macro no integrados en gold data

**Tiempo estimado para resolver críticos**: 2-3 horas
