# Auditoría de Sistemas - Proyecto Caria
**Fecha**: 2025-11-11
**Estado**: Implementación completa en progreso

---

## RESUMEN EJECUTIVO

Los 4 sistemas especializados están **implementados y funcionales**, pero hay **mejoras críticas** pendientes antes de producción. La arquitectura desacoplada funciona correctamente, pero algunos componentes necesitan refinamiento.

### Estado General
- ✅ **Sistema I (HMM)**: Implementado, modelo entrenado, necesita mejoras en normalización
- ⚠️ **Sistema II (RAG)**: Implementado, necesita configuración de pgvector y completar enriquecimiento
- ✅ **Sistema III (Factores)**: Completamente implementado, listo para uso
- ⚠️ **Sistema IV (DCF)**: Implementado, necesita manejo de deuda neta y FCF negativo

---

## SISTEMA I: Motor de Régimen HMM

### Archivo: `src/caria/models/regime/hmm_regime_detector.py`

#### ✅ Funcionalidades Correctas
- Implementación completa de HMM no supervisado con `hmmlearn`
- 4 estados latentes: expansion, slowdown, recession, stress
- Features macro bien seleccionadas: yield_curve_slope, VIX, sentiment, credit_spread
- Métodos save/load implementados correctamente
- Normalización z-score de features
- Predicción de regímenes históricos y actuales

#### ⚠️ PROBLEMAS IDENTIFICADOS

**P1.1 - CRÍTICO: Normalización inconsistente en predicción**
- **Ubicación**: `predict_proba()` líneas 223-225
- **Problema**: Normaliza usando estadísticas de las features **actuales** en lugar de las del **entrenamiento**
```python
# ❌ INCORRECTO (línea 224)
feature_array = (feature_array - np.nanmean(feature_array)) / (np.nanstd(feature_array) + 1e-6)
```
- **Impacto**: Predicciones inconsistentes entre entrenamiento y producción
- **Solución**: Guardar mean/std durante `fit()` y reutilizarlas en `predict_proba()`

**P1.2 - MEDIO: Mapeo hardcodeado de estados a regímenes**
- **Ubicación**: líneas 236-241
- **Problema**: Asume orden fijo (0=expansion, 1=slowdown, 2=recession, 3=stress) sin validar características de cada estado
- **Solución**: Analizar características de cada estado post-entrenamiento (means, covarianzas) para asignar etiquetas semánticas dinámicamente

**P1.3 - MENOR: Validación mínima de features**
- **Problema**: Solo valida >= 2 features y >= 100 observaciones
- **Solución**: Agregar validación de calidad de datos (missing values, outliers)

#### 📋 Recomendaciones de Mejora
1. Agregar persistencia de scaler (mean/std) junto con el modelo
2. Implementar análisis automático de características de estados HMM
3. Agregar métricas de confianza más sofisticadas (entropy, transition probabilities)
4. Considerar HMM jerárquico (estados dentro de regímenes)

---

## SISTEMA II: Servicio RAG (Socio Racional)

### Archivo: `src/caria/services/rag_service.py`

#### ✅ Funcionalidades Correctas
- Estructura completa con embeddings, vector store, retriever
- Soporte para LLM local (Ollama + transformers como fallback)
- Análisis crítico con identificación de sesgos
- Manejo graceful de errores (fallback a análisis simple)
- Parsing estructurado de respuestas LLM

#### ⚠️ PROBLEMAS IDENTIFICADOS

**P2.1 - CRÍTICO: Enriquecimiento de consultas incompleto**
- **Ubicación**: `enrich_query()` líneas 90-103
- **Problema**: No carga fundamentals ni prices desde base de datos
```python
# TODO: Cargar fundamentals y prices desde base de datos (línea 91)
```
- **Impacto**: Contexto empobrecido para análisis
- **Solución**: Integrar con base de datos de fundamentals y prices

**P2.2 - MEDIO: Búsqueda híbrida no implementada**
- **Ubicación**: `_hybrid_search()` líneas 105-126
- **Problema**: Solo hace búsqueda vectorial pura, sin filtros SQL
- **Solución**: Implementar filtros SQL por ticker, fecha, themes en pgvector

**P2.3 - MENOR: Dependencia externa de pgvector**
- **Problema**: RAG completamente deshabilitado si PostgreSQL no está disponible
- **Solución**: Implementar fallback a búsqueda local (FAISS, Chroma)

#### 📋 Recomendaciones de Mejora
1. Implementar carga de datos estructurados (fundamentals, prices, macro)
2. Agregar filtros SQL en búsqueda híbrida
3. Implementar caché de embeddings para queries comunes
4. Agregar métricas de calidad de respuestas (RAGAS)
5. Considerar re-ranking de chunks recuperados

---

## SISTEMA III: Motor de Factores Cuantitativos

### Archivo: `src/caria/models/factors/factor_screener.py`

#### ✅ Funcionalidades Correctas
- Implementación completa de 5 factores canónicos
- Normalización por percentiles (rank-based)
- Cross-sectional ranking por fecha
- Pesos ajustables por factor
- `RegimeAwareFactorScreener` con pesos dinámicos por régimen
- Composite score bien diseñado

#### ⚠️ PROBLEMAS IDENTIFICADOS

**P3.1 - MENOR: Validación de columnas insuficiente**
- **Ubicación**: Métodos `_calculate_*_score()`
- **Problema**: Asume existencia de columnas específicas sin validación completa
- **Ejemplo**: `_calculate_momentum_score()` intenta calcular returns si no existe
- **Solución**: Validar todas las columnas necesarias y dar warnings claros

**P3.2 - MENOR: Manejo de datos missing**
- **Problema**: `fillna(0.5)` puede introducir bias en ranking
- **Solución**: Considerar exclusión de stocks con features faltantes o imputación más sofisticada

#### 📋 Recomendaciones de Mejora
1. Agregar validación exhaustiva de columnas requeridas
2. Implementar manejo robusto de missing values (imputación, exclusión)
3. Agregar análisis de feature importance (cuáles factores dominan el ranking)
4. Considerar pesos adaptativos por sector/industria
5. Agregar backtesting de estrategia de factores

#### 🎯 EXCELENTE DISEÑO
- La separación de factores individuales permite análisis granular
- RegimeAwareFactorScreener es un diseño brillante que conecta Sistema I y III
- Rank normalization evita outliers dominando el score

---

## SISTEMA IV: Motor de Valuación DCF

### Archivo: `src/caria/models/valuation/dcf_valuator.py`

#### ✅ Funcionalidades Correctas
- Implementación DCF completa con proyección de FCF
- Ajuste dinámico de WACC según régimen macro
- Cálculo correcto de valor terminal
- Generación de explicaciones interpretables
- Soporte para proyecciones de NLP (Sistema II)

#### ⚠️ PROBLEMAS IDENTIFICADOS

**P4.1 - MEDIO: Simplificación excesiva de deuda**
- **Ubicación**: línea 149
- **Problema**: Asume sin deuda neta
```python
# Valor por acción (asumiendo sin deuda neta por simplicidad)
fair_value_per_share = enterprise_value / shares_outstanding
```
- **Impacto**: Valuaciones incorrectas para empresas con deuda significativa
- **Solución**: Incorporar net debt y cash en el cálculo

**P4.2 - MEDIO: No maneja FCF negativo**
- **Problema**: Si `current_fcf` es negativo, proyecciones son incorrectas
- **Solución**: Validar FCF positivo o usar metodología alternativa (revenue multiple)

**P4.3 - MENOR: Terminal growth fijo**
- **Problema**: 3% puede ser alto para algunas industrias o bajo para otras
- **Solución**: Ajustar terminal growth por industria o régimen

#### 📋 Recomendaciones de Mejora
1. Agregar cálculo de Enterprise Value → Equity Value (- debt + cash)
2. Implementar validación de FCF y metodologías alternativas para FCF negativo
3. Ajustar terminal growth por industria/régimen
4. Agregar análisis de sensibilidad (WACC ±1%, growth ±1%)
5. Implementar DCF con múltiples escenarios (bull/base/bear)

---

## AUDITORÍA DE DATOS

### Data Silver
✅ `data/silver/macro/` - Datos macro procesados
✅ `data/silver/regime/` - Predicciones de régimen HMM

### Data Gold
✅ `data/gold/train.parquet` - Dataset de entrenamiento
✅ `data/gold/val.parquet` - Dataset de validación
✅ `data/gold/test.parquet` - Dataset de prueba
✅ `data/gold/metadata/` - Metadatos

### Modelos
✅ `models/regime_hmm_model.pkl` - Modelo HMM entrenado
⚠️ Modelos legacy (quality_model.pkl, momentum_model.pkl, etc.) - **DEPRECATED**

---

## AUDITORÍA DE API

### Archivo: `services/api/app.py`

#### ✅ Funcionalidades Correctas
- FastAPI bien estructurada con 4 routers
- Inicialización de todos los servicios (Regime, Factor, Valuation)
- Manejo graceful de errores (servicios opcionales)
- Healthcheck endpoint completo
- Path resolution correcto entre services/ y caria_data/

#### ⚠️ OBSERVACIONES
- RAG opcional (no bloquea API si PostgreSQL no disponible)
- Modelo legacy opcional (CARIA_MODEL_CHECKPOINT)

---

## PRIORIDADES DE CORRECCIÓN

### 🔴 CRÍTICAS (Antes de producción)
1. **P1.1**: Normalización inconsistente en HMM (Sistema I)
2. **P2.1**: Enriquecimiento de consultas incompleto (Sistema II)
3. **P4.1**: Incorporar deuda neta en DCF (Sistema IV)

### 🟡 IMPORTANTES (Corto plazo)
4. **P1.2**: Mapeo automático de estados HMM
5. **P2.2**: Búsqueda híbrida SQL + vectorial
6. **P4.2**: Manejo de FCF negativo
7. **P3.1**: Validación de columnas en factores

### 🟢 MEJORAS (Mediano plazo)
8. Análisis de feature importance en factores
9. Análisis de sensibilidad en DCF
10. Caché de embeddings en RAG
11. Fallback local para RAG sin PostgreSQL

---

## CONCLUSIÓN

**Estado General**: ✅ BUENO - Arquitectura sólida, implementación funcional

**Listo para desarrollo**: ✅ SÍ
**Listo para producción**: ⚠️ NO (necesita correcciones críticas)

### Siguientes Pasos Recomendados
1. Corregir P1.1 (normalización HMM) - **30 minutos**
2. Completar P2.1 (enriquecimiento RAG) - **2 horas**
3. Corregir P4.1 (deuda neta DCF) - **1 hora**
4. Levantar API y probar endpoints - **1 hora**
5. Configurar pgvector y cargar embeddings - **2 horas**

**Tiempo estimado para correcciones críticas**: 6-8 horas
