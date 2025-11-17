# 🗺️ Guía de Próximos Pasos - Caria

## ✅ Estado Actual

**Completado**:
- ✅ Sistema I (HMM Régimen) - **ENTRENADO Y FUNCIONANDO**
- ✅ Sistema II (RAG) - Implementado, necesita embeddings en pgvector
- ✅ Sistema III (Factores) - Implementado, listo para usar
- ✅ Sistema IV (Valuación) - Implementado, listo para usar
- ✅ MLOps (Purged CV) - Implementado
- ✅ Endpoints API - Todos creados
- ✅ Documentación de integración UI

## 🎯 Próximos Pasos Recomendados (en orden)

### PASO 1: Verificar y Probar Sistemas ✅ (5 min)

**Objetivo**: Asegurarte de que todos los sistemas funcionan correctamente.

```powershell
# 1. Verificar que el modelo HMM está entrenado
ls models/regime_hmm_model.pkl

# 2. Probar endpoint de régimen (si tienes API corriendo)
# GET http://localhost:8000/api/regime/current

# 3. Verificar datos disponibles
ls data/silver/fundamentals/
ls data/silver/technicals/
```

**Resultado esperado**: Todos los archivos existen y los endpoints responden.

---

### PASO 2: Configurar Sistema II (RAG) - Embeddings en pgvector 📚 (15-30 min)

**Objetivo**: Cargar embeddings de sabiduría en pgvector para que el RAG funcione.

**Pasos**:

1. **Verificar PostgreSQL con pgvector**:
```powershell
# Verificar que PostgreSQL está corriendo y tiene pgvector
python -c "from caria.retrieval.vector_store import VectorStore; from caria.config.settings import Settings; s = Settings.from_yaml('configs/base.yaml'); vs = VectorStore.from_settings(s); print('✅ pgvector configurado')"
```

2. **Cargar embeddings de sabiduría**:
```powershell
# Ejecutar pipeline de sabiduría (si existe)
python scripts/orchestration/run_wisdom_pipeline.py
# O manualmente usando el servicio
```

3. **Probar endpoint RAG**:
```powershell
# POST http://localhost:8000/api/analysis/challenge
# Body: {"thesis": "Buy NVDA because AI is the future", "ticker": "NVDA"}
```

**Documentación**: Ver `docs/rag_playbook.md` si existe.

---

### PASO 3: Probar Sistema III (Factores) 🔍 (10 min)

**Objetivo**: Verificar que el screening de factores funciona.

**Pasos**:

1. **Verificar datos de fundamentals**:
```powershell
python -c "import pandas as pd; df = pd.read_parquet('data/silver/fundamentals/quality_signals.parquet'); print(f'✅ {len(df)} observaciones de quality'); print(df.columns.tolist()[:10])"
```

2. **Probar endpoint de factores**:
```powershell
# POST http://localhost:8000/api/factors/screen
# Body: {"top_n": 20}
```

**Resultado esperado**: Lista de empresas rankeadas por composite score.

---

### PASO 4: Probar Sistema IV (Valuación) 💰 (10 min)

**Objetivo**: Verificar que la valuación funciona para empresas consolidadas.

**Pasos**:

1. **Probar endpoint de valuación**:
```powershell
# POST http://localhost:8000/api/valuation/AAPL
# O con body: {"ticker": "AAPL", "current_price": 150.0}
```

**Resultado esperado**: Valuación DCF con explicación de por qué es caro/barato.

**Nota**: Para empresas pre-revenue, necesitarás datos cualitativos (actualmente usa placeholders).

---

### PASO 5: Levantar API y Verificar Endpoints 🚀 (15 min)

**Objetivo**: Asegurarte de que todos los endpoints funcionan.

**Pasos**:

1. **Levantar API**:
```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\services
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**✅ Nota**: Los paths están configurados automáticamente. No necesitas configurar PYTHONPATH.

2. **Verificar healthcheck**:
```powershell
# GET http://localhost:8000/health
# Debe mostrar estado de todos los servicios
```

3. **Probar cada endpoint**:
   - `GET /api/regime/current` - Régimen macro
   - `POST /api/factors/screen` - Screening de factores
   - `POST /api/valuation/{ticker}` - Valuación
   - `POST /api/analysis/challenge` - Challenge thesis (RAG)

**Documentación**: Ver `docs/ui_integration.md` para detalles de cada endpoint.

---

### PASO 6: Conectar con UI 🎨 (Variable)

**Objetivo**: Conectar los endpoints con la interfaz de usuario existente.

**Pasos**:

1. **Revisar documentación de integración**:
   - `docs/ui_integration.md` - Mapeo completo de endpoints a UI

2. **Conectar cada componente**:
   - **MODEL OUTLOOK**: `GET /api/regime/current`
   - **IDEAL PORTFOLIO**: `POST /api/factors/screen` + `GET /api/valuation/{ticker}`
   - **TOP MOVERS**: `POST /api/factors/screen`
   - **Challenge Thesis**: `POST /api/analysis/challenge`

3. **Probar flujo completo**:
   - Abrir UI
   - Verificar que cada componente muestra datos correctos
   - Probar interacciones

**Ejemplos de código**: Ver `docs/ui_integration.md` para ejemplos TypeScript/React.

---

### PASO 7: Mejorar Datos y Modelos 🔧 (Opcional)

**Objetivo**: Mejorar calidad de datos y modelos según necesidad.

**Tareas opcionales**:

1. **Datos cualitativos para Scorecard**:
   - Implementar ingesta de datos cualitativos (team quality, opportunity size, etc.)
   - Mejorar `ScorecardValuator` con datos reales

2. **NLP para proyecciones DCF**:
   - Integrar análisis de earnings calls
   - Extraer proyecciones de crecimiento desde NLP

3. **Más datos macro**:
   - Agregar más series FRED si es necesario
   - Mejorar features macro

4. **Entrenar más modelos**:
   - Ajustar hiperparámetros del HMM si es necesario
   - Entrenar modelos de factores si quieres usar ML

---

## 📋 Checklist Rápido

Usa este checklist para trackear tu progreso:

- [ ] **PASO 1**: Verificar sistemas funcionando
- [ ] **PASO 2**: Configurar RAG (pgvector + embeddings)
- [ ] **PASO 3**: Probar Sistema III (Factores)
- [ ] **PASO 4**: Probar Sistema IV (Valuación)
- [ ] **PASO 5**: Levantar API y verificar endpoints
- [ ] **PASO 6**: Conectar con UI
- [ ] **PASO 7**: Mejoras opcionales

---

## 🆘 Troubleshooting

### Si un endpoint no funciona:

1. **Verificar servicio en healthcheck**: `GET /health`
2. **Revisar logs**: Ver qué error específico aparece
3. **Verificar datos**: Asegúrate de que los datos necesarios existen
4. **Verificar configuración**: Revisa `configs/base.yaml`

### Si falta un módulo:

```powershell
# Instalar desde requirements.txt
pip install -r requirements.txt

# O instalar específico
pip install nombre-del-modulo
```

### Si hay errores de paths:

**SIEMPRE ejecuta desde `caria_data/`**:
```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data
```

---

## 📚 Documentación de Referencia

- `EJECUTAR_SCRIPTS.md` - Cómo ejecutar scripts
- `IMPLEMENTATION_SUMMARY.md` - Resumen completo de implementación
- `docs/ui_integration.md` - Integración con UI
- `docs/mlops_protocol.md` - Protocolo MLOps
- `SETUP_COMPLETO.md` - Estado del setup

---

## 🎯 Recomendación Inmediata

**Empieza con PASO 5** (Levantar API) si quieres ver resultados rápidos. Es la forma más rápida de verificar que todo funciona.

Luego sigue con PASO 6 (Conectar UI) para tener el sistema completo funcionando.

---

## 💡 Tips

1. **Usa el healthcheck**: Siempre verifica `/health` primero
2. **Revisa logs**: Los logs de Prefect y FastAPI son muy útiles
3. **Empieza simple**: Prueba un endpoint a la vez
4. **Documenta problemas**: Si encuentras algo, documenta la solución

---

**¿Listo para empezar?** Recomiendo comenzar con **PASO 5** (Levantar API) para ver todo funcionando rápidamente.

