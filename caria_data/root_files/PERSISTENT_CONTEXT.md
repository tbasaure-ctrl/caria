# PERSISTENT CONTEXT — Wise Adviser

Este archivo debe ser incluido en cada sesión de Cursor para mantener contexto.

## Visión General

**Wise Adviser** es tu compañero racional de inversión. No predice precios, te ayuda a pensar claramente combinando:

1. **Análisis Cuantitativo**: DCF, múltiplos, reverse DCF
2. **Contexto Macro**: Régimen de mercado (normal, crisis, mania, QE)
3. **Señales Micro**: Calidad de negocio (ROIC, reinversión)
4. **Sabiduría Histórica**: 9,500 chunks de Graham, Buffett, Fisher, Marks
5. **Detección de Sesgos**: FOMO, confirmation bias, recency bias

## Objetivos del Sistema

- ✅ Valuación fundamentada (DCF + múltiplos)
- ✅ Contexto de régimen (macro environment)
- ✅ Sabiduría aplicable (RAG retrieval)
- ✅ Preguntas reflexivas (challenge assumptions)
- ✅ Detección de sesgos (behavioral finance)

## Archivos Críticos (Prioridad para Indexar)

### Must-Read (Cursor debe leer siempre)
1. `AI_CONTEXT.md` — Arquitectura, convenciones, prompts
2. `data_schema/schema.yaml` — Estructura de datos
3. `data_schema/data_dictionary.csv` — Diccionario de campos

### Código Core
4. `src/valuation/valuation_engine.py` — Motor de valuación
5. `src/models/wise_adviser_model.py` — Modelo principal
6. `src/models/encoders.py` — Encoders multi-modales
7. `src/retrieval/mcp_server.py` — RAG server
8. `scripts/embed_and_index.py` — Pipeline de embeddings

### Notebooks de Referencia
9. `notebooks/eda_valuation.ipynb` — Exploración de valuación
10. `notebooks/wisdom_embedding_test.ipynb` — Tests de embeddings

## Workflow Típico

### Análisis de un Ticker
```
Usuario: "Analiza NVDA"
  ↓
1. Valuation Engine → DCF + múltiplos
2. MCP Search → Wisdom chunks (themes: [valuation, risk])
3. Regime Detection → Contexto macro actual
4. Bias Check → Detectar FOMO/overconfidence
5. Reflexive Output → Números + Sabiduría + Preguntas
```

### Agregar Nueva Sabiduría
```
1. Editar: raw/chunks/wisdom_corpus_unified.jsonl
2. Ejecutar: python scripts/embed_and_index.py
3. Verificar: Query MCP /search con nuevo tema
```

### Entrenar Modelo
```
1. Preparar datos: scripts/01_download_data.py
2. Feature engineering: src/features/
3. Entrenar: scripts/02_train_model.py --epochs 50
4. Evaluar: notebooks/model_evaluation.ipynb
```

## Reglas para Cursor (IMPORTANTE)

### 🔴 Nunca hacer sin confirmar:
- Ejecutar scripts que modifiquen datos
- Entrenar modelos (consume tiempo/GPU)
- Modificar schema sin migration
- Eliminar checkpoints

### 🟢 Siempre hacer:
- Leer AI_CONTEXT.md antes de modificar código
- Verificar point-in-time correctness en features
- Agregar tests para nuevas funcionalidades
- Documentar cambios en schema
- Usar @Folders para incluir contexto relevante

### 🟡 Preguntar antes:
- Refactorings grandes
- Cambios en arquitectura del modelo
- Nuevas dependencias externas

## Prompts Estándar (Copiar/Pegar)

### Análisis Completo de Ticker
```
@Folders: src/valuation/, data_schema/

Analiza {TICKER} usando:
1. Valuación (DCF + múltiplos) de valuation_engine.py
2. Wisdom retrieval (MCP search con themes: [valuation, discipline])
3. Régimen actual (si modelo entrenado)
4. Preguntas reflexivas

Output: Resumen ejecutivo con números + sabiduría + verdict
```

### Refactoring con Tests
```
@Folders: src/features/, tests/

Refactoriza {MODULO} para:
1. Separar concerns (ingestion vs processing)
2. Mejorar testability
3. Mantener backward compatibility

Proponer cambios como PR con tests incluidos.
```

### Debug de Embeddings
```
@Folders: scripts/, infrastructure/mcp_server/

El MCP search no retorna resultados esperados para query: "{QUERY}"

Debug:
1. Verificar embedding generation en embed_and_index.py
2. Check índice en pgvector (SELECT COUNT(*) FROM wisdom_chunks)
3. Test filters en MCP /search endpoint
4. Revisar similarity threshold
```

## Estado Actual del Proyecto

### ✅ Completado
- [ ] Estructura base del repo
- [ ] Valuación (DCF + múltiplos)
- [ ] Wisdom corpus (9.5K chunks)
- [ ] Schema de datos definido
- [ ] Docker compose setup

### 🔄 En Progreso
- [ ] Embeddings indexados en pgvector
- [ ] MCP server funcional
- [ ] Feature engineering completo
- [ ] Tests de integración

### ⏳ Pendiente
- [ ] Modelo entrenado (encoders + fusion)
- [ ] Regime detection en producción
- [ ] Backtesting framework
- [ ] UI web

## Datos Sensibles (NO commitear)

Variables en `.env` (template en `.env.example`):
- `FMP_API_KEY` — Financial Modeling Prep
- `FRED_API_KEY` — Federal Reserve data
- `OPENAI_API_KEY` — Para embeddings
- `POSTGRES_PASSWORD` — DB password

## Comandos Útiles

### Setup Inicial
```bash
# Instalar dependencias
poetry install

# Levantar infraestructura
cd infrastructure && docker-compose up -d

# Verificar DB
psql -h localhost -U wise_user -d wise_adviser_db -c "SELECT COUNT(*) FROM wisdom_chunks;"

# Generar embeddings
python scripts/embed_and_index.py --input raw/chunks/
```

### Testing
```bash
# Tests completos
pytest tests/ -v --cov=src

# Test específico
pytest tests/test_valuation_engine.py -v

# Test de point-in-time
pytest tests/test_point_in_time.py -v
```

### Debugging
```bash
# Logs de entrenamiento
tail -f models/logs/training_latest.log

# Health check MCP server
curl http://localhost:8000/health

# Test MCP search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "valuation margin of safety", "top_k": 5}'
```

## Contacto y Soporte

Para issues o preguntas:
- Check `docs/FAQ.md`
- Review `notebooks/troubleshooting.ipynb`
- Ver logs en `models/logs/`

---

**Este archivo debe estar siempre abierto en Cursor** para mantener contexto persistente.

Última actualización: 2025-01-07
