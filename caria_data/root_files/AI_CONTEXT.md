# AI Context — Wise Adviser Project

## Propósito
Sistema de análisis de inversiones que combina:
- **Macroeconomía**: Detección de regímenes (QE, crisis, expansión)
- **Análisis Micro**: Calidad de negocios (ROIC, reinversión, márgenes)
- **Valuación**: DCF, múltiplos, reverse DCF
- **Psicología/Sentimiento**: Detección de manías, pánico, sesgos conductuales
- **Sabiduría histórica**: 9,500+ chunks de Graham, Buffett, Fisher, Marks, Dalio

El objetivo NO es predecir precios, sino ayudar al usuario a **pensar claramente** cuando las emociones nublan el juicio.

---

## Arquitectura del Proyecto

### Estructura de Carpetas
```
wise_adviser/
├── src/
│   ├── data_ingestion/        # APIs: FMP, FRED
│   ├── features/              # Engineering: macro, micro, behavioral
│   ├── models/                # Deep learning: encoders + fusion
│   ├── valuation/             # DCF, múltiplos, análisis
│   ├── retrieval/             # RAG, MCP server
│   └── training/              # PyTorch datasets, loops
├── scripts/
│   ├── 01_download_data.py    # Ingesta de datos
│   ├── 02_train_model.py      # Entrenamiento del modelo
│   ├── embed_and_index.py     # Generar embeddings para RAG
│   ├── valuation_engine.py    # Motor DCF/múltiplos
│   └── wise_advisor_with_valuation.py  # Integración final
├── raw/                       # Datos crudos (parquet)
│   └── chunks/                # wisdom_corpus_unified.jsonl (9.5K chunks)
├── data/                      # Datos procesados
├── silver_parquet/            # Features procesados
├── gold_parquet/              # Model-ready data
├── models/                    # Checkpoints (.pth)
├── data_schema/               # Schema + diccionario de datos
│   ├── schema.yaml
│   └── data_dictionary.csv
├── notebooks/                 # EDA, ejemplos
│   ├── eda_valuation.ipynb
│   ├── regime_detection_baseline.ipynb
│   └── wisdom_embedding_test.ipynb
├── infrastructure/            # Docker, MCP server
│   ├── docker-compose.yml
│   ├── init_db.sql
│   └── mcp_server/
│       ├── app.py
│       ├── Dockerfile
│       └── requirements.txt
├── experiments/               # Manifests de índices
├── tests/                     # Unit/integration tests
│   ├── test_point_in_time.py
│   ├── test_valuation_engine.py
│   └── test_rag_retrieval.py
├── .env.example               # Template de variables
├── requirements.txt
├── pyproject.toml             # Poetry config
└── README.md
```

---

## Dimensiones del Modelo (Multi-Modal Fusion)

| Dimensión | Inputs | Encoder | Output Latent |
|-----------|--------|---------|---------------|
| **Macro** | Tasas, inflación, M2, yield curve | Bi-LSTM (90 días) | 256-dim |
| **Market Structure** | VIX, correlaciones, breadth | Self-Attention | 256-dim |
| **Behavioral** | Sentiment, retail flows, opciones | Feedforward | 256-dim |
| **Micro** | ROIC, reinversión, márgenes | Feedforward | 256-dim |
| **Historical** | Crashes/manías etiquetados | LSTM pre-entrenado | 256-dim |
| **Wisdom** | Embeddings de 9.5K chunks | Transformer | 256-dim |

**Fusion**: Cross-modal attention → MLP → 3 tareas simultáneas:
1. Clasificación de régimen (normal, crash, mania, QE, recesión)
2. Predicción de retornos (20 días)
3. Probabilidad de drawdown (>10%)

---

## Convenciones de Código

### Tecnología
- **Python**: 3.11+
- **Gestión de dependencias**: `poetry` (pyproject.toml)
- **Deep Learning**: PyTorch 2.x
- **Data**: Pandas, Parquet
- **Testing**: `pytest` con cobertura >80%
- **Embeddings**: `text-embedding-ada-002` (OpenAI) o modelo local
- **Chunk size**: 400 tokens con overlap de 50

### Estilo
- **PEP 8** con líneas de 88 caracteres (Black formatter)
- Type hints obligatorios (`def function(x: float) -> dict:`)
- Docstrings: Google style
- Imports: `isort` ordenados
- Tests: Cobertura >80% para módulos críticos

### Point-in-Time Correctness
- **Crítico**: Todos los datos fundamentales respetan fechas "as-reported"
- Nunca usar datos futuros en features (lookahead bias)
- Fechas en formato ISO 8601 (YYYY-MM-DD)

### Versionado de Datos
- Raw data: `raw/YYYY-MM-DD/ticker.parquet`
- Features: `silver_parquet/v{version}/`
- Índices de embeddings: Campo obligatorio `index_version`

---

## Archivos Clave para Cursor

### Código Principal
- `src/models/wise_adviser_model.py` — Arquitectura principal
- `src/models/encoders.py` — Encoders multi-modales
- `src/valuation/valuation_engine.py` — DCF, múltiplos, reverse DCF
- `scripts/wise_advisor_with_valuation.py` — Integración valuation + wisdom
- `src/config.py` — Configuración global

### Data Schema
- `data_schema/schema.yaml` — Definición de tablas y campos
- `data_schema/data_dictionary.csv` — Diccionario de datos

### Notebooks
- `notebooks/eda_valuation.ipynb` — Exploración de valuación
- `notebooks/regime_detection_baseline.ipynb` — Clustering de regímenes
- `notebooks/wisdom_embedding_test.ipynb` — Verificar embeddings

### Infraestructura
- `infrastructure/docker-compose.yml` — Postgres + pgvector + MCP
- `infrastructure/mcp_server/app.py` — RAG server para wisdom retrieval
- `infrastructure/init_db.sql` — Setup inicial de DB

---

## Workflow de Datos

### Pipeline de Ingesta
1. **Download** (`scripts/01_download_data.py`)
   - FMP: Precios, fundamentales
   - FRED: Macro (tasas, inflación, M2)
   - Twitter/Reddit APIs: Sentiment (opcional)

2. **Normalize** (`src/data_ingestion/`)
   - Dedupe por (ticker, date)
   - Rellenar missings (forward-fill con límite 5 días)
   - Validar fechas as-reported

3. **Feature Engineering** (`src/features/`)
   - Macro: Yield curve slope, policy regime (HMM)
   - Market: Rolling correlations, breadth (advance/decline)
   - Micro: ROIC trends, reinvestment quality

4. **Embeddings** (`scripts/embed_and_index.py`)
   - Chunk wisdom docs (400 tokens, overlap 50)
   - Generar embeddings (OpenAI o local)
   - Guardar en vector DB con metadata

5. **Gold Data** (`gold_parquet/`)
   - Alinear todas las dimensiones por (ticker, date)
   - Ventanas de lookback (90 días macro, 20 días micro)

### RAG Pipeline (Wisdom Retrieval)
```
User query: "¿Debo comprar NVDA ahora?"
    ↓
Embed query → Search vector DB (pgvector)
    ↓
Filters: {sentiment: positive, themes: [valuation, discipline]}
    ↓
Retrieve top-5 chunks → Feed to LLM con contexto DCF
    ↓
Output: Análisis reflexivo con citas de Buffett, Graham
```

---

## Metadata Mínimo para Documentos

Usa estos campos para filtrar y explicar contexto:

- `id` (UUID)
- `source` (macro, micro, report, news, tweet, book)
- `date` (ISO 8601)
- `ticker` (si aplica)
- `industry` (tech, finance, energy, etc.)
- `country` (ISO 3166-1 alpha-2: US, CL, etc.)
- `author` (nombre del autor)
- `sentiment_score` ([-1, 1])
- `confidence` ([0, 1])
- `embedding_model` (text-embedding-ada-002, etc.)
- `index_version` (v1, v2, etc.)

Ejemplo de uso: 
```python
filters = {
    "industry": "fintech",
    "country": "CL",
    "date_range": ["2018-01-01", "2024-12-31"],
    "sentiment_score_min": 0.2
}
```

---

## Prompts / Templates Preferidos

### Análisis de Acciones (VAL_SUMMARY)
```
Analiza {ticker} considerando:
1. Valuación DCF (fair value vs precio actual)
2. Régimen de mercado actual (macro)
3. Sabiduría de inversores (buscar en corpus: valuation, risk)
4. Preguntas reflexivas (¿pagarías más si sube 20%?)

Formato: Primero números, luego contexto, luego sabiduría, luego preguntas.
```

### Detección de Sesgos (BIAS_CHECK)
```
El usuario dice: "{user_statement}"
Contexto: {ticker}, precio actual ${price}, momentum reciente {momentum}.

Identifica posibles sesgos:
- FOMO (Fear of Missing Out)
- Confirmation bias
- Recency bias
- Overconfidence

Busca sabiduría relevante para contrarrestar el sesgo.
```

### Resumen Ejecutivo (EXEC_SUMMARY)
```
Resume en 3-4 párrafos:
1. Situación actual del ticker
2. Valuación (fair value, MOS, implied growth)
3. Régimen de mercado
4. Recomendación con sabiduría aplicable
```

### Checklist Pre-Decisión (PRE_DECISION)
```
Antes de {acción} en {ticker}:
☐ ¿Entiendes el negocio?
☐ ¿Valuación razonable?
☐ ¿Régimen favorable?
☐ ¿Identificaste tus sesgos?
☐ ¿Qué dicen los grandes inversores?
```

---

## Credenciales (NO en repo)

Crear `.env` en raíz:
```bash
# APIs de datos
FMP_API_KEY=your-fmp-api-key-here
FRED_API_KEY=your_key_here
TWITTER_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_key_here

# Embeddings
OPENAI_API_KEY=your_key_here

# Vector DB
POSTGRES_USER=wise_user
POSTGRES_PASSWORD=secure_password_change_me
POSTGRES_DB=wise_adviser_db
PGVECTOR_CONNECTION=postgresql://wise_user:secure_password_change_me@localhost:5432/wise_adviser_db

# MLflow (opcional)
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## Ejecución Rápida

### Descargar datos
```bash
python scripts/01_download_data.py --ticker AAPL --start 2020-01-01
```

### Generar embeddings
```bash
python scripts/embed_and_index.py --input raw/chunks/ --output embeddings/
```

### Entrenar modelo
```bash
python scripts/02_train_model.py --epochs 50 --batch_size 32 --gpus 1
```

### Análisis con valuación
```bash
python scripts/wise_advisor_with_valuation.py --ticker NVDA
```

### Levantar infraestructura (RAG)
```bash
cd infrastructure
docker-compose up -d
```

---

## Testing

### Ejecutar tests
```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Tests críticos
- `tests/test_point_in_time.py` — Verificar no hay lookahead bias
- `tests/test_valuation_engine.py` — DCF con valores conocidos
- `tests/test_rag_retrieval.py` — Recall de wisdom chunks

---

## Reglas Arquitectónicas

1. **Todo acceso a DB pasa por `src/db/`** (abstracción)
2. **Scripts son idempotentes** (correr 2x = mismo resultado)
3. **Features nunca en raw/** (solo en silver_parquet/)
4. **Checkpoints con metadata** (epoch, loss, hyperparams en JSON)
5. **Embeddings versionados** (campo `embedding_model` + `index_version`)
6. **No ejecutar scripts sin confirmar** con usuario
7. **Proponer cambios estructurales** con PR + tests

---

## Instrucciones para Cursor

### Al modificar código
- **Antes de editar**: Leer `data_schema/schema.yaml` para entender estructura
- **Nuevas features**: Agregar test en `tests/features/`
- **Nuevos encoders**: Documentar dimensión de output en docstring
- **Scripts de ingesta**: Verificar point-in-time correctness

### Al refactorizar
- **Separar ingestion de feature engineering** (diferentes módulos)
- **Encoders modulares** (cada uno en su clase)
- **Config centralizado** (usar `src/config.py`, no hardcodear)
- **Sugerir PR** con cambios y tests

### Al debuggear
- **Check logs**: `models/logs/training_{timestamp}.log`
- **Visualizar embeddings**: `notebooks/debug_embeddings.ipynb`
- **Regime predictions**: `notebooks/debug_regime_classification.ipynb`

### MCP Server (RAG)
- **Endpoint search**: `POST http://localhost:8000/search`
- **Filters recomendados**: ticker, industry, themes, date_range, sentiment_score
- **Siempre incluir**: `index_version` para consistencia

---

## Do's and Don'ts

### ✅ DO
- Versionar datos (raw, silver, gold con v1, v2...)
- Point-in-time correctness siempre
- Tests para features críticos
- Documentar cambios en schema con migrations
- Guardar experimentos en MLflow/W&B
- Usar @Folders en Cursor para contexto

### ❌ DON'T
- Hardcodear API keys
- Usar datos futuros en features (lookahead bias)
- Modificar raw/ después de descarga (inmutable)
- Entrenar sin validation split
- Deployar sin tests de integración
- Ejecutar scripts sin confirmar con usuario

---

## Próximos Pasos (Roadmap)

### Fase 1: Completar RAG (Esta semana)
- [ ] Levantar Postgres + pgvector
- [ ] Embeddings de 9.5K wisdom chunks
- [ ] MCP server con endpoints `/search` y `/get`
- [ ] Conectar Cursor al MCP

### Fase 2: Training del modelo (2 semanas)
- [ ] Etiquetar episodios históricos (crashes 1929, 2000, 2008, 2020)
- [ ] Pre-entrenar historical encoder
- [ ] Entrenar modelo completo multi-task
- [ ] Backtesting por régimen

### Fase 3: Integración completa (1 mes)
- [ ] UI web (Flask/FastAPI) para consultas
- [ ] Real-time regime detection
- [ ] Alertas por email/Slack cuando cambia régimen
- [ ] Portfolio tracking

---

## Recursos Externos

- **FMP API Docs**: https://site.financialmodelingprep.com/developer/docs
- **FRED API**: https://fred.stlouisfed.org/docs/api/
- **pgvector**: https://github.com/pgvector/pgvector
- **MCP examples**: https://github.com/yusufferdogan/mcp-pgvector-server
- **Cursor Docs**: https://cursor.sh/docs

---

**Última actualización**: 2025-01-07  
**Versión del contexto**: v1.0  
**Mantenido por**: Wise Adviser Team
