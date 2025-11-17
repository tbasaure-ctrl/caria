# RAG Playbook

## Objetivo
Proveer un flujo consistente para ingestión de sabiduría histórica, generación de embeddings y exposición vía MCP/LLM.

## Pipeline
1. **Curación de Contenido**: añadir documentos a `data/raw/wisdom/{index_version}/` en formato JSONL con metadatos completos.
2. **Chunking**: `src/caria/embeddings/chunker.py` divide documentos según reglas (400 tokens, overlap 50).
3. **Embeddings**: `src/caria/embeddings/generator.py` llama a OpenAI u otro modelo local.
4. **Indexación**: `src/caria/retrieval/vector_store.py` sincroniza con `pgvector`, mantiene historial de versiones.
5. **Serving**: `services/mcp/server.py` expone endpoints `/search`, `/get`, `/upsert` con filtros avanzados.

## Buenas Prácticas
- Mantener `index_version` y `embedding_model` en cada registro.
- Agendar `services/workers/jobs/rag_refresh.py` para refrescar índices tras nuevos datos.
- Probar recall mediante `tests/integration/test_rag_retrieval.py` antes de liberar cambios.
- Documentar nuevos prompts y plantillas en `docs/prompts.md` (pendiente).

