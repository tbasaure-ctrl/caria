## Caria

Caria es una plataforma de inteligencia de inversiones centrada en mejorar la toma de decisiones humanas a través de datos macroeconómicos, señales micro, análisis de valuación, aprendizaje profundo y recuperación aumentada por conocimiento (RAG).

### Objetivos Principales
- Ingestar datos financieros y macroeconómicos versionados con punto-en-tiempo.
- Construir pipelines reproducibles para features multi-modales y embeddings semánticos.
- Entrenar modelos de deep learning que fusionen señales macro, micro, sentimentales y sabiduría histórica.
- Servir un sistema RAG y API analítica para generar reportes y recomendaciones reflexivas.

### Guía Rápida
1. Configura dependencias con `poetry install` y copia `.env.example` a `.env` con tus credenciales privadas.
2. Ejecuta `python scripts/orchestration/run_ingestion.py --pipeline-config configs/pipelines/ingestion.yaml` para descargar datos crudos.
3. Corre los pipelines de features, embeddings y entrenamiento según lo definido en `src/caria/pipelines/`.
4. Levanta los servicios (`services/api`, `services/mcp`) con Docker o `uvicorn` para exponer funcionalidades.

Consulta `docs/architecture.md` para una descripción completa de la arquitectura y flujos de datos.

