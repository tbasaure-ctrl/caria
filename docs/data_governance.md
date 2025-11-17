# Gobierno de Datos

## Versionado de Capas
- **Raw**: particionado por fecha de extracción (`data/raw/{source}/{YYYY-MM-DD}/`), inmutable.
- **Bronze**: normalización mínima; se documenta transformación en `docs/pipelines.md` (futuro) y se valida con Great Expectations.
- **Silver**: features intermedias con llaves claras (`ticker`, `date`).
- **Gold**: datasets listos para entrenamiento y serving, con snapshots etiquetadas (`gold/version=v1/`).

## Metadatos Obligatorios
- `data_catalog.yaml` (por crear) enumerará cada dataset con owner, SLA y políticas de refresco.
- Cada job registra ejecución y checksum en PostgreSQL (`src/caria/data_access/jobs_registry.py`).
- Embeddings almacenan `embedding_model`, `embedding_dim`, `index_version`, `created_at`.

## Calidad
- Validaciones automáticas antes de promover datos a Silver/Gold.
- Alertas Prefect + Slack/Webhook ante fallas.
- Lógica de point-in-time verificada mediante tests en `tests/pipelines/`.

## Seguridad y Acceso
- Variables sensibles en `.env` gestionadas con Vault en ambientes productivos.
- Roles RBAC para bases de datos y vector store.
- Logs de acceso y auditoría para APIs externas.

