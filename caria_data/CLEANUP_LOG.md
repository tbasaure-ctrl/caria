# Log de Limpieza - Eliminación de Contaminación por Look-Ahead Bias

**Fecha**: 2025-01-XX
**Motivo**: Eliminación completa de `regime_labels` y referencias relacionadas que introducen look-ahead bias según diagnóstico crítico.

## Archivos Eliminados

1. **`scripts/orchestration/run_regime_annotation.py`** - Script para ejecutar pipeline contaminado
2. **`silver/events/regime_context.parquet`** - Datos con etiquetas ex-post facto

## Archivos Modificados

### Esquemas y Diccionarios de Datos

1. **`data_schema/data_dictionary.csv`**
   - Eliminadas líneas 27-31: Definición de tabla `regime_labels`
   - Razón: Tabla contiene etiquetas ex-post facto que causan look-ahead bias

2. **`data_schema/schema.yaml`**
   - Eliminadas líneas 98-106: Definición de tabla `regime_labels`
   - Razón: Mismo que arriba

3. **`infrastructure/init_db.sql`**
   - Eliminadas líneas 114-122: `CREATE TABLE regime_labels`
   - Razón: Tabla no debe existir en base de datos

### Pipelines

4. **`src/caria/pipelines/gold_builder_pipeline.py`**
   - Eliminada función `_label_regimes()` completa (líneas 139-158)
   - Eliminada lógica de carga de `events_df` y llamada a `_label_regimes()`
   - Eliminadas columnas `regime_label`, `regime_name`, `target_regime` de `_compute_targets()`
   - Eliminadas columnas `start_date`, `end_date`, `type`, `region`, `description`, `sources`, `regime_name` de `exclude_columns`
   - Agregado comentario: "Skip events datasets - regime detection will be done via HMM (Sistema I) in future"
   - Razón: Función aplicaba etiquetas ex-post a datos históricos

5. **`src/caria/pipelines/regime_annotation_pipeline.py`**
   - Marcado como DEPRECATED en docstring
   - Agregado comentario explicando que será reemplazado por HMM no supervisado
   - Razón: Pipeline genera datos contaminados, se mantiene solo para referencia histórica

### Configuraciones

6. **`configs/pipelines/gold_builder.yaml`**
   - Eliminada entrada `events/regime_context.parquet` (líneas 22-24)
   - Agregado comentario explicando remoción y referencia a Sistema I (HMM)

7. **`configs/pipelines/gold_builder_with_macro.yaml`**
   - Eliminada entrada `events/regime_context.parquet` (líneas 22-24)
   - Agregado comentario explicando remoción y referencia a Sistema I (HMM)

8. **`configs/pipelines/feature_store.yaml`**
   - Eliminada entrada `event_regime_context` de features (líneas 42-45)
   - Eliminada referencia de `event_regime_context` de `multi_modal_inputs` (línea 62)
   - Agregado comentario explicando remoción

## Impacto

- **Pipelines de entrenamiento**: Ya no usarán datos contaminados con look-ahead bias
- **Gold Builder**: Funciona sin `regime_context`, preparado para recibir probabilidades de régimen del Sistema I (HMM)
- **Base de datos**: Schema limpio sin tabla `regime_labels`

## Próximos Pasos

1. Implementar Sistema I: HMM no supervisado para detección de régimen (reemplazo correcto)
2. Actualizar pipelines para consumir probabilidades de régimen del HMM en lugar de labels
3. Eliminar completamente `regime_annotation_pipeline.py` una vez que Sistema I esté implementado

## Validación

- [ ] Ejecutar `gold_builder_flow` sin errores
- [ ] Verificar que no hay referencias restantes a `regime_labels` o `regime_context`
- [ ] Confirmar que modelos entrenados no usan datos contaminados

