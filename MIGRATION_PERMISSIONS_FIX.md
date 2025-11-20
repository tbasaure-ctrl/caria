# Fix para Error de Permisos en Migración 013

## 🔴 Problema Encontrado

Al ejecutar la migración 013, obtienes el error:
```
Execution failed. All statements are aborted. Details: pq: must be owner of table refresh_tokens
```

## ✅ Solución

He actualizado la migración para usar bloques `DO $$` que verifican si las columnas existen antes de intentar agregarlas. Esto es más permisivo y no requiere ser el owner de la tabla.

## 📋 Opciones para Ejecutar la Migración

### Opción 1: Usar la Versión Actualizada (Recomendada)

La migración `013_fix_missing_columns.sql` ahora usa bloques `DO $$` que verifican la existencia de columnas antes de agregarlas. Esto debería funcionar incluso si no eres el owner.

**Pasos:**
1. Copia el contenido actualizado de `013_fix_missing_columns.sql`
2. Pégalo en el SQL Editor de Cloud SQL
3. Ejecuta la migración

### Opción 2: Conectarse como Usuario postgres

Si la Opción 1 no funciona, necesitas conectarte como el usuario `postgres` (superusuario):

**Desde Cloud Shell:**
```bash
gcloud sql connect caria-db --user=postgres --project=caria-backend
```

Luego ejecuta la migración desde ahí.

**Desde Google Cloud Console:**
1. Ve a Cloud SQL → Instancias → `caria-db`
2. Haz clic en "Conectar usando Cloud Shell"
3. Se conectará automáticamente como `postgres`
4. Ejecuta: `psql -d caria -f /path/to/013_fix_missing_columns.sql`

### Opción 3: Usar la Versión Alternative

He creado `013_fix_missing_columns_alternative.sql` que incluye comandos `GRANT` para asegurar permisos antes de ejecutar `ALTER TABLE`.

## 🔍 Verificar que Funcionó

Después de ejecutar la migración, verifica que las columnas y tablas existen:

```sql
-- Verificar columna revoked en refresh_tokens
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'refresh_tokens' AND column_name = 'revoked';

-- Verificar columnas arena en community_posts
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'community_posts' 
AND column_name IN ('is_arena_post', 'arena_community', 'arena_thread_id', 'arena_round_id');

-- Verificar que thesis_arena_threads existe
SELECT table_name 
FROM information_schema.tables 
WHERE table_name = 'thesis_arena_threads';

-- Verificar que model_retraining_triggers existe
SELECT table_name 
FROM information_schema.tables 
WHERE table_name = 'model_retraining_triggers';
```

Si todas las queries devuelven resultados, la migración fue exitosa.

## 📝 Nota Importante

Si sigues teniendo problemas de permisos, puede ser que el usuario con el que te conectas no tenga permisos suficientes. En ese caso:

1. Conéctate como `postgres` (superusuario)
2. O pide a tu administrador de Cloud SQL que ejecute la migración
3. O pide que se otorguen permisos `ALTER TABLE` a tu usuario

Los cambios están en GitHub y listos para usar.


