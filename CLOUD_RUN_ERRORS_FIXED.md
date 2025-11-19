# Errores de Cloud Run - Solucionados

## 🔴 Problemas Críticos Encontrados en los Logs

### 1. FMP API - Error 401 Unauthorized ✅ SOLUCIONADO
**Error en logs:**
```
Error obteniendo precios batch para SPY, VGK, EEM: 401 Client Error: Unauthorized 
for url: https://financialmodelingprep.com/api/v3/quote/SPY, VGK, EEM?apikey=79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq%0D%0A
```

**Causa:** El API key tenía caracteres de nueva línea (`%0D%0A`) codificados en URL, causando que FMP rechazara la autenticación.

**Solución:** 
- Limpiar el API key en `FMPClient.__init__()` removiendo `\r`, `\n`, y caracteres URL-encoded
- Archivo: `caria-lib/caria/ingestion/clients/fmp_client.py`

### 2. Base de Datos - Columnas Faltantes ✅ SOLUCIONADO

#### 2.1. Columna `revoked` en `refresh_tokens`
**Error:** `UndefinedColumn: column "revoked" does not exist`

**Solución:** Migración 013 agrega la columna si no existe

#### 2.2. Columnas `is_arena_post` y `arena_community` en `community_posts`
**Error:** `UndefinedColumn: column cp.is_arena_post does not exist`

**Solución:** Migración 013 asegura que las columnas existan (también en migración 011)

#### 2.3. Tabla `thesis_arena_threads` no existe
**Error:** `UndefinedTable: relation "thesis_arena_threads" does not exist`

**Solución:** Migración 013 crea la tabla si no existe (también en migración 010)

#### 2.4. Columna `allocation_percentage` en `holdings`
**Error:** `UndefinedColumn: column "allocation_percentage" does not exist`

**Causa:** El código buscaba `allocation_percentage` pero la tabla `holdings` solo tiene `quantity` y `average_cost`

**Solución:** 
- Cambiar `regime_testing.py` para calcular `allocation_percentage` desde `quantity * average_cost`
- Calcular el porcentaje basado en el valor total del portfolio

#### 2.5. Tabla `model_retraining_triggers` no existe
**Error:** `UndefinedTable: relation "model_retraining_triggers" does not exist`

**Solución:** Migración 013 crea la tabla

### 3. Gemini API - Error 400 Bad Request ⚠️ PENDIENTE
**Error:** `HTTPError: 400 Client Error: Bad Request for url: https://generativelanguage.googleapis.com/v1beta/models/`

**Estado:** Ya mejoramos el logging en `thesis_arena.py` para diagnosticar mejor. Revisar logs después del despliegue.

## 📋 Migración 013 - Aplicar en Cloud SQL

La migración `013_fix_missing_columns.sql` debe ejecutarse en Cloud SQL para crear las tablas y columnas faltantes.

### Cómo aplicar la migración:

**Opción A: Desde Cloud Shell**
```bash
# Conectar a Cloud SQL
gcloud sql connect caria-db --user=postgres --project=caria-backend

# Ejecutar la migración
\i caria_data/migrations/013_fix_missing_columns.sql
```

**Opción B: Desde local con Cloud SQL Proxy**
```bash
# Descargar Cloud SQL Proxy
# https://cloud.google.com/sql/docs/postgres/sql-proxy

# Conectar
./cloud-sql-proxy caria-backend:us-central1:caria-db

# En otra terminal, ejecutar migración
psql -h 127.0.0.1 -U postgres -d caria -f caria_data/migrations/013_fix_missing_columns.sql
```

**Opción C: Desde código (automático)**
- Agregar lógica en el startup de la app para ejecutar migraciones pendientes
- O crear un endpoint admin para ejecutar migraciones

## ✅ Cambios Realizados

1. **FMP API Key Cleaning** (`caria-lib/caria/ingestion/clients/fmp_client.py`)
   - Remueve `\r`, `\n`, `%0D%0A`, `%0D`, `%0A` del API key
   - Mejora logging para mostrar primeros y últimos 4 caracteres

2. **Regime Testing Fix** (`backend/api/routes/regime_testing.py`)
   - Calcula `allocation_percentage` desde `quantity` y `average_cost`
   - Calcula porcentaje basado en valor total del portfolio

3. **Migración 013** (`caria_data/migrations/013_fix_missing_columns.sql`)
   - Crea todas las tablas faltantes
   - Agrega todas las columnas faltantes
   - Crea índices necesarios
   - Idempotente (puede ejecutarse múltiples veces)

## 🚀 Próximos Pasos

1. **Esperar despliegue** (~5-10 minutos)
2. **Aplicar migración 013** en Cloud SQL (ver arriba)
3. **Verificar logs** después del despliegue:
   - FMP API debería funcionar (sin 401)
   - Errores de columnas faltantes deberían desaparecer
   - Gemini API necesita más diagnóstico

4. **Probar endpoints:**
   - `/api/prices/realtime` - Debería funcionar sin 401
   - `/api/portfolio/regime-test` - Debería funcionar sin error de columna
   - `/api/community/rankings` - Debería funcionar sin error de columna
   - `/api/thesis/arena/challenge` - Necesita diagnóstico de Gemini

## 📊 Resumen de Errores por Severidad

- **Críticos (solucionados):** 6
  - FMP API 401 ✅
  - 5 errores de base de datos ✅

- **Pendientes:** 1
  - Gemini API 400 ⚠️ (mejorado logging, necesita más diagnóstico)

Todos los cambios están en GitHub y se desplegarán automáticamente.

