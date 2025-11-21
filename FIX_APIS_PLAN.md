# Plan para Arreglar APIs y Conexiones

## Problemas Identificados

1. **FMP_API_KEY no configurada** → Precios y valuación no funcionan
2. **GEMINI_API_KEY no funciona** → Chat/Análisis usa fallback básico
3. **Database connection issues** → Algunos servicios intentan conectar a "postgres" hostname

---

## Fix 1: Agregar FMP_API_KEY a Cloud Run

### Opción A: Como Secret (Recomendado)
```bash
# Crear secret
gcloud secrets create fmp-api-key --project=caria-backend --data-file=-

# O si ya existe, actualizar
echo "your-fmp-api-key-here" | gcloud secrets versions add fmp-api-key --project=caria-backend --data-file=-
```

### Opción B: Como Env Var (Más simple, menos seguro)
Agregar a GitHub Actions workflow:
```yaml
--set-env-vars=...,FMP_API_KEY=your-fmp-api-key-here
```

### Cambios Necesarios

**File: `.github/workflows/deploy-cloud-run.yml`**
- Agregar `FMP_API_KEY` a `--set-secrets` o `--set-env-vars`
- Si usamos secret: `--set-secrets=FMP_API_KEY=fmp-api-key:latest`

---

## Fix 2: Verificar GEMINI_API_KEY

### Verificar que el secret existe y tiene valor
```bash
gcloud secrets versions access latest --secret=gemini-api-key --project=caria-backend
```

### Verificar que el código puede accederlo
- El código en `analysis.py` usa `os.getenv("GEMINI_API_KEY")`
- El secret está configurado como `GEMINI_API_KEY=gemini-api-key:latest`
- Debería funcionar, pero verificar logs para ver si hay errores

### Posibles problemas
1. Secret no tiene valor válido
2. Permisos incorrectos
3. Código no puede leer el secret en runtime

---

## Fix 3: Arreglar Database Connections

### Problema
Algunos servicios intentan conectarse a hostname "postgres" en lugar de usar Cloud SQL socket.

### Archivos a revisar
- `backend/api/websocket_chat.py` (línea 48): `host=os.getenv("POSTGRES_HOST", "postgres")`
- `backend/api/routes/tactical_allocation.py` (línea 46): Conexión directa a postgres
- Cualquier otro archivo que use `POSTGRES_HOST=postgres`

### Solución
Usar `DATABASE_URL` o Cloud SQL socket en lugar de hostname "postgres".

---

## Plan de Implementación

### Paso 1: Crear/Verificar Secrets
```bash
# Verificar FMP_API_KEY secret
gcloud secrets describe fmp-api-key --project=caria-backend

# Si no existe, crearlo
echo "your-fmp-api-key-here" | gcloud secrets create fmp-api-key --project=caria-backend --data-file=-

# Verificar GEMINI_API_KEY
gcloud secrets versions access latest --secret=gemini-api-key --project=caria-backend
```

### Paso 2: Actualizar GitHub Actions Workflow
- Agregar `FMP_API_KEY` a `--set-secrets`
- Verificar que `GEMINI_API_KEY` esté correctamente configurado

### Paso 3: Arreglar Database Connections
- Modificar `websocket_chat.py` para usar `DATABASE_URL` o Cloud SQL socket
- Modificar `tactical_allocation.py` para usar conexión correcta
- Buscar otros archivos con conexiones hardcodeadas a "postgres"

### Paso 4: Test y Deploy
- Probar endpoints localmente si es posible
- Deploy y verificar que funcionan

---

## Archivos a Modificar

1. `.github/workflows/deploy-cloud-run.yml` - Agregar FMP_API_KEY
2. `backend/api/websocket_chat.py` - Arreglar conexión DB
3. `backend/api/routes/tactical_allocation.py` - Arreglar conexión DB
4. Buscar otros archivos con `POSTGRES_HOST=postgres`

---

## Testing Checklist

- [ ] FMP_API_KEY configurada en Cloud Run
- [ ] Precios funcionan: `/api/prices/realtime/{ticker}`
- [ ] Valuación funciona: `/api/valuation/quick?ticker=AAPL`
- [ ] GEMINI_API_KEY funciona: `/api/analysis/challenge` devuelve análisis real
- [ ] Chat funciona: WebSocket conecta y responde
- [ ] Database connections funcionan: No más errores "could not translate host name postgres"

