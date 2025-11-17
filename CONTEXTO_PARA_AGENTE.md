# Contexto para Continuar el Proyecto Caria

## Estado del Proyecto - 17 Nov 2025

### Situación Actual

El proyecto Caria es una aplicación financiera con:
- **Backend**: FastAPI en Python desplegado en Google Cloud Run
- **Frontend**: React/Vite desplegado en Vercel
- **Base de datos**: PostgreSQL en Google Cloud SQL
- **Biblioteca core**: `caria-lib/` con modelos, servicios, pipelines, etc.

### Reorganización Reciente Completada

Se realizó una reorganización completa de la estructura del proyecto para resolver problemas de deployment:

**ANTES:**
```
notebooks/
├── services/          (API FastAPI)
├── caria_data/
│   ├── src/caria/     (Biblioteca core)
│   └── caria-app/     (Frontend React)
└── src/               (Duplicado, eliminado)
```

**AHORA:**
```
notebooks/
├── backend/           ✅ API FastAPI (renombrado desde services/)
├── caria-lib/         ✅ Biblioteca core (desde caria_data/src/caria/)
├── frontend/          ✅ React app (desde caria_data/caria-app/)
├── backups/src_old/   ✅ Backup del src/ duplicado
└── [otros directorios sin cambios]
```

### Correcciones Realizadas

#### 1. Rutas Hardcodeadas Actualizadas ✅

**Archivos corregidos:**
- `backend/api/dependencies.py` - Actualizado de `services/` y `caria_data/src/` a `backend/` y `caria-lib/`
- `backend/api/app.py` - Actualizado comentarios y búsqueda de configs
- `backend/api/_setup_caria_paths.py` - Actualizado rutas de desarrollo

**Cambios específicos:**
```python
# ANTES:
CARIA_DATA_SRC = CURRENT_FILE.parent.parent.parent / "caria_data" / "src"
DOCKER_CARIA_SRC = Path("/app/caria_data/src")

# AHORA:
CARIA_LIB = CURRENT_FILE.parent.parent.parent / "caria-lib"
DOCKER_CARIA_LIB = Path("/app/caria-lib")
```

#### 2. Dockerfile Verificado ✅

El Dockerfile en `backend/Dockerfile` está correctamente configurado:
- Copia `backend/` → `/app/backend/`
- Copia `caria-lib/` → `/app/caria-lib/`
- PYTHONPATH configurado: `/app/caria-lib:/app/backend`
- Verifica que `caria/models/auth.py` existe antes de iniciar

#### 3. Conexión Cloud SQL Mejorada ✅

En `backend/api/dependencies.py`, se mejoró el manejo de DATABASE_URL:
- Soporte para socket Unix de Cloud SQL: `postgresql://user:pass@/db?host=/cloudsql/instance`
- Manejo de casos edge (hostname None, socket implícito)
- Mejor logging de errores

#### 4. Start Script Actualizado ✅

`backend/start.sh` actualizado con nuevas rutas:
- PYTHONPATH: `/app/caria-lib:/app/backend`
- Verificación de imports antes de iniciar
- Cambio a directorio `/app/backend` antes de ejecutar uvicorn

### Problemas Actuales

#### 1. Deployment Falla ⚠️

El último intento de deployment falló:
```
ERROR: (gcloud.run.deploy) Build failed; check build logs for details
```

**URL de logs del build:**
https://console.cloud.google.com/cloud-build/builds;region=us-central1/e36e0d4f-439a-4977-8dc5-cca205d6019c?project=418525923468

**Comando usado:**
```bash
gcloud run deploy caria-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=10 \
  --set-env-vars "RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768" \
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
  --add-cloudsql-instances "caria-backend:us-central1:caria-db" \
  --set-env-vars "DATABASE_URL=postgresql://postgres:Theolucas7@/caria?host=/cloudsql/caria-backend:us-central1:caria-db,CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app"
```

#### 2. Servicio Actual en Cloud Run

**URL del servicio:** `https://caria-api-418525923468.us-central1.run.app`

**Estado:** El servicio está corriendo pero usando una versión ANTIGUA con rutas antiguas. Los logs muestran:
- Está usando `/app/caria_data/src` y `/app/services` (rutas antiguas)
- Esto significa que el deployment anterior todavía está activo

#### 3. Frontend No Conecta

El frontend en Vercel muestra error:
- "Unable to connect to the server"
- URL: `https://caria-api-418525923468.us-central1.run.app`
- El backend devuelve 404 en rutas básicas

### Tareas Pendientes

#### CRÍTICO - Resolver Ahora

1. **Investigar Error de Build**
   - Revisar logs del build fallido en Cloud Build Console
   - Identificar qué archivo/código está causando el fallo
   - Posibles causas:
     - Archivo faltante en nueva estructura
     - Requisito de dependencia no resuelto
     - Error de sintaxis en código Python
     - Problema con .gcloudignore o .dockerignore

2. **Desplegar Versión Corregida**
   - Una vez corregido el build, desplegar a Cloud Run
   - Verificar logs después del deployment
   - Confirmar que usa `/app/caria-lib` y `/app/backend` (nuevas rutas)

3. **Probar Endpoints Backend**
   - `/health` - Health check general
   - `/health/live` - Liveness probe
   - `/health/ready` - Readiness probe
   - `/api/auth/login` - Endpoint de autenticación
   - `/docs` - Documentación Swagger

#### IMPORTANTE - Después

4. **Configurar Playwright Tests**
   - Crear `tests/e2e/backend.test.ts`
   - Probar endpoints críticos automáticamente
   - Integrar con CI/CD

5. **Configurar GitHub Actions**
   - Crear/actualizar `.github/workflows/deploy-cloud-run.yml`
   - CI/CD automático en push a main
   - Ejecutar tests antes de deployment

6. **Verificar Frontend**
   - Confirmar que `VITE_API_URL` en Vercel apunta a Cloud Run
   - Probar login completo desde frontend
   - Verificar CORS configurado correctamente

### Archivos Clave Modificados

1. `backend/api/dependencies.py` - Rutas actualizadas, Cloud SQL mejorado
2. `backend/api/app.py` - Rutas de configs actualizadas
3. `backend/api/_setup_caria_paths.py` - Rutas de desarrollo actualizadas
4. `backend/Dockerfile` - Ya correcto (usar `backend/` y `caria-lib/`)
5. `backend/start.sh` - Ya correcto (usar `/app/caria-lib` y `/app/backend`)
6. `cloudbuild.yaml` - Ya actualizado para usar `backend/Dockerfile`

### Configuración de Cloud Run

**Proyecto:** `caria-backend`
**Región:** `us-central1`
**Servicio:** `caria-api`
**URL:** `https://caria-api-418525923468.us-central1.run.app`

**Variables de entorno:**
- `RETRIEVAL_PROVIDER=gemini`
- `RETRIEVAL_EMBEDDING_DIM=768`
- `DATABASE_URL=postgresql://postgres:Theolucas7@/caria?host=/cloudsql/caria-backend:us-central1:caria-db`
- `CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app`

**Secrets:**
- `GEMINI_API_KEY` desde `gemini-api-key:latest`

**Cloud SQL:**
- Instancia: `caria-backend:us-central1:caria-db`

### Estructura Actual Verificada

```
notebooks/
├── backend/              ✅ 37,601 archivos (API completa)
│   ├── api/
│   ├── Dockerfile
│   ├── start.sh
│   └── requirements.txt
├── caria-lib/            ✅ 80 archivos (biblioteca core)
│   ├── caria/
│   └── requirements.txt
├── frontend/             ✅ 22,912 archivos (React app)
│   └── caria-app/
├── backups/
│   └── src_old/          ✅ Backup del duplicado
└── cloudbuild.yaml       ✅ Actualizado
```

### Comandos Útiles

**Ver logs del build fallido:**
```bash
gcloud builds list --region=us-central1 --limit=1
gcloud builds log [BUILD_ID] --region=us-central1
```

**Ver logs del servicio actual:**
```bash
gcloud run services logs read caria-api --region=us-central1 --limit=50
```

**Verificar estructura en Cloud Build:**
```bash
# Ver qué está en .gcloudignore
cat .gcloudignore

# Verificar que backend/ y caria-lib/ existen
ls -la backend/ caria-lib/
```

**Probar build local (opcional):**
```bash
docker build -t caria-api-test -f backend/Dockerfile .
```

### Notas Importantes

1. **NO eliminar `backups/src_old/`** - Es un backup de seguridad
2. **NO modificar rutas en Dockerfile o start.sh** - Ya están correctas
3. **Verificar que todos los imports usen `caria.*`** - No rutas absolutas
4. **El servicio actual usa versión vieja** - Necesita redeploy

### Próximos Pasos Sugeridos

1. **INMEDIATO:** Revisar logs del build para identificar error exacto
2. **CRÍTICO:** Corregir el error de build
3. **CRÍTICO:** Desplegar versión corregida y verificar logs
4. **IMPORTANTE:** Probar endpoints manualmente
5. **NICE TO HAVE:** Configurar tests automatizados

### Contacto / Referencias

- Proyecto GCP: `caria-backend` (ID: 418525923468)
- Frontend Vercel: `https://caria-git-main-tomas-projects-70a0592d.vercel.app`
- Backend Cloud Run: `https://caria-api-418525923468.us-central1.run.app`

---

**Última actualización:** 17 Nov 2025
**Estado:** Build fallando, necesita investigación y corrección

