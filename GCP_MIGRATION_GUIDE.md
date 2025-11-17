# 🚀 Migración a Google Cloud Platform (GCP)

## ¿Por qué GCP?

- ✅ **Integración nativa con Gemini** - Ya estás usando Gemini API
- ✅ **Cloud Run** - Serverless, escala automáticamente, paga por uso
- ✅ **Cloud SQL** - PostgreSQL con soporte para pgvector
- ✅ **Mejor rendimiento** - Red global de Google
- ✅ **Más fácil de mantener** - Menos configuración que Railway

## Arquitectura Propuesta

```
Frontend (Vercel)
    ↓
Backend (Cloud Run) ←→ Cloud SQL (PostgreSQL + pgvector)
    ↓
Gemini API (ya configurado)
```

## 📋 Pasos de Migración

### 1. Preparar Google Cloud Project

```bash
# Instalar Google Cloud SDK si no lo tienes
# https://cloud.google.com/sdk/docs/install

# Login a Google Cloud
gcloud auth login

# Crear proyecto (o usar uno existente)
gcloud projects create caria-backend --name="Caria Backend"

# Seleccionar proyecto
gcloud config set project caria-backend
```

**⚠️ IMPORTANTE: Habilitar Facturación**

Antes de habilitar APIs, necesitas vincular una cuenta de facturación:

1. **Opción A - Desde la consola web:**
   - Ve a: https://console.cloud.google.com/billing
   - Click en "Link a billing account" o "Create billing account"
   - Sigue las instrucciones (puedes usar tarjeta de crédito)
   - **Nota**: GCP tiene $300 de crédito gratis para nuevos usuarios

2. **Opción B - Desde la línea de comandos:**
   ```bash
   # Listar cuentas de facturación disponibles
   gcloud billing accounts list
   
   # Vincular cuenta de facturación al proyecto
   gcloud billing projects link caria-backend --billing-account=BILLING_ACCOUNT_ID
   ```

Una vez habilitada la facturación, continúa:

```bash
# Habilitar APIs necesarias
gcloud services enable \
    run.googleapis.com \
    sqladmin.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    containerregistry.googleapis.com
```

### 2. Crear Cloud SQL (PostgreSQL con pgvector)

```bash
# Crear instancia de Cloud SQL PostgreSQL
gcloud sql instances create caria-db \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=us-central1 \
    --root-password=TU_PASSWORD_SEGURO_AQUI

# Crear base de datos
gcloud sql databases create caria --instance=caria-db

# Obtener la connection string
gcloud sql instances describe caria-db --format="value(connectionName)"
# Guarda esto: proyecto:region:caria-db
```

**Importante**: Cloud SQL PostgreSQL 15+ soporta pgvector nativamente. Solo necesitas habilitarlo:

```sql
-- Conectar a la base de datos y ejecutar:
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Configurar Secret Manager (para API keys)

```bash
# Guardar Gemini API Key
echo -n "tu-gemini-api-key" | gcloud secrets create gemini-api-key --data-file=-

# Guardar Database Password (si no usas connection string)
echo -n "tu-db-password" | gcloud secrets create db-password --data-file=-

# Dar permisos a Cloud Run para acceder a secrets
gcloud secrets add-iam-policy-binding gemini-api-key \
    --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### 4. Construir y Desplegar en Cloud Run

```bash
# Desde el directorio notebooks/
cd /path/to/notebooks

# Construir imagen
gcloud builds submit --tag gcr.io/caria-backend/caria-api

# Desplegar a Cloud Run
gcloud run deploy caria-api \
    --image gcr.io/caria-backend/caria-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars "PORT=8080" \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
    --add-cloudsql-instances proyecto:region:caria-db \
    --set-env-vars "DATABASE_URL=postgresql://user:password@/caria?host=/cloudsql/proyecto:region:caria-db"
```

### 5. Configurar Variables de Entorno en Cloud Run

```bash
# Obtener la URL de Cloud Run después del deploy
CLOUD_RUN_URL=$(gcloud run services describe caria-api --region=us-central1 --format="value(status.url)")

# Configurar CORS origins
gcloud run services update caria-api \
    --region=us-central1 \
    --set-env-vars "CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app"

# Configurar otras variables
gcloud run services update caria-api \
    --region=us-central1 \
    --update-env-vars "RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768"
```

### 6. Actualizar Frontend (Vercel)

1. Ve a Vercel Dashboard → Tu proyecto → Settings → Environment Variables
2. Actualiza `VITE_API_URL` con la URL de Cloud Run:
   ```
   VITE_API_URL=https://caria-api-xxxxx-uc.a.run.app
   ```
3. Redeploy el frontend

## 🔧 Archivos de Configuración

### Dockerfile para Cloud Run

Ya está optimizado en `services/Dockerfile`. Cloud Run usa el puerto `$PORT` automáticamente.

### cloudbuild.yaml (CI/CD)

Crea `cloudbuild.yaml` en la raíz del proyecto para automatizar builds.

### Variables de Entorno Necesarias

```bash
# Database
DATABASE_URL=postgresql://user:password@/caria?host=/cloudsql/proyecto:region:caria-db

# Gemini
GEMINI_API_KEY=tu-key (desde Secret Manager)

# CORS
CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app

# Retrieval
RETRIEVAL_PROVIDER=gemini
RETRIEVAL_EMBEDDING_DIM=768

# Port (Cloud Run lo configura automáticamente)
PORT=8080
```

## 🚀 Comandos Rápidos

### Deploy Inicial Completo

```bash
# 1. Construir y desplegar
gcloud builds submit --tag gcr.io/caria-backend/caria-api

# 2. Deploy a Cloud Run
gcloud run deploy caria-api \
    --image gcr.io/caria-backend/caria-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars "PORT=8080,RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768" \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
    --add-cloudsql-instances proyecto:region:caria-db \
    --set-env-vars "DATABASE_URL=postgresql://user:password@/caria?host=/cloudsql/proyecto:region:caria-db,CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app"
```

### Ver Logs

```bash
gcloud run services logs read caria-api --region=us-central1 --limit=50
```

### Actualizar Despliegue

```bash
# Solo reconstruir y redesplegar
gcloud builds submit --tag gcr.io/caria-backend/caria-api
gcloud run deploy caria-api --image gcr.io/caria-backend/caria-api --region us-central1
```

## 💰 Costos Estimados

- **Cloud Run**: ~$0.40/millón de requests (muy barato para empezar)
- **Cloud SQL (db-f1-micro)**: ~$7.50/mes
- **Cloud Build**: Primeros 120 minutos/día gratis
- **Total estimado**: ~$10-15/mes para empezar

## ✅ Ventajas vs Railway

1. ✅ **Mejor integración con Gemini** - Mismo ecosistema Google
2. ✅ **Cloud SQL con pgvector** - Soporte nativo, más fácil de configurar
3. ✅ **Secret Manager** - Gestión segura de API keys
4. ✅ **Escalado automático** - Cloud Run escala a 0 cuando no hay tráfico
5. ✅ **Logs integrados** - Cloud Logging es muy bueno
6. ✅ **CI/CD nativo** - Cloud Build integrado

## 🔍 Troubleshooting

### Error: "Could not connect to Cloud SQL"

- Verifica que agregaste `--add-cloudsql-instances` al deploy
- Verifica que la connection string usa `/cloudsql/` como host

### Error: "Vector extension not found"

- Conecta a Cloud SQL y ejecuta: `CREATE EXTENSION IF NOT EXISTS vector;`
- O el código lo creará automáticamente (ya está implementado)

### Error: "Secret not found"

- Verifica que el secret existe: `gcloud secrets list`
- Verifica permisos: `gcloud secrets get-iam-policy gemini-api-key`

## 📝 Checklist de Migración

- [ ] Crear proyecto GCP
- [ ] Habilitar APIs necesarias
- [ ] Crear Cloud SQL instance
- [ ] Habilitar extensión pgvector en Cloud SQL
- [ ] Crear secrets en Secret Manager
- [ ] Construir imagen Docker
- [ ] Desplegar a Cloud Run
- [ ] Configurar variables de entorno
- [ ] Probar endpoints
- [ ] Actualizar Vercel con nueva URL
- [ ] Probar login, chat, valuation
- [ ] Monitorear logs y métricas

## 🎯 Próximos Pasos

1. Ejecuta los comandos de setup
2. Despliega la primera versión
3. Prueba los endpoints básicos
4. Migra el tráfico gradualmente
5. Desactiva Railway una vez confirmado que todo funciona

