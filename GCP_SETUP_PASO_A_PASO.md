# 🚀 Setup GCP - Paso a Paso (Ya habilitaste las APIs)

## ✅ Estado Actual
- ✅ APIs habilitadas
- ⏭️ Siguiente: Crear Cloud SQL

## Paso 1: Crear Cloud SQL (PostgreSQL)

Ejecuta estos comandos desde PowerShell (o desde donde tengas gcloud):

```powershell
# 1. Configurar proyecto (si no lo tienes configurado)
gcloud config set project 418525923468

# 2. Crear instancia de Cloud SQL PostgreSQL
# Esto puede tomar 5-10 minutos
gcloud sql instances create caria-db `
    --database-version=POSTGRES_15 `
    --tier=db-f1-micro `
    --region=us-central1 `
    --root-password=TU_PASSWORD_SEGURO_AQUI

# 3. Crear base de datos
gcloud sql databases create caria --instance=caria-db

# 4. Obtener connection name (lo necesitarás después)
gcloud sql instances describe caria-db --format="value(connectionName)"
```

**Guarda el connection name** que te devuelva (algo como: `proyecto:region:caria-db`)

## Paso 2: Habilitar pgvector en Cloud SQL

```powershell
# Conectar a la base de datos usando Cloud SQL Proxy o desde la consola
# Opción más fácil: Usar la consola web de Cloud SQL

# Ve a: https://console.cloud.google.com/sql/instances/caria-db
# Click en "Open Cloud Shell" o "Connect using Cloud Shell"
# Ejecuta:
```

```sql
-- Conectar a la base de datos caria
\c caria

-- Habilitar extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;
```

**O** el código lo creará automáticamente cuando inicie (ya está implementado).

## Paso 3: Configurar Secret Manager (Gemini API Key)

```powershell
# 1. Crear secret con tu Gemini API Key
# Reemplaza TU_GEMINI_KEY con tu key real
echo -n "TU_GEMINI_KEY" | gcloud secrets create gemini-api-key --data-file=-

# Si el secret ya existe, agregar nueva versión:
echo -n "TU_GEMINI_KEY" | gcloud secrets versions add gemini-api-key --data-file=-

# 2. Obtener PROJECT_NUMBER
$PROJECT_NUMBER = gcloud projects describe 418525923468 --format="value(projectNumber)"

# 3. Dar permisos a Cloud Run para acceder al secret
gcloud secrets add-iam-policy-binding gemini-api-key `
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" `
    --role="roles/secretmanager.secretAccessor"
```

## Paso 4: Construir y Desplegar a Cloud Run

```powershell
# 1. Navegar al directorio del proyecto
cd C:\key\wise_adviser_cursor_context\notebooks

# 2. Construir imagen Docker
gcloud builds submit --tag gcr.io/418525923468/caria-api

# 3. Obtener connection name de Cloud SQL (del Paso 1)
# Debería ser algo como: proyecto:region:caria-db
$CLOUDSQL_INSTANCE = "TU_CONNECTION_NAME_AQUI"

# 4. Crear DATABASE_URL
# Reemplaza PASSWORD con la password que pusiste al crear Cloud SQL
$DB_PASSWORD = "TU_PASSWORD"
$DATABASE_URL = "postgresql://postgres:$DB_PASSWORD@/caria?host=/cloudsql/$CLOUDSQL_INSTANCE"

# 5. Desplegar a Cloud Run
gcloud run deploy caria-api `
    --image gcr.io/418525923468/caria-api:latest `
    --platform managed `
    --region us-central1 `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300 `
    --max-instances 10 `
    --set-env-vars "PORT=8080,RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768" `
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" `
    --add-cloudsql-instances $CLOUDSQL_INSTANCE `
    --set-env-vars "DATABASE_URL=$DATABASE_URL,CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app"

# 6. Obtener URL del servicio
gcloud run services describe caria-api --region=us-central1 --format="value(status.url)"
```

**Guarda la URL** que te devuelva (algo como: `https://caria-api-xxxxx-uc.a.run.app`)

## Paso 5: Actualizar Frontend (Vercel)

1. Ve a: https://vercel.com → Tu proyecto "caria"
2. Settings → Environment Variables
3. Actualiza `VITE_API_URL` con la URL de Cloud Run que obtuviste
4. Redeploy el frontend

## Paso 6: Probar

```powershell
# Probar health endpoint
$API_URL = "TU_URL_DE_CLOUD_RUN"
curl "$API_URL/health"

# Deberías ver algo como:
# {"status":"ok"}
```

## 🔍 Troubleshooting

### Error: "Instance already exists"
- La instancia ya existe, puedes usarla o crear una con otro nombre

### Error: "Permission denied"
- Verifica que tienes permisos de "Owner" o "Editor" en el proyecto

### Error: "Secret not found"
- Verifica que creaste el secret: `gcloud secrets list`

### Error: "Could not connect to Cloud SQL"
- Verifica que agregaste `--add-cloudsql-instances` al deploy
- Verifica que la connection string es correcta

## 📝 Checklist

- [ ] Cloud SQL creado
- [ ] Base de datos "caria" creada
- [ ] pgvector habilitado (o se habilitará automáticamente)
- [ ] Secret Manager configurado con Gemini API Key
- [ ] Imagen Docker construida
- [ ] Desplegado a Cloud Run
- [ ] URL de Cloud Run obtenida
- [ ] Vercel actualizado con nueva URL
- [ ] Probado health endpoint
- [ ] Probado login, chat, valuation

## 🎯 Comandos Rápidos (Copia y Pega)

```powershell
# Setup completo (ajusta las variables)
$PROJECT_ID = "418525923468"
$REGION = "us-central1"
$DB_PASSWORD = "TU_PASSWORD"
$GEMINI_KEY = "TU_GEMINI_KEY"
$CLOUDSQL_INSTANCE = "TU_CONNECTION_NAME"

# Crear Cloud SQL
gcloud sql instances create caria-db --database-version=POSTGRES_15 --tier=db-f1-micro --region=$REGION --root-password=$DB_PASSWORD
gcloud sql databases create caria --instance=caria-db

# Secret Manager
echo -n $GEMINI_KEY | gcloud secrets create gemini-api-key --data-file=-
$PROJECT_NUMBER = gcloud projects describe $PROJECT_ID --format="value(projectNumber)"
gcloud secrets add-iam-policy-binding gemini-api-key --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"

# Build y Deploy
cd C:\key\wise_adviser_cursor_context\notebooks
gcloud builds submit --tag gcr.io/$PROJECT_ID/caria-api
$DATABASE_URL = "postgresql://postgres:$DB_PASSWORD@/caria?host=/cloudsql/$CLOUDSQL_INSTANCE"
gcloud run deploy caria-api --image gcr.io/$PROJECT_ID/caria-api:latest --platform managed --region $REGION --allow-unauthenticated --memory 2Gi --cpu 2 --timeout 300 --max-instances 10 --set-env-vars "PORT=8080,RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768" --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" --add-cloudsql-instances $CLOUDSQL_INSTANCE --set-env-vars "DATABASE_URL=$DATABASE_URL,CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app"
```

