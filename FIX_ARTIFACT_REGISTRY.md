# 🔧 Fix: Migrar a Artifact Registry

## Problema
Container Registry (gcr.io) está deprecado. Necesitas usar Artifact Registry.

## Solución Rápida

### Paso 1: Crear repositorio en Artifact Registry

```powershell
# Crear repositorio Docker en Artifact Registry
gcloud artifacts repositories create caria-api-repo `
    --repository-format=docker `
    --location=us-central1 `
    --description="Caria API Docker images"
```

### Paso 2: Configurar autenticación Docker

```powershell
# Configurar autenticación para Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### Paso 3: Construir y push con Artifact Registry

```powershell
# Construir imagen
gcloud builds submit --tag us-central1-docker.pkg.dev/418525923468/caria-api-repo/caria-api:latest

# O usar Docker directamente:
docker build -t us-central1-docker.pkg.dev/418525923468/caria-api-repo/caria-api:latest -f services/Dockerfile .
docker push us-central1-docker.pkg.dev/418525923468/caria-api-repo/caria-api:latest
```

### Paso 4: Desplegar a Cloud Run con nueva imagen

```powershell
# Desplegar usando Artifact Registry
gcloud run deploy caria-api `
    --image us-central1-docker.pkg.dev/418525923468/caria-api-repo/caria-api:latest `
    --platform managed `
    --region us-central1 `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300 `
    --max-instances 10 `
    --set-env-vars "PORT=8080,RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768" `
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" `
    --add-cloudsql-instances caria-backend:us-central1:caria-db `
    --set-env-vars "DATABASE_URL=postgresql://postgres:TU_PASSWORD@/caria?host=/cloudsql/caria-backend:us-central1:caria-db,CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app"
```

## Comandos Completos (Copia y Pega)

```powershell
# 1. Crear repositorio
gcloud artifacts repositories create caria-api-repo --repository-format=docker --location=us-central1 --description="Caria API Docker images"

# 2. Configurar autenticación
gcloud auth configure-docker us-central1-docker.pkg.dev

# 3. Construir y push
gcloud builds submit --tag us-central1-docker.pkg.dev/418525923468/caria-api-repo/caria-api:latest

# 4. Desplegar (ajusta DATABASE_URL con tu password)
$DB_PASSWORD = "TU_PASSWORD"
$DATABASE_URL = "postgresql://postgres:$DB_PASSWORD@/caria?host=/cloudsql/caria-backend:us-central1:caria-db"

gcloud run deploy caria-api `
    --image us-central1-docker.pkg.dev/418525923468/caria-api-repo/caria-api:latest `
    --platform managed `
    --region us-central1 `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300 `
    --max-instances 10 `
    --set-env-vars "PORT=8080,RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768" `
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" `
    --add-cloudsql-instances caria-backend:us-central1:caria-db `
    --set-env-vars "DATABASE_URL=$DATABASE_URL,CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app"
```

