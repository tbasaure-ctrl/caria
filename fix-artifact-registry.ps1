# Script rapido para migrar a Artifact Registry y desplegar
# Ejecuta: .\fix-artifact-registry.ps1

Write-Host "Migrando a Artifact Registry..." -ForegroundColor Blue
Write-Host ""

$PROJECT_ID = "418525923468"
$REGION = "us-central1"
$REPO_NAME = "caria-api-repo"
$SERVICE_NAME = "caria-api"
$CLOUDSQL_INSTANCE = "caria-backend:us-central1:caria-db"

# 1. Crear repositorio en Artifact Registry
Write-Host "1. Creando repositorio en Artifact Registry..." -ForegroundColor Yellow
$createRepo = gcloud artifacts repositories create $REPO_NAME --repository-format=docker --location=$REGION --description="Caria API Docker images" --project=$PROJECT_ID 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "   Repositorio creado" -ForegroundColor Green
} else {
    Write-Host "   Repositorio ya existe o error (continuando...)" -ForegroundColor Yellow
}

# 2. Configurar autenticacion Docker
Write-Host "2. Configurando autenticacion Docker..." -ForegroundColor Yellow
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
Write-Host "   Autenticacion configurada" -ForegroundColor Green

# 3. Construir y push imagen
$IMAGE_TAG = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:latest"
Write-Host "3. Construyendo y subiendo imagen..." -ForegroundColor Yellow
Write-Host "   Imagen: $IMAGE_TAG" -ForegroundColor Cyan

gcloud builds submit --tag $IMAGE_TAG

if ($LASTEXITCODE -ne 0) {
    Write-Host "   Error en build" -ForegroundColor Red
    exit 1
}

Write-Host "   Imagen construida y subida" -ForegroundColor Green

# 4. Obtener password de Cloud SQL
Write-Host ""
Write-Host "4. Configurando Cloud SQL..." -ForegroundColor Yellow
$DB_PASSWORD = Read-Host "   Password de Cloud SQL (root)"
$DATABASE_URL = "postgresql://postgres:$DB_PASSWORD@/caria?host=/cloudsql/$CLOUDSQL_INSTANCE"

# 5. Desplegar a Cloud Run
Write-Host ""
Write-Host "5. Desplegando a Cloud Run..." -ForegroundColor Yellow

gcloud run deploy $SERVICE_NAME --image $IMAGE_TAG --platform managed --region $REGION --allow-unauthenticated --memory 2Gi --cpu 2 --timeout 300 --max-instances 10 --set-env-vars "PORT=8080,RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768" --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" --add-cloudsql-instances $CLOUDSQL_INSTANCE --set-env-vars "DATABASE_URL=$DATABASE_URL,CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Despliegue completado!" -ForegroundColor Green
    Write-Host ""
    
    # Obtener URL del servicio
    $SERVICE_URL = gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"
    Write-Host "URL del servicio: $SERVICE_URL" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Proximos pasos:" -ForegroundColor Yellow
    Write-Host "   1. Actualiza Vercel con: VITE_API_URL=$SERVICE_URL" -ForegroundColor White
    Write-Host "   2. Prueba el endpoint: curl $SERVICE_URL/health" -ForegroundColor White
} else {
    Write-Host "   Error en despliegue" -ForegroundColor Red
    exit 1
}
