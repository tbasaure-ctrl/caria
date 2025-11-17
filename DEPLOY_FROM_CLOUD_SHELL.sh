#!/bin/bash
# Script completo para desplegar desde Cloud Shell
# Copia y pega esto completo en Cloud Shell

set -e

echo "🚀 Iniciando despliegue de Caria API..."

# Configuración
PROJECT_ID="caria-backend"
REGION="us-central1"
REPO_NAME="caria-api-repo"
SERVICE_NAME="caria-api"
CLOUDSQL_INSTANCE="caria-backend:us-central1:caria-db"
DB_PASSWORD="Theolucas7"

# Configurar proyecto
echo "📋 Configurando proyecto..."
gcloud config set project ${PROJECT_ID}
gcloud config set run/region ${REGION}

# Clonar repositorio
echo "📥 Clonando repositorio..."
cd ~
if [ -d "caria" ]; then
    echo "   Repositorio ya existe, actualizando..."
    cd caria
    git pull
else
    git clone https://github.com/tbasaure-ctrl/caria.git
    cd caria
fi

# Verificar estructura
echo "🔍 Verificando estructura..."
if [ ! -d "services" ] || [ ! -d "caria_data" ]; then
    echo "   ⚠️  Estructura incorrecta. Buscando directorio correcto..."
    if [ -d "notebooks" ]; then
        cd notebooks
    fi
fi

echo "   Directorio actual: $(pwd)"
echo "   Contenido: $(ls -la | head -10)"

# Crear repositorio Artifact Registry
echo "📦 Creando repositorio Artifact Registry..."
gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --location=${REGION} \
    --description="Caria API Docker images" \
    --project=${PROJECT_ID} 2>/dev/null || echo "   ✓ Repositorio ya existe"

# Configurar Docker
echo "🔐 Configurando Docker..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Construir imagen
echo "🔨 Construyendo imagen Docker..."
echo "   Esto puede tomar 15-20 minutos..."
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"
gcloud builds submit --tag ${IMAGE_TAG}

if [ $? -ne 0 ]; then
    echo "   ❌ Error en build"
    exit 1
fi

echo "   ✓ Imagen construida exitosamente"

# Desplegar
echo "🚀 Desplegando a Cloud Run..."
DATABASE_URL="postgresql://postgres:${DB_PASSWORD}@/caria?host=/cloudsql/${CLOUDSQL_INSTANCE}"

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_TAG} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars "RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768" \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
    --add-cloudsql-instances ${CLOUDSQL_INSTANCE} \
    --set-env-vars "DATABASE_URL=${DATABASE_URL},CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app"

if [ $? -ne 0 ]; then
    echo "   ❌ Error en despliegue"
    echo ""
    echo "   Revisa los logs:"
    echo "   gcloud run services logs read ${SERVICE_NAME} --region=${REGION} --limit=200"
    exit 1
fi

# Obtener URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo ""
echo "=========================================="
echo "✅ DESPLIEGUE COMPLETADO!"
echo "=========================================="
echo ""
echo "📍 URL del servicio: ${SERVICE_URL}"
echo ""
echo "📝 Próximos pasos:"
echo "   1. Actualiza Vercel con: VITE_API_URL=${SERVICE_URL}"
echo "   2. Prueba: curl ${SERVICE_URL}/health"
echo ""

