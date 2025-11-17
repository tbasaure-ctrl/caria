#!/bin/bash
# Script completo para desplegar Caria API desde Google Cloud Shell
# Copia y pega este script completo en Cloud Shell

set -e  # Exit on error

echo "=========================================="
echo "🚀 Desplegando Caria API a Cloud Run"
echo "=========================================="
echo ""

# Configuración
PROJECT_ID="caria-backend"
REGION="us-central1"
REPO_NAME="caria-api-repo"
SERVICE_NAME="caria-api"
CLOUDSQL_INSTANCE="caria-backend:us-central1:caria-db"
DB_PASSWORD="Theolucas7"
IMAGE_TAG="us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

# Configurar proyecto
echo "📋 Configurando proyecto..."
gcloud config set project ${PROJECT_ID}
gcloud config set run/region ${REGION}

# Paso 1: Crear repositorio Artifact Registry (si no existe)
echo ""
echo "📦 Paso 1: Verificando repositorio Artifact Registry..."
if ! gcloud artifacts repositories describe ${REPO_NAME} --location=${REGION} --project=${PROJECT_ID} &>/dev/null; then
    echo "   Creando repositorio..."
    gcloud artifacts repositories create ${REPO_NAME} \
        --repository-format=docker \
        --location=${REGION} \
        --description="Caria API Docker images" \
        --project=${PROJECT_ID}
else
    echo "   ✓ Repositorio ya existe"
fi

# Paso 2: Configurar autenticación Docker
echo ""
echo "🔐 Paso 2: Configurando autenticación Docker..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Paso 3: Subir archivos (si no están ya en Cloud Shell)
echo ""
echo "📤 Paso 3: Verificando archivos..."
if [ ! -d "notebooks" ]; then
    echo "   ⚠️  Directorio 'notebooks' no encontrado"
    echo "   Por favor, sube tus archivos usando:"
    echo "   - Click en el ícono de carpeta (☁️) en Cloud Shell"
    echo "   - O usa: gcloud cloud-shell scp local-path cloud-shell:~/notebooks"
    echo ""
    read -p "   Presiona Enter cuando hayas subido los archivos..."
fi

# Paso 4: Construir imagen
echo ""
echo "🔨 Paso 4: Construyendo imagen Docker..."
echo "   Esto puede tomar 15-20 minutos..."
cd notebooks
gcloud builds submit --tag ${IMAGE_TAG}

if [ $? -ne 0 ]; then
    echo "   ❌ Error en build"
    exit 1
fi

echo "   ✓ Imagen construida exitosamente"

# Paso 5: Desplegar a Cloud Run
echo ""
echo "🚀 Paso 5: Desplegando a Cloud Run..."
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
    echo "   Revisa los logs con:"
    echo "   gcloud run services logs read ${SERVICE_NAME} --region=${REGION} --limit=200"
    exit 1
fi

# Paso 6: Obtener URL
echo ""
echo "🌐 Paso 6: Obteniendo URL del servicio..."
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
echo "   2. Prueba el endpoint: curl ${SERVICE_URL}/health"
echo ""
echo "🔍 Ver logs:"
echo "   gcloud run services logs read ${SERVICE_NAME} --region=${REGION} --limit=50"
echo ""

