#!/bin/bash
# Script de despliegue rápido a Google Cloud Run
# Uso: ./deploy-gcp.sh

set -e

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Desplegando Caria API a Google Cloud Run${NC}"

# Verificar que gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI no está instalado. Instálalo desde: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# Obtener proyecto actual
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ No hay proyecto GCP configurado. Ejecuta: gcloud config set project TU_PROYECTO${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Proyecto: $PROJECT_ID${NC}"

# Obtener región
REGION=${REGION:-us-central1}
echo -e "${GREEN}✓ Región: $REGION${NC}"

# Nombre del servicio
SERVICE_NAME="caria-api"
echo -e "${GREEN}✓ Servicio: $SERVICE_NAME${NC}"

# Construir imagen usando Artifact Registry (gcr.io está deprecado)
ARTIFACT_REGISTRY_REPO=${ARTIFACT_REGISTRY_REPO:-caria-api-repo}
ARTIFACT_REGISTRY_LOCATION=${ARTIFACT_REGISTRY_LOCATION:-us-central1}
IMAGE_TAG="us-central1-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_REPO/$SERVICE_NAME:latest"

echo -e "\n${BLUE}📦 Construyendo imagen Docker...${NC}"
echo -e "${YELLOW}   Usando Artifact Registry: $IMAGE_TAG${NC}"
gcloud builds submit \
    --tag $IMAGE_TAG \
    .

# Verificar si Cloud SQL instance está configurada
CLOUDSQL_INSTANCE=${CLOUDSQL_INSTANCE:-""}
if [ -z "$CLOUDSQL_INSTANCE" ]; then
    echo -e "${RED}⚠️  CLOUDSQL_INSTANCE no está configurado${NC}"
    echo -e "${BLUE}   Configúralo así: export CLOUDSQL_INSTANCE=proyecto:region:caria-db${NC}"
    echo -e "${BLUE}   O edita este script para agregarlo${NC}"
fi

# Variables de entorno
DATABASE_URL=${DATABASE_URL:-""}
CORS_ORIGINS=${CORS_ORIGINS:-"https://caria-git-main-tomas-projects-70a0592d.vercel.app"}

# Deploy a Cloud Run
echo -e "\n${BLUE}🚀 Desplegando a Cloud Run...${NC}"

DEPLOY_ARGS=(
    "run" "deploy" "$SERVICE_NAME"
    "--image" "$IMAGE_TAG"
    "--platform" "managed"
    "--region" "$REGION"
    "--allow-unauthenticated"
    "--memory" "2Gi"
    "--cpu" "2"
    "--timeout" "300"
    "--max-instances" "10"
    "--set-env-vars" "PORT=8080,RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768"
    "--set-secrets" "GEMINI_API_KEY=gemini-api-key:latest"
)

# Agregar Cloud SQL si está configurado
if [ ! -z "$CLOUDSQL_INSTANCE" ]; then
    DEPLOY_ARGS+=("--add-cloudsql-instances" "$CLOUDSQL_INSTANCE")
fi

# Agregar variables de entorno
ENV_VARS="PORT=8080,RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768"
if [ ! -z "$DATABASE_URL" ]; then
    ENV_VARS="$ENV_VARS,DATABASE_URL=$DATABASE_URL"
fi
if [ ! -z "$CORS_ORIGINS" ]; then
    ENV_VARS="$ENV_VARS,CORS_ORIGINS=$CORS_ORIGINS"
fi

DEPLOY_ARGS+=("--set-env-vars" "$ENV_VARS")

gcloud "${DEPLOY_ARGS[@]}"

# Obtener URL del servicio
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo -e "\n${GREEN}✅ Despliegue completado!${NC}"
echo -e "${GREEN}🌐 URL: $SERVICE_URL${NC}"
echo -e "\n${BLUE}📝 Próximos pasos:${NC}"
echo -e "   1. Actualiza Vercel con: VITE_API_URL=$SERVICE_URL"
echo -e "   2. Prueba el endpoint: curl $SERVICE_URL/health"
echo -e "   3. Ver logs: gcloud run services logs read $SERVICE_NAME --region=$REGION"

