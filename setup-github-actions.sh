#!/bin/bash
# Script para configurar GitHub Actions con Workload Identity Federation
# Ejecuta esto en Cloud Shell

set -e

PROJECT_ID="caria-backend"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
POOL_NAME="github-pool"
PROVIDER_NAME="github-provider"
SERVICE_ACCOUNT="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"
GITHUB_REPO="tbasaure-ctrl/caria"

echo "=========================================="
echo "🔧 Configurando GitHub Actions"
echo "=========================================="
echo ""

# Paso 1: Crear Service Account
echo "📋 Paso 1: Creando Service Account..."
if ! gcloud iam service-accounts describe ${SERVICE_ACCOUNT} --project=${PROJECT_ID} &>/dev/null; then
    gcloud iam service-accounts create github-actions \
        --project=${PROJECT_ID} \
        --display-name="GitHub Actions Service Account"
    echo "   ✓ Service Account creado"
else
    echo "   ✓ Service Account ya existe"
fi

# Paso 2: Dar permisos
echo ""
echo "🔐 Paso 2: Configurando permisos..."
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/artifactregistry.admin" \
    --condition=None 2>/dev/null || echo "   Permiso ya otorgado"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/run.admin" \
    --condition=None 2>/dev/null || echo "   Permiso ya otorgado"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/iam.serviceAccountUser" \
    --condition=None 2>/dev/null || echo "   Permiso ya otorgado"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None 2>/dev/null || echo "   Permiso ya otorgado"

echo "   ✓ Permisos configurados"

# Paso 3: Crear Workload Identity Pool
echo ""
echo "🏊 Paso 3: Creando Workload Identity Pool..."
if ! gcloud iam workload-identity-pools describe ${POOL_NAME} --project=${PROJECT_ID} --location="global" &>/dev/null; then
    gcloud iam workload-identity-pools create ${POOL_NAME} \
        --project=${PROJECT_ID} \
        --location="global" \
        --display-name="GitHub Actions Pool"
    echo "   ✓ Pool creado"
else
    echo "   ✓ Pool ya existe"
fi

# Paso 4: Crear Provider
echo ""
echo "🔗 Paso 4: Creando Provider..."
if ! gcloud iam workload-identity-pools providers describe ${PROVIDER_NAME} \
    --project=${PROJECT_ID} \
    --location="global" \
    --workload-identity-pool=${POOL_NAME} &>/dev/null; then
    
    gcloud iam workload-identity-pools providers create-oidc ${PROVIDER_NAME} \
        --project=${PROJECT_ID} \
        --location="global" \
        --workload-identity-pool=${POOL_NAME} \
        --display-name="GitHub Provider" \
        --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
        --issuer-uri="https://token.actions.githubusercontent.com"
    echo "   ✓ Provider creado"
else
    echo "   ✓ Provider ya existe"
fi

# Paso 5: Vincular Service Account
echo ""
echo "🔗 Paso 5: Vinculando Service Account..."
gcloud iam service-accounts add-iam-policy-binding ${SERVICE_ACCOUNT} \
    --project=${PROJECT_ID} \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_NAME}/attribute.repository/${GITHUB_REPO}" \
    --condition=None 2>/dev/null || echo "   Vinculación ya existe"

echo "   ✓ Service Account vinculado"

# Paso 6: Obtener Provider Name
echo ""
echo "📝 Paso 6: Obteniendo información del Provider..."
PROVIDER_FULL_NAME=$(gcloud iam workload-identity-pools providers describe ${PROVIDER_NAME} \
    --project=${PROJECT_ID} \
    --location="global" \
    --workload-identity-pool=${POOL_NAME} \
    --format="value(name)")

echo ""
echo "=========================================="
echo "✅ CONFIGURACIÓN COMPLETADA!"
echo "=========================================="
echo ""
echo "📋 Información importante:"
echo ""
echo "PROJECT_ID: ${PROJECT_ID}"
echo "PROJECT_NUMBER: ${PROJECT_NUMBER}"
echo "SERVICE_ACCOUNT: ${SERVICE_ACCOUNT}"
echo "WORKLOAD_IDENTITY_PROVIDER: ${PROVIDER_FULL_NAME}"
echo ""
echo "📝 Próximos pasos:"
echo "1. El archivo .github/workflows/deploy-cloud-run.yml ya está configurado"
echo "2. Haz commit y push a GitHub:"
echo "   git add .github/workflows/deploy-cloud-run.yml"
echo "   git commit -m 'Add GitHub Actions workflow'"
echo "   git push origin main"
echo ""
echo "3. Verifica el despliegue en:"
echo "   https://github.com/${GITHUB_REPO}/actions"
echo ""








