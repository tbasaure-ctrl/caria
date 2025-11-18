#!/bin/bash
# Script COMPLETO para configurar GitHub Actions
# Ejecuta esto en Cloud Shell: bash CONFIGURAR_TODO.sh

set -e

PROJECT_ID="caria-backend"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
POOL_NAME="github-pool"
PROVIDER_NAME="github-provider"
SERVICE_ACCOUNT="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"
GITHUB_REPO="tbasaure-ctrl/caria"

echo "=========================================="
echo "🚀 CONFIGURACIÓN COMPLETA DE GITHUB ACTIONS"
echo "=========================================="
echo ""

# 1. Crear Service Account
echo "1️⃣ Creando Service Account..."
if ! gcloud iam service-accounts describe ${SERVICE_ACCOUNT} --project=${PROJECT_ID} &>/dev/null; then
    gcloud iam service-accounts create github-actions \
        --project=${PROJECT_ID} \
        --display-name="GitHub Actions Service Account"
    echo "   ✅ Service Account creado"
else
    echo "   ✅ Service Account ya existe"
fi

# 2. Dar permisos
echo ""
echo "2️⃣ Configurando permisos..."
for ROLE in "roles/artifactregistry.admin" "roles/run.admin" "roles/iam.serviceAccountUser" "roles/secretmanager.secretAccessor"; do
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="${ROLE}" \
        --condition=None 2>/dev/null && echo "   ✅ ${ROLE}" || echo "   ⚠️  ${ROLE} ya configurado"
done

# 3. Crear Workload Identity Pool
echo ""
echo "3️⃣ Creando Workload Identity Pool..."
if ! gcloud iam workload-identity-pools describe ${POOL_NAME} --project=${PROJECT_ID} --location="global" &>/dev/null; then
    gcloud iam workload-identity-pools create ${POOL_NAME} \
        --project=${PROJECT_ID} \
        --location="global" \
        --display-name="GitHub Actions Pool"
    echo "   ✅ Pool creado"
else
    echo "   ✅ Pool ya existe"
fi

# 4. Crear Provider
echo ""
echo "4️⃣ Creando Provider..."
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
    echo "   ✅ Provider creado"
else
    echo "   ✅ Provider ya existe"
fi

# 5. Vincular Service Account
echo ""
echo "5️⃣ Vinculando Service Account..."
MEMBER="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_NAME}/attribute.repository/${GITHUB_REPO}"

gcloud iam service-accounts add-iam-policy-binding ${SERVICE_ACCOUNT} \
    --project=${PROJECT_ID} \
    --role="roles/iam.workloadIdentityUser" \
    --member="${MEMBER}" \
    --condition=None 2>/dev/null && echo "   ✅ Vinculación creada" || echo "   ✅ Vinculación ya existe"

# 6. Obtener Provider Name
echo ""
echo "6️⃣ Obteniendo información..."
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
echo "📋 Valores importantes:"
echo ""
echo "PROJECT_ID: ${PROJECT_ID}"
echo "PROJECT_NUMBER: ${PROJECT_NUMBER}"
echo "SERVICE_ACCOUNT: ${SERVICE_ACCOUNT}"
echo "WORKLOAD_IDENTITY_PROVIDER: ${PROVIDER_FULL_NAME}"
echo ""
echo "📝 El archivo .github/workflows/deploy-cloud-run.yml ya está configurado"
echo ""
echo "🚀 Próximos pasos:"
echo "1. Haz commit y push a GitHub:"
echo "   git add .github/workflows/deploy-cloud-run.yml"
echo "   git commit -m 'Add GitHub Actions workflow'"
echo "   git push origin main"
echo ""
echo "2. Verifica el despliegue en:"
echo "   https://github.com/${GITHUB_REPO}/actions"
echo ""





