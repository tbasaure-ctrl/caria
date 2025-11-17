#!/bin/bash
# Script de setup inicial para Google Cloud Platform
# Uso: ./setup-gcp.sh

set -e

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🔧 Configurando Google Cloud Platform para Caria${NC}\n"

# Verificar que gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI no está instalado${NC}"
    echo -e "${YELLOW}   Instálalo desde: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# Login si es necesario
echo -e "${BLUE}1. Verificando autenticación...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}   No hay sesión activa. Iniciando login...${NC}"
    gcloud auth login
fi
echo -e "${GREEN}✓ Autenticado${NC}\n"

# Crear o seleccionar proyecto
echo -e "${BLUE}2. Configurando proyecto...${NC}"
read -p "   ¿Crear nuevo proyecto? (y/n): " CREATE_PROJECT
if [ "$CREATE_PROJECT" = "y" ]; then
    read -p "   Nombre del proyecto: " PROJECT_NAME
    PROJECT_ID=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
    gcloud projects create $PROJECT_ID --name="$PROJECT_NAME" || true
    gcloud config set project $PROJECT_ID
else
    read -p "   ID del proyecto existente: " PROJECT_ID
    gcloud config set project $PROJECT_ID
fi
echo -e "${GREEN}✓ Proyecto: $PROJECT_ID${NC}\n"

# Verificar facturación
echo -e "${BLUE}3. Verificando facturación...${NC}"
BILLING_ENABLED=$(gcloud billing projects describe $PROJECT_ID --format="value(billingEnabled)" 2>/dev/null || echo "false")
if [ "$BILLING_ENABLED" != "true" ]; then
    echo -e "${RED}❌ Facturación no habilitada${NC}"
    echo -e "${YELLOW}   Necesitas habilitar facturación antes de continuar${NC}"
    echo -e "${YELLOW}   Opciones:${NC}"
    echo -e "${YELLOW}   1. Ve a: https://console.cloud.google.com/billing${NC}"
    echo -e "${YELLOW}   2. Crea una cuenta de facturación y vincúlala a este proyecto${NC}"
    echo -e "${YELLOW}   3. O ejecuta: gcloud billing projects link $PROJECT_ID --billing-account=BILLING_ACCOUNT_ID${NC}"
    echo -e "${YELLOW}${NC}"
    echo -e "${YELLOW}   Ver GCP_BILLING_SETUP.md para más detalles${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Facturación habilitada${NC}\n"

# Habilitar APIs
echo -e "${BLUE}4. Habilitando APIs necesarias...${NC}"
gcloud services enable \
    run.googleapis.com \
    sqladmin.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    containerregistry.googleapis.com \
    --project=$PROJECT_ID
echo -e "${GREEN}✓ APIs habilitadas${NC}\n"

# Crear Cloud SQL
echo -e "${BLUE}5. Configurando Cloud SQL...${NC}"
read -p "   ¿Crear nueva instancia de Cloud SQL? (y/n): " CREATE_DB
if [ "$CREATE_DB" = "y" ]; then
    read -p "   Nombre de la instancia (default: caria-db): " DB_INSTANCE
    DB_INSTANCE=${DB_INSTANCE:-caria-db}
    
    read -p "   Región (default: us-central1): " REGION
    REGION=${REGION:-us-central1}
    
    read -sp "   Password para root: " DB_PASSWORD
    echo ""
    
    echo -e "${YELLOW}   Creando instancia (esto puede tomar varios minutos)...${NC}"
    gcloud sql instances create $DB_INSTANCE \
        --database-version=POSTGRES_15 \
        --tier=db-f1-micro \
        --region=$REGION \
        --root-password=$DB_PASSWORD \
        --project=$PROJECT_ID || echo -e "${YELLOW}   Instancia ya existe o error${NC}"
    
    # Crear base de datos
    gcloud sql databases create caria --instance=$DB_INSTANCE --project=$PROJECT_ID || true
    
    # Obtener connection name
    CONNECTION_NAME=$(gcloud sql instances describe $DB_INSTANCE --format="value(connectionName)" --project=$PROJECT_ID)
    echo -e "${GREEN}✓ Cloud SQL creado: $CONNECTION_NAME${NC}"
else
    read -p "   Nombre de la instancia existente: " DB_INSTANCE
    CONNECTION_NAME=$(gcloud sql instances describe $DB_INSTANCE --format="value(connectionName)" --project=$PROJECT_ID)
    echo -e "${GREEN}✓ Usando instancia existente: $CONNECTION_NAME${NC}"
fi
echo ""

# Habilitar pgvector
echo -e "${BLUE}6. Habilitando extensión pgvector...${NC}"
echo -e "${YELLOW}   Conecta a tu base de datos y ejecuta:${NC}"
echo -e "${YELLOW}   CREATE EXTENSION IF NOT EXISTS vector;${NC}"
echo -e "${YELLOW}   O el código lo creará automáticamente al iniciar${NC}\n"

# Configurar Secret Manager
echo -e "${BLUE}7. Configurando Secret Manager...${NC}"
read -p "   ¿Tienes una Gemini API Key? (y/n): " HAS_KEY
if [ "$HAS_KEY" = "y" ]; then
    read -sp "   Gemini API Key: " GEMINI_KEY
    echo ""
    
    # Crear secret
    echo -n "$GEMINI_KEY" | gcloud secrets create gemini-api-key \
        --data-file=- \
        --project=$PROJECT_ID 2>/dev/null || \
    echo -n "$GEMINI_KEY" | gcloud secrets versions add gemini-api-key \
        --data-file=- \
        --project=$PROJECT_ID
    
    # Dar permisos a Cloud Run
    PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
    gcloud secrets add-iam-policy-binding gemini-api-key \
        --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
        --role="roles/secretmanager.secretAccessor" \
        --project=$PROJECT_ID
    
    echo -e "${GREEN}✓ Secret creado${NC}"
else
    echo -e "${YELLOW}   Obtén una key en: https://makersuite.google.com/app/apikey${NC}"
    echo -e "${YELLOW}   Luego ejecuta:${NC}"
    echo -e "${YELLOW}   echo -n 'tu-key' | gcloud secrets create gemini-api-key --data-file=-${NC}"
fi
echo ""

# Resumen
echo -e "${GREEN}✅ Configuración completada!${NC}\n"
echo -e "${BLUE}📝 Variables para usar:${NC}"
echo -e "   PROJECT_ID=$PROJECT_ID"
echo -e "   REGION=$REGION"
echo -e "   CLOUDSQL_INSTANCE=$CONNECTION_NAME"
echo -e "   DATABASE_URL=postgresql://postgres:PASSWORD@/caria?host=/cloudsql/$CONNECTION_NAME"
echo -e "\n${BLUE}🚀 Próximo paso:${NC}"
echo -e "   ./deploy-gcp.sh"
echo -e "\n${YELLOW}   O configura estas variables y ejecuta:${NC}"
echo -e "   export CLOUDSQL_INSTANCE=$CONNECTION_NAME"
echo -e "   export DATABASE_URL=postgresql://postgres:PASSWORD@/caria?host=/cloudsql/$CONNECTION_NAME"
echo -e "   ./deploy-gcp.sh"

