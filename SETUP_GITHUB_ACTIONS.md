# 🚀 Configurar GitHub Actions para Despliegue Automático

## Paso 1: Crear Service Account en Google Cloud

Ejecuta estos comandos en Cloud Shell o localmente:

```bash
PROJECT_ID="caria-backend"
SERVICE_ACCOUNT="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"

# Crear service account
gcloud iam service-accounts create github-actions \
    --project=${PROJECT_ID} \
    --display-name="GitHub Actions Service Account"

# Dar permisos necesarios
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor"
```

## Paso 2: Configurar Workload Identity Federation

```bash
PROJECT_ID="caria-backend"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
POOL_NAME="github-pool"
PROVIDER_NAME="github-provider"
SERVICE_ACCOUNT="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"

# Crear Workload Identity Pool
gcloud iam workload-identity-pools create ${POOL_NAME} \
    --project=${PROJECT_ID} \
    --location="global" \
    --display-name="GitHub Actions Pool"

# Crear Provider
gcloud iam workload-identity-pools providers create-oidc ${PROVIDER_NAME} \
    --project=${PROJECT_ID} \
    --location="global" \
    --workload-identity-pool=${POOL_NAME} \
    --display-name="GitHub Provider" \
    --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
    --issuer-uri="https://token.actions.githubusercontent.com"

# Vincular Service Account con el Pool
gcloud iam service-accounts add-iam-policy-binding ${SERVICE_ACCOUNT} \
    --project=${PROJECT_ID} \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_NAME}/attribute.repository/tbasaure-ctrl/caria"
```

## Paso 3: Obtener el Workload Identity Provider

Después de ejecutar los comandos anteriores, obtén el provider:

```bash
gcloud iam workload-identity-pools providers describe ${PROVIDER_NAME} \
    --project=${PROJECT_ID} \
    --location="global" \
    --workload-identity-pool=${POOL_NAME} \
    --format="value(name)"
```

Esto te dará algo como:
```
projects/418525923468/locations/global/workloadIdentityPools/github-pool/providers/github-provider
```

## Paso 4: Actualizar el Workflow

El archivo `.github/workflows/deploy-cloud-run.yml` ya está creado con los valores correctos. Solo necesitas:

1. Actualizar `WORKLOAD_IDENTITY_PROVIDER` con el valor que obtuviste en el Paso 3
2. Subir el archivo a GitHub

## Paso 5: Subir a GitHub

```bash
cd C:\key\wise_adviser_cursor_context\notebooks
git add .github/workflows/deploy-cloud-run.yml
git commit -m "Add GitHub Actions workflow for Cloud Run deployment"
git push origin main
```

## ✅ Verificación

Una vez que hagas push a `main`, GitHub Actions automáticamente:
1. Construirá la imagen Docker
2. La subirá a Artifact Registry
3. Desplegará a Cloud Run

Puedes ver el progreso en: https://github.com/tbasaure-ctrl/caria/actions





