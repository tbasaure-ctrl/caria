@echo off
REM Script de despliegue para CMD (Command Prompt)
REM Ejecuta: deploy-cmd.bat

echo ===========================================
echo Desplegando Caria API a Google Cloud Run
echo ===========================================
echo.

REM Configurar variables
set PROJECT_ID=caria-backend
set REGION=us-central1
set REPO_NAME=caria-api-repo
set SERVICE_NAME=caria-api
set CLOUDSQL_INSTANCE=caria-backend:us-central1:caria-db
set DB_PASSWORD=Theolucas7
set IMAGE_TAG=us-central1-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%SERVICE_NAME%:latest

echo Paso 1: Crear repositorio Artifact Registry...
gcloud artifacts repositories create %REPO_NAME% --repository-format=docker --location=%REGION% --description="Caria API Docker images" --project=%PROJECT_ID%
if errorlevel 1 (
    echo Repositorio ya existe o error (continuando...)
)

echo.
echo Paso 2: Configurar autenticacion Docker...
gcloud auth configure-docker %REGION%-docker.pkg.dev --quiet

echo.
echo Paso 3: Construir y subir imagen...
echo Imagen: %IMAGE_TAG%
cd /d C:\key\wise_adviser_cursor_context\notebooks
gcloud builds submit --tag %IMAGE_TAG%
if errorlevel 1 (
    echo Error en build
    pause
    exit /b 1
)

echo.
echo Paso 4: Desplegar a Cloud Run...
set DATABASE_URL=postgresql://postgres:%DB_PASSWORD%@/caria?host=/cloudsql/%CLOUDSQL_INSTANCE%

gcloud run deploy %SERVICE_NAME% --image %IMAGE_TAG% --platform managed --region %REGION% --allow-unauthenticated --memory 2Gi --cpu 2 --timeout 300 --max-instances 10 --set-env-vars "PORT=8080,RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768" --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" --add-cloudsql-instances %CLOUDSQL_INSTANCE% --set-env-vars "DATABASE_URL=postgresql://postgres:%DB_PASSWORD%@/caria?host=/cloudsql/%CLOUDSQL_INSTANCE%,CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app"

if errorlevel 1 (
    echo Error en despliegue
    pause
    exit /b 1
)

echo.
echo Paso 5: Obtener URL del servicio...
for /f "delims=" %%i in ('gcloud run services describe %SERVICE_NAME% --region=%REGION% --format="value(status.url)"') do set SERVICE_URL=%%i

echo.
echo ===========================================
echo Despliegue completado!
echo ===========================================
echo.
echo URL del servicio: %SERVICE_URL%
echo.
echo Proximos pasos:
echo 1. Actualiza Vercel con: VITE_API_URL=%SERVICE_URL%
echo 2. Prueba el endpoint: curl %SERVICE_URL%/health
echo.
pause

