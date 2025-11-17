@echo off
REM Desplegar servicio a Cloud Run
REM Ejecuta: DEPLOY_AHORA.bat

echo ===========================================
echo Desplegando servicio a Cloud Run...
echo ===========================================
echo.

set SERVICE_NAME=caria-api
set REGION=us-central1
set IMAGE_TAG=us-central1-docker.pkg.dev/caria-backend/caria-api-repo/caria-api:latest
set CLOUDSQL_INSTANCE=caria-backend:us-central1:caria-db
set DB_PASSWORD=Theolucas7
set DATABASE_URL=postgresql://postgres:%DB_PASSWORD%@/caria?host=/cloudsql/%CLOUDSQL_INSTANCE%

gcloud run deploy %SERVICE_NAME% --image %IMAGE_TAG% --platform managed --region %REGION% --allow-unauthenticated --memory 2Gi --cpu 2 --timeout 300 --max-instances 10 --set-env-vars "RETRIEVAL_PROVIDER=gemini,RETRIEVAL_EMBEDDING_DIM=768" --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" --add-cloudsql-instances %CLOUDSQL_INSTANCE% --set-env-vars "DATABASE_URL=%DATABASE_URL%,CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app"

if errorlevel 1 (
    echo ERROR: Fallo el despliegue
    echo.
    echo Revisa los logs con:
    echo gcloud run services logs read %SERVICE_NAME% --region=%REGION% --limit=200
    pause
    exit /b 1
)

echo.
echo ===========================================
echo Obteniendo URL del servicio...
echo ===========================================
for /f "delims=" %%i in ('gcloud run services describe %SERVICE_NAME% --region=%REGION% --format="value(status.url)"') do set SERVICE_URL=%%i

echo.
echo ===========================================
echo DESPLIEGUE COMPLETADO!
echo ===========================================
echo.
echo URL del servicio: %SERVICE_URL%
echo.
echo Proximos pasos:
echo 1. Actualiza Vercel con: VITE_API_URL=%SERVICE_URL%
echo 2. Prueba el endpoint: curl %SERVICE_URL%/health
echo.
pause

