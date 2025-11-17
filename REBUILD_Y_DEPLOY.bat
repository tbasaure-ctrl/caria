@echo off
REM Reconstruir imagen y desplegar
REM Ejecuta: REBUILD_Y_DEPLOY.bat

echo ===========================================
echo Reconstruyendo imagen con Dockerfile corregido...
echo ===========================================
echo.

set IMAGE_TAG=us-central1-docker.pkg.dev/caria-backend/caria-api-repo/caria-api:latest

gcloud builds submit --tag %IMAGE_TAG%

if errorlevel 1 (
    echo ERROR: Fallo la construccion de la imagen
    echo Revisa los logs del build para ver si el directorio models se copio correctamente
    pause
    exit /b 1
)

echo.
echo ===========================================
echo Build exitoso! Ahora desplegando...
echo ===========================================
echo.

call DEPLOY_AHORA.bat

