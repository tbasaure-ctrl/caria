@echo off
echo ===========================================
echo Verificando despliegue de Caria API
echo ===========================================
echo.

REM Obtener URL del servicio
echo Obteniendo URL del servicio...
for /f "delims=" %%i in ('gcloud run services describe caria-api --region=us-central1 --format="value(status.url)"') do set SERVICE_URL=%%i

echo.
echo URL del servicio: %SERVICE_URL%
echo.

REM Probar health endpoint
echo Probando endpoint /health...
curl %SERVICE_URL%/health

echo.
echo.
echo ===========================================
echo Proximos pasos:
echo ===========================================
echo 1. Actualiza Vercel con esta URL:
echo    VITE_API_URL=%SERVICE_URL%
echo.
echo 2. Prueba login, chat y valuation desde el frontend
echo.
pause

