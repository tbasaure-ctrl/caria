@echo off
REM Desplegar con Dockerfile simplificado
REM Ejecuta: DEPLOY_SIMPLE.bat

echo ===========================================
echo Usando Dockerfile simplificado...
echo ===========================================
echo.

REM Backup del Dockerfile actual
if exist services\Dockerfile (
    echo Haciendo backup del Dockerfile actual...
    copy services\Dockerfile services\Dockerfile.backup
)

REM Usar Dockerfile simplificado
if exist services\Dockerfile.SIMPLE (
    echo Usando Dockerfile.SIMPLE...
    copy services\Dockerfile.SIMPLE services\Dockerfile
) else (
    echo ERROR: Dockerfile.SIMPLE no encontrado!
    pause
    exit /b 1
)

REM Construir imagen
echo.
echo Construyendo imagen...
set IMAGE_TAG=us-central1-docker.pkg.dev/caria-backend/caria-api-repo/caria-api:latest
gcloud builds submit --tag %IMAGE_TAG%

if errorlevel 1 (
    echo ERROR: Fallo la construccion
    echo Restaurando Dockerfile original...
    if exist services\Dockerfile.backup (
        copy services\Dockerfile.backup services\Dockerfile
    )
    pause
    exit /b 1
)

echo.
echo Build exitoso! Desplegando...
call DEPLOY_AHORA.bat

