@echo off
REM Script para subir el workflow a GitHub y activar despliegue
REM Ejecuta: PUSH_A_GITHUB.bat

echo ===========================================
echo Subiendo workflow a GitHub...
echo ===========================================
echo.

REM Verificar que estamos en la rama main
git checkout main

REM Agregar el workflow
echo Agregando workflow...
git add .github/workflows/deploy-cloud-run.yml

REM Commit
echo Haciendo commit...
git commit -m "Add GitHub Actions workflow for Cloud Run deployment"

REM Push (esto activará el despliegue automático)
echo Subiendo a GitHub...
echo Esto activará el despliegue automático...
git push origin main

if errorlevel 1 (
    echo ERROR: Fallo el push
    echo Verifica que tengas acceso al repositorio
    pause
    exit /b 1
)

echo.
echo ===========================================
echo Push completado!
echo ===========================================
echo.
echo GitHub Actions ahora está construyendo y desplegando...
echo.
echo Verifica el progreso en:
echo https://github.com/tbasaure-ctrl/caria/actions
echo.
echo El despliegue tomará aproximadamente 15-20 minutos
echo.
pause





