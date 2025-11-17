@echo off
REM Script para inicializar Postgres con pgvector para Caria RAG

echo ===================================
echo Caria - Setup PostgreSQL + pgvector
echo ===================================

REM Verificar si Docker Desktop está corriendo
docker ps >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Desktop no esta corriendo.
    echo Por favor, inicia Docker Desktop y ejecuta este script nuevamente.
    pause
    exit /b 1
)

echo [OK] Docker Desktop detectado

REM Verificar si el contenedor ya existe
docker ps -a | findstr "caria-postgres" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Contenedor 'caria-postgres' ya existe. Eliminando...
    docker rm -f caria-postgres
)

echo [1/4] Descargando imagen pgvector...
docker pull ankane/pgvector:latest

echo [2/4] Creando contenedor caria-postgres...
docker run -d ^
  --name caria-postgres ^
  -e POSTGRES_PASSWORD=Theolucas7 ^
  -e POSTGRES_USER=caria_user ^
  -e POSTGRES_DB=caria ^
  -p 5432:5432 ^
  -v caria-pgdata:/var/lib/postgresql/data ^
  ankane/pgvector:latest

echo [3/4] Esperando a que Postgres inicie (15 segundos)...
timeout /t 15 /nobreak

echo [4/4] Creando extension vector y schema rag...
docker exec caria-postgres psql -U caria_user -d caria -c "CREATE EXTENSION IF NOT EXISTS vector;"
docker exec caria-postgres psql -U caria_user -d caria -c "CREATE SCHEMA IF NOT EXISTS rag;"
docker exec caria-postgres psql -U caria_user -d caria -c "GRANT ALL PRIVILEGES ON SCHEMA rag TO caria_user;"

echo.
echo ===================================
echo [SUCCESS] PostgreSQL configurado!
echo ===================================
echo.
echo Detalles de conexion:
echo   Host: localhost
echo   Port: 5432
echo   Database: caria
echo   User: caria_user
echo   Password: Theolucas7
echo.
echo Para probar la conexion:
echo   docker exec -it caria-postgres psql -U caria_user -d caria
echo.
echo Para detener:
echo   docker stop caria-postgres
echo.
echo Para iniciar nuevamente:
echo   docker start caria-postgres
echo.
pause
