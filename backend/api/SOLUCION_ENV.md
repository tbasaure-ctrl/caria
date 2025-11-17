# Solución: Variables de Entorno

## Problema
Cuando ejecutas `uvicorn app:app` directamente, no carga automáticamente el archivo `.env`, causando errores como:
- `POSTGRES_PASSWORD environment variable is required`
- `Error de encoding al conectar a PostgreSQL`

## Solución Implementada

### 1. Carga automática de .env en `app.py`
El archivo `app.py` ahora carga automáticamente el archivo `.env` al inicio, antes de cualquier otra operación.

### 2. Carga automática en `dependencies.py`
El archivo `dependencies.py` también carga el `.env` como respaldo cuando se necesita la conexión a la base de datos.

### 3. Script para actualizar contraseña
Ejecuta:
```powershell
python update_env_password.py
```

Esto actualiza el archivo `.env` con la contraseña correcta.

## Cómo Usar

### Opción 1: Usar el script de inicio (Recomendado)
```powershell
python start_api.py
```
Este script carga el `.env` automáticamente.

### Opción 2: Uvicorn directo (Ahora funciona)
```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Ahora también carga el `.env` automáticamente gracias a los cambios en `app.py`.

### Opción 3: Configurar variables manualmente
```powershell
$env:POSTGRES_PASSWORD='Theolucas7'
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Verificación

Después de reiniciar la API, deberías ver:
- ✅ No más errores de "POSTGRES_PASSWORD environment variable is required"
- ✅ No más errores de encoding
- ✅ Health check muestra `"database": "available"`
- ✅ Puedes registrar usuarios en `/api/auth/register`

## Archivo .env

El archivo `.env` está en `services/api/.env` y contiene:
```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=caria_user
POSTGRES_PASSWORD=Theolucas7
POSTGRES_DB=caria
FMP_API_KEY=79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq
```

**Nota**: El archivo `.env` está en `.gitignore` por seguridad, así que no se subirá al repositorio.

