# Solución al Error de Importación

## Problema
```
ModuleNotFoundError: No module named 'api'
```

## Causa
Cuando uvicorn ejecuta en modo `--reload`, crea un proceso hijo que no hereda los cambios en `sys.path` del proceso padre.

## Solución Implementada

Se actualizaron dos archivos:

### 1. `app.py`
- Agrega `services/` al `sys.path` ANTES de cualquier import
- Esto permite que Python encuentre el módulo `api`

### 2. `start_api.py`
- Configura `PYTHONPATH` como variable de entorno
- Los procesos hijos de uvicorn heredan esta variable
- También configura `sys.path` para el proceso actual

## Cómo Usar

**SIEMPRE usa el script de inicio:**
```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\services\api
python start_api.py
```

**NO uses uvicorn directamente** a menos que configures PYTHONPATH manualmente:
```powershell
# NO recomendado - puede fallar
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Si lo haces, configura PYTHONPATH primero:
$env:PYTHONPATH = "C:\key\wise_adviser_cursor_context\notebooks\services;C:\key\wise_adviser_cursor_context\notebooks\caria_data\src"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Verificación

Si la API inicia correctamente, deberías ver:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Y deberías poder acceder a:
- http://localhost:8000/docs
- http://localhost:8000/health

