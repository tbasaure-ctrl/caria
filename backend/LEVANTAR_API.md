# 🚀 Guía para Levantar la API

## ✅ Problemas Resueltos

1. **Paths configurados**: El `app.py` y todos los routes ahora configuran automáticamente los paths para encontrar el módulo `caria`.
2. **Dependencias instaladas**: `pgvector` y `psycopg2-binary` instalados.

## 🎯 Cómo Levantar la API

### Opción 1: Desde el directorio `services/` (Recomendado)

```powershell
# 1. Ir al directorio services
cd C:\key\wise_adviser_cursor_context\notebooks\services

# 2. Levantar API
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Opción 2: Con path absoluto

```powershell
# Desde cualquier directorio
uvicorn C:\key\wise_adviser_cursor_context\notebooks\services\api.app:app --host 0.0.0.0 --port 8000
```

## ✅ Verificación

Una vez levantada la API, deberías ver:

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 🧪 Probar Endpoints

### 1. Healthcheck

```powershell
# PowerShell
Invoke-WebRequest -Uri http://localhost:8000/health | Select-Object -ExpandProperty Content

# O abrir en navegador
# http://localhost:8000/health
```

**Respuesta esperada**:
```json
{
  "status": "ok",
  "rag": "available",
  "regime": "available",
  "factors": "available",
  "valuation": "available"
}
```

### 2. Régimen Macro (Sistema I)

```powershell
Invoke-WebRequest -Uri http://localhost:8000/api/regime/current | Select-Object -ExpandProperty Content
```

### 3. Screening de Factores (Sistema III)

```powershell
$body = @{top_n=10} | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8000/api/factors/screen -Method POST -Body $body -ContentType "application/json" | Select-Object -ExpandProperty Content
```

### 4. Valuación (Sistema IV)

```powershell
Invoke-WebRequest -Uri http://localhost:8000/api/valuation/AAPL | Select-Object -ExpandProperty Content
```

## 🔧 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'caria'`

**Solución**: El `app.py` ahora configura los paths automáticamente. Si persiste:
1. Verifica que `caria_data/src/caria/` existe
2. Verifica que estás ejecutando desde `services/`

### Error: `FileNotFoundError: configs/base.yaml`

**Solución**: El `app.py` busca automáticamente en `caria_data/configs/base.yaml`. Si persiste:
1. Verifica que el archivo existe
2. Usa variable de entorno: `$env:CARIA_SETTINGS_PATH="C:\key\wise_adviser_cursor_context\notebooks\caria_data\configs\base.yaml"`

### Error: Servicio no disponible en healthcheck

**Causas comunes**:
- Modelo HMM no entrenado → `regime: "unavailable"`
- pgvector no configurado → `rag: "disabled"`
- Datos faltantes → servicios pueden fallar

**Solución**: Revisa los logs de la API para ver el error específico.

## 📝 Notas

- La API busca automáticamente `caria_data/src/` y `caria_data/configs/`
- No necesitas configurar PYTHONPATH manualmente
- Los servicios se inicializan automáticamente al levantar la API

