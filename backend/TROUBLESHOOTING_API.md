# 🔧 Troubleshooting - API

## Problema: Error de Encoding en PostgreSQL

### Error:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf3 in position 85
```

### Causa:
La connection string de PostgreSQL tiene caracteres especiales mal codificados, probablemente en la contraseña o nombre de usuario.

### Solución:

**Opción 1: Usar variables de entorno con encoding correcto**

```powershell
# Configurar variables de entorno con valores simples (sin caracteres especiales)
$env:POSTGRES_USER = "caria_user"
$env:POSTGRES_PASSWORD = "changeme"
$env:POSTGRES_DB = "caria"
$env:POSTGRES_HOST = "localhost"
$env:POSTGRES_PORT = "5432"
```

**Opción 2: Deshabilitar RAG temporalmente**

Si no necesitas RAG ahora, la API funcionará sin él. Los otros endpoints (régimen, factores, valuación) funcionarán normalmente.

**Opción 3: Verificar connection string en configs/base.yaml**

Abre `caria_data/configs/base.yaml` y verifica que la connection string no tenga caracteres especiales:

```yaml
vector_store:
  connection: postgresql://caria_user:changeme@localhost:5432/caria
```

Si tu contraseña tiene caracteres especiales, usa URL encoding:
- `@` → `%40`
- `#` → `%23`
- `$` → `%24`
- etc.

## Problema: Modelo HMM no encontrado

### Error:
```
Modelo HMM no encontrado en models\regime_hmm_model.pkl
```

### Solución:

El modelo está en `caria_data/models/regime_hmm_model.pkl`. El servicio ahora lo busca automáticamente. Si persiste:

1. Verifica que el modelo existe:
```powershell
ls C:\key\wise_adviser_cursor_context\notebooks\caria_data\models\regime_hmm_model.pkl
```

2. Si no existe, entrénalo:
```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data
python scripts/orchestration/run_regime_hmm.py
```

## Estado de la API

La API puede funcionar **parcialmente** sin PostgreSQL o sin modelo HMM:

- ✅ **Sistema III (Factores)**: Funciona siempre (no requiere PostgreSQL ni HMM)
- ✅ **Sistema IV (Valuación)**: Funciona siempre (usa HMM si está disponible, pero tiene fallback)
- ⚠️ **Sistema I (Régimen)**: Requiere modelo HMM entrenado
- ⚠️ **Sistema II (RAG)**: Requiere PostgreSQL con pgvector

## Verificar Estado

```powershell
# Healthcheck muestra estado de cada servicio
Invoke-WebRequest -Uri http://localhost:8000/health | Select-Object -ExpandProperty Content
```

Respuesta esperada:
```json
{
  "status": "ok",
  "rag": "disabled",           // Si PostgreSQL no está disponible
  "regime": "available",        // Si modelo HMM está entrenado
  "factors": "available",       // Siempre disponible
  "valuation": "available"      // Siempre disponible
}
```

## Próximos Pasos

1. **Si PostgreSQL no funciona**: Puedes usar la API sin RAG. Los otros sistemas funcionan.
2. **Si necesitas RAG**: Configura PostgreSQL correctamente o usa una connection string sin caracteres especiales.
3. **Si modelo HMM no está**: Entrénalo con `run_regime_hmm.py` (ya lo hiciste, debería estar).

