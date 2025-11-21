# Guía de Lanzamiento - CARIA con Monte Carlo y Precios en Tiempo Real

## ✅ Funcionalidades Implementadas

### 1. Monte Carlo Valuation
- ✅ Sistema de presets por industria/etapa
- ✅ Integración con DCF y múltiplos
- ✅ Endpoint `/api/valuation/{ticker}/monte-carlo`
- ✅ Visualizaciones (histograma y paths) en base64
- ✅ Parámetros personalizables

### 2. Precios en Tiempo Real
- ✅ Métodos en FMPClient para precios en tiempo real
- ✅ Endpoint `/api/prices/realtime` (batch)
- ✅ Endpoint `/api/prices/realtime/{ticker}` (individual)

### 3. Sistema de Holdings
- ✅ Tabla `holdings` en base de datos
- ✅ Endpoints CRUD completos:
  - `GET /api/holdings` - Listar holdings
  - `POST /api/holdings` - Crear/actualizar holding
  - `DELETE /api/holdings/{id}` - Eliminar holding
  - `GET /api/holdings/with-prices` - Holdings con precios en tiempo real

## 🚀 Pasos para Lanzar

### Paso 1: Configurar Variables de Entorno

**Opción A: Script Automático (Recomendado)**

```bash
cd services/api
python setup_env.py
# Edita el archivo .env generado y configura POSTGRES_PASSWORD
```

**Opción B: Manual**

Crea un archivo `.env` en `services/api/` con:

```bash
# Base de datos PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=caria_user
POSTGRES_PASSWORD=tu_password_aqui
POSTGRES_DB=caria

# FMP API Key (para precios en tiempo real) - YA CONFIGURADA
FMP_API_KEY=your-fmp-api-key-here

# Gemini API Key (opcional - para RAG/chat, Llama será backup si no está configurada)
GEMINI_API_KEY=tu_gemini_api_key_aqui

# JWT Secret Key (se genera automáticamente si usas el script)
JWT_SECRET_KEY=tu_secret_key_seguro_aqui

# CORS Origins
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

Luego carga las variables:

```bash
# Linux/Mac
export $(cat .env | xargs)

# Windows PowerShell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
```

**Nota**: 
- La API key de FMP (`your-fmp-api-key-here`) ya está configurada y se usará automáticamente para los precios en tiempo real.
- `GEMINI_API_KEY` es opcional. Si no está configurada, el sistema usará Llama (Ollama) como backup automáticamente.

### Paso 2: Generar Archivos de Datos Requeridos

Antes de iniciar la API, asegúrate de que todos los archivos de datos requeridos existan:

**Generar macro_features.parquet**:

```bash
cd caria_data
python scripts/generate_macro_features.py
```

Este script:
- Busca `fred_data.parquet` automáticamente
- Genera `macro_features.parquet` con todas las features necesarias para el modelo HMM
- Guarda el archivo en la ubicación correcta según la configuración

**Verificar que todos los archivos existan**:

```bash
cd caria_data
python scripts/verify_data_files.py
```

Este script verifica:
- ✅ `macro_features.parquet` (requerido para régimen)
- ✅ `quality_signals.parquet` (requerido para factor screening)
- ✅ `value_signals.parquet` (requerido para factor screening)
- ✅ `momentum_signals.parquet` (requerido para factor screening)
- ✅ `regime_hmm_model.pkl` (requerido para detección de régimen)

Si algún archivo falta, el script te indicará qué hacer.

### Paso 3: Ejecutar Migración de Base de Datos

Ejecuta la migración para crear la tabla de holdings:

**Opción A: Script Python (Recomendado - Más fácil)**

```bash
cd services/api

# Configurar contraseña primero (elige una opción):
# PowerShell:
$env:POSTGRES_PASSWORD='tu_password_aqui'

# O pasarla como argumento:
python run_migration.py --password tu_password_aqui

# O el script te pedirá la contraseña interactivamente
python run_migration.py
```

**Opción B: Desde psql**

```bash
psql -U caria_user -d caria -f caria_data/infrastructure/migrations/add_holdings_table.sql
# Te pedirá la contraseña
```

**Opción C: Desde Python directo**

```bash
# Primero configura la contraseña:
# PowerShell:
$env:POSTGRES_PASSWORD='tu_password_aqui'

# Luego ejecuta:
python -c "
import psycopg2
import os
from pathlib import Path

conn = psycopg2.connect(
    host=os.getenv('POSTGRES_HOST', 'localhost'),
    port=int(os.getenv('POSTGRES_PORT', '5432')),
    user=os.getenv('POSTGRES_USER', 'caria_user'),
    password=os.getenv('POSTGRES_PASSWORD'),
    database=os.getenv('POSTGRES_DB', 'caria')
)
migration_file = Path('caria_data/infrastructure/migrations/add_holdings_table.sql')
with open(migration_file, 'r') as f:
    conn.cursor().execute(f.read())
conn.commit()
conn.close()
print('Migración exitosa!')
"
```

### Paso 3: Verificar que la Base de Datos Está Inicializada

Asegúrate de que la tabla `users` existe (de la inicialización previa):

```sql
-- Verificar que existe
SELECT * FROM users LIMIT 1;

-- Si no existe, ejecutar init_db.sql completo
```

### Paso 4: Instalar Dependencias

```bash
# Backend (API)
cd services/api
pip install -r requirements.txt

# Frontend (si vas a actualizar la UI)
cd caria_data/caria-app
npm install
```

### Paso 5: Iniciar la API

**Opción A: Script de inicio (Recomendado)**

```bash
cd services/api
python start_api.py
```

Este script:
- ✅ Carga automáticamente variables de `.env` si existe
- ✅ Verifica que las variables críticas estén configuradas
- ✅ Muestra estado de configuración antes de iniciar

**Opción B: Uvicorn directo**

```bash
cd services/api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Nota**: Si tienes errores de importación, asegúrate de estar en el directorio `services/api` cuando ejecutes uvicorn.

### Paso 6: Verificar que Todo Funciona

#### Probar Monte Carlo Valuation:
```bash
curl -X POST "http://localhost:8000/api/valuation/AAPL/monte-carlo" \
  -H "Authorization: Bearer TU_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "n_paths": 10000,
    "country_risk": "low"
  }'
```

#### Probar Precios en Tiempo Real:
```bash
curl -X POST "http://localhost:8000/api/prices/realtime" \
  -H "Authorization: Bearer TU_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL"]
  }'
```

#### Probar Holdings:
```bash
# Crear un holding
curl -X POST "http://localhost:8000/api/holdings" \
  -H "Authorization: Bearer TU_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "quantity": 10,
    "average_cost": 150.0,
    "notes": "Mi primera posición"
  }'

# Obtener holdings con precios
curl "http://localhost:8000/api/holdings/with-prices" \
  -H "Authorization: Bearer TU_TOKEN"
```

## 📝 Próximos Pasos (Opcional - UI)

Para actualizar la UI y mostrar precios reales:

1. **Actualizar `MarketIndices.tsx`** para llamar a `/api/prices/realtime` con índices principales
2. **Actualizar `Portfolio.tsx`** para llamar a `/api/holdings/with-prices`
3. **Agregar componente** para gestionar holdings (agregar/editar/eliminar)

## 🔍 Verificación de Salud

Verifica que todos los servicios están funcionando:

```bash
curl http://localhost:8000/health
```

Deberías ver:
```json
{
  "status": "ok",
  "database": "available",
  "auth": "available",
  "valuation": "available",
  ...
}
```

## ⚠️ Notas Importantes

1. **FMP API Key**: Asegúrate de tener una API key válida de FMP. El plan gratuito tiene límites de requests.

2. **Rate Limiting**: Los endpoints tienen rate limiting configurado. Si necesitas más requests, ajusta en `dependencies.py`.

3. **Base de Datos**: La tabla `holdings` se crea automáticamente con la migración. Si ya existe, la migración es idempotente.

4. **Monte Carlo**: Las simulaciones pueden tardar unos segundos (especialmente con 10,000 paths). Considera usar menos paths para desarrollo.

5. **Visualizaciones**: Las imágenes se retornan como base64. En producción, considera guardarlas en storage y retornar URLs.

## 🎨 Profesionalización de UI

Si quieres mejorar la apariencia profesional de la interfaz, consulta:

- **`GUIA_PROFESIONALIZACION_UI.md`**: Opciones para profesionalizar la UI (WordPress, librerías profesionales, etc.)
- **`GUIA_EDICION_UI.md`**: Cómo editar y modificar componentes React
- **`caria_data/caria-app/docs/COMPONENT_STRUCTURE.md`**: Estructura detallada de componentes

## 🐛 Troubleshooting

### Error: "Archivo de features macro no encontrado"

**Solución**: Ejecuta el script de generación:
```bash
cd caria_data
python scripts/generate_macro_features.py
```

Luego verifica que se haya creado:
```bash
python scripts/verify_data_files.py
```

### Error: "No se encontraron datos de fundamentals o técnicos"

Esto significa que los archivos de fundamentals no están en la ubicación esperada.

**Solución**: Verifica que los archivos existan:
```bash
cd caria_data
python scripts/verify_data_files.py
```

Los archivos deben estar en:
- `caria_data/silver/fundamentals/quality_signals.parquet`
- `caria_data/silver/fundamentals/value_signals.parquet`
- `caria_data/silver/technicals/momentum_signals.parquet`

Si no existen, necesitas ejecutar los pipelines de fundamentals y técnicos.

### Error: "No LLM provider available"

El sistema intenta usar LLMs en este orden (configurable):
1. Gemini (si `GEMINI_API_KEY` está configurada)
2. Llama/Ollama (si Ollama está instalado y corriendo)
3. OpenAI (si `OPENAI_API_KEY` está configurada)

**Solución**: 
- Configura al menos uno de estos providers
- Para Gemini: Agrega `GEMINI_API_KEY` a tu `.env`
- Para Llama: Instala Ollama y ejecuta `ollama pull llama3`
- El sistema automáticamente usará el primero disponible como fallback

### Error: "FMP_API_KEY no configurado"
- Verifica que la variable de entorno `FMP_API_KEY` esté configurada
- Reinicia el servidor después de configurarla

### Error: "Base de datos no disponible"
- Verifica que PostgreSQL esté corriendo
- Verifica las credenciales en las variables de entorno
- Verifica que la base de datos `caria` existe

### Error: "Table holdings does not exist"
- Ejecuta la migración `add_holdings_table.sql`
- Verifica que tienes permisos para crear tablas

### Monte Carlo tarda mucho
- Reduce `n_paths` a 1,000 o 5,000 para desarrollo
- En producción, considera ejecutar en background con jobs async

## ✅ Checklist Pre-Lanzamiento

- [ ] Variables de entorno configuradas
- [ ] Base de datos inicializada (tabla `users` existe)
- [ ] Migración de `holdings` ejecutada
- [ ] FMP API Key configurada y funcionando
- [ ] API iniciada y responde en `/health`
- [ ] Endpoints de Monte Carlo funcionando
- [ ] Endpoints de precios funcionando
- [ ] Endpoints de holdings funcionando
- [ ] Autenticación funcionando (puedes hacer login/register)

¡Listo para lanzar! 🚀

