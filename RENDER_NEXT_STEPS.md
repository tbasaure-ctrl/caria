# Pasos Después de Configurar Render

## ✅ Paso 1: Verificar que el Build Funciona

1. **Ve a Logs** en Render dashboard
2. Espera a que termine el build (5-10 minutos)
3. Deberías ver: `Starting server on port 8080`
4. Si falla, revisa los logs para ver el error específico

## ✅ Paso 2: Configurar Variables de Entorno

Ve a **Settings** → **Environment** y agrega estas variables:

### Base de Datos (Neon)
```
DATABASE_URL = postgresql://[user]:[password]@[host].neon.tech/[dbname]?sslmode=require
```
**Obtén esto de:** Neon dashboard → tu proyecto → Connection String

### API Keys
```
LLAMA_API_KEY = gsk_****************************************************
LLAMA_API_URL = https://api.groq.com/openai/v1/chat/completions
LLAMA_MODEL = llama-3.1-8b-instruct
```

### Reddit
```
REDDIT_CLIENT_ID = [tu-client-id]
REDDIT_CLIENT_SECRET = [tu-client-secret]
REDDIT_USER_AGENT = Caria-Investment-App-v1.0
```

### FMP
```
FMP_API_KEY = [tu-fmp-api-key]
```

### Security
```
JWT_SECRET_KEY = [genera-uno-nuevo]
```
**Generar con:** `python -c "import secrets; print(secrets.token_urlsafe(32))"`

### App Config
```
CORS_ORIGINS = https://caria-way.com;https://caria-git-main-tomas-projects-70a0592d.vercel.app
RETRIEVAL_PROVIDER = local
RETRIEVAL_EMBEDDING_MODEL = nomic-embed-text-v1
RETRIEVAL_EMBEDDING_DIM = 768
PORT = 8080
```

**⚠️ IMPORTANTE:** 
- `RETRIEVAL_PROVIDER` debe ser `local` (NO `llama`)
- `llama` es solo para LLM, no para embeddings
- Los embeddings usan modelos locales con sentence-transformers

## ✅ Paso 3: Habilitar pgvector en Neon

1. Ve a Neon dashboard: https://console.neon.tech
2. Click en tu proyecto
3. Ve a **"SQL Editor"**
4. Ejecuta:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
5. Verifica:
```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

## ✅ Paso 4: Ejecutar Migraciones de Base de Datos

### Opción A: Desde Render Shell
1. En Render dashboard → **"Shell"** tab
2. Ejecuta:
```bash
python backend/api/db_bootstrap.py
```

### Opción B: Desde Neon SQL Editor
1. Ve a Neon SQL Editor
2. Copia y pega el contenido de:
   - `caria_data/migrations/init.sql`
   - `caria_data/migrations/012_model_portfolios.sql`
3. Ejecuta cada uno

## ✅ Paso 5: Verificar que Todo Funciona

### Test Health Endpoint
```bash
curl https://caria-api.onrender.com/health
```

**Debería retornar:**
```json
{
  "status": "ok",
  "database": "available",
  "rag": "available",
  "regime": "available",
  "factors": "available"
}
```

### Test Secrets Status
```bash
curl https://caria-api.onrender.com/api/debug/secrets-status
```

**Debería mostrar:**
```json
{
  "secrets_configured": {
    "llama_api_key": true,
    "reddit_client_id": true,
    "fmp_api_key": true,
    ...
  }
}
```

### Test Registro de Usuario
```bash
curl -X POST https://caria-api.onrender.com/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "test123456",
    "username": "testuser"
  }'
```

**Debería retornar:** `201 Created` con access_token

## ✅ Paso 6: Actualizar Frontend (Vercel)

1. Ve a Vercel dashboard
2. Tu proyecto → **Settings** → **Environment Variables**
3. Actualiza:
```
VITE_API_URL = https://caria-api.onrender.com
```
4. **Redeploy** el frontend

## ✅ Paso 7: Verificar Frontend Funciona

1. Abre tu sitio en Vercel
2. Intenta registrarte
3. Verifica que los widgets carguen datos
4. Revisa la consola del navegador para errores

## Troubleshooting

### Si Health Check Falla
- Revisa logs de Render
- Verifica que DATABASE_URL esté correcto
- Verifica que todas las variables de entorno estén configuradas

### Si RAG muestra "disabled"
- Verifica que ejecutaste `CREATE EXTENSION vector;` en Neon
- Revisa logs para errores de conexión a base de datos

### Si APIs no funcionan
- Verifica que las API keys estén correctas en Render Environment
- Revisa logs para errores específicos de cada API

## URLs Importantes

- **Render Dashboard**: https://dashboard.render.com/web/caria-api
- **Render Logs**: https://dashboard.render.com/web/caria-api/logs
- **Neon Dashboard**: https://console.neon.tech
- **Backend URL**: https://caria-api.onrender.com
- **Frontend URL**: Tu URL de Vercel

