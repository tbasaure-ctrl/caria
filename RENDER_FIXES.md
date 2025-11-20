# 🔧 Correcciones para Render - Problemas Detectados

## ❌ Problemas Encontrados en los Logs

### 1. **Base de Datos: Conectando a localhost en lugar de Neon**
```
connection to server at "localhost" (127.0.0.1), port 5432 failed
```

**Causa:** `DATABASE_URL` no está configurado en Render.

**Solución:**
1. Ve a Render Dashboard → tu servicio → **Settings** → **Environment**
2. Agrega:
   ```
   DATABASE_URL = postgresql://[user]:[password]@[host].neon.tech/[dbname]?sslmode=require
   ```
3. **Obtén el connection string de Neon:**
   - Ve a https://console.neon.tech
   - Click en tu proyecto
   - Ve a **"Connection Details"**
   - Copia el **Connection String** (formato: `postgresql://user:pass@host.neon.tech/dbname?sslmode=require`)

### 2. **RAG: "llama" no es un proveedor de embeddings válido**
```
No se pudo inicializar el stack RAG: Proveedor de embeddings no soportado: llama
```

**Causa:** `RETRIEVAL_PROVIDER=llama` está configurado, pero "llama" es para LLM, no para embeddings.

**Solución:** Ya corregido en `render.yaml`. Cambia en Render Dashboard:
- **Settings** → **Environment**
- Cambia `RETRIEVAL_PROVIDER` de `llama` a `local`
- Agrega también:
  ```
  RETRIEVAL_EMBEDDING_MODEL = nomic-embed-text-v1
  ```

### 3. **Puerto: Servidor corriendo en 10000 en lugar de 8080**
```
Uvicorn running on http://0.0.0.0:10000
```

**Causa:** Render puede estar configurando PORT=10000 automáticamente.

**Solución:** Ya agregado en `render.yaml`. Verifica en Render Dashboard:
- **Settings** → **Environment**
- Agrega: `PORT = 8080`
- O verifica que Render no esté sobrescribiendo el puerto

### 4. **HMM Model: Archivo no encontrado (CRÍTICO)**
```
Modelo HMM no encontrado en /app/models/regime_hmm_model.pkl
```

**⚠️ CRÍTICO:** El modelo HMM es necesario para:
- `/api/regime/current` - Detección de régimen macro (tiene fallback pero valores por defecto)
- `/api/portfolio/regime-test` - Testing de portfolios por régimen
- `/api/tactical/allocation` - Asignación táctica basada en régimen

**Estado actual:** ✅ El modelo existe en `caria_data/models/regime_hmm_model.pkl` y está en git.

**Problema:** El Dockerfile intenta copiarlo pero Render no lo encuentra en `/app/models/`.

**Solución:**
1. Verifica en logs de Render build que el modelo se copió:
   - Busca: `"✓ Regime HMM model copied to /app/models/"`
   - Si ves: `"⚠ Warning: regime_hmm_model.pkl not found"`, el modelo no se copió
2. Verifica la ruta en el Dockerfile (línea 38):
   - Debe buscar en `/app/caria_data/models/regime_hmm_model.pkl`
   - Y copiar a `/app/models/regime_hmm_model.pkl`
3. Si no se copia, verifica que `caria_data/models/regime_hmm_model.pkl` esté en el contexto de Docker
4. Ver guía completa en `HMM_MODEL_SETUP.md` para más detalles

## ✅ Pasos para Corregir TODO

### Paso 1: Actualizar Variables de Entorno en Render

Ve a **Render Dashboard** → **caria-api** → **Settings** → **Environment**

**Elimina estas variables si existen:**
- `RETRIEVAL_PROVIDER = llama` ❌

**Agrega/Actualiza estas variables:**

```bash
# Base de Datos (CRÍTICO)
DATABASE_URL = postgresql://[user]:[password]@[host].neon.tech/[dbname]?sslmode=require

# Embeddings (RAG)
RETRIEVAL_PROVIDER = local
RETRIEVAL_EMBEDDING_MODEL = nomic-embed-text-v1
RETRIEVAL_EMBEDDING_DIM = 768

# Puerto
PORT = 8080

# API Keys
LLAMA_API_KEY = gsk_****************************************************
LLAMA_API_URL = https://api.groq.com/openai/v1/chat/completions
LLAMA_MODEL = llama-3.1-8b-instruct

# Reddit (si las tienes)
REDDIT_CLIENT_ID = [tu-client-id]
REDDIT_CLIENT_SECRET = [tu-client-secret]
REDDIT_USER_AGENT = Caria-Investment-App-v1.0

# FMP (si la tienes)
FMP_API_KEY = [tu-fmp-api-key]

# Security
JWT_SECRET_KEY = [genera-uno-nuevo]
# Generar con: python -c "import secrets; print(secrets.token_urlsafe(32))"

# CORS
CORS_ORIGINS = https://caria-way.com;https://caria-git-main-tomas-projects-70a0592d.vercel.app
```

### Paso 2: Habilitar pgvector en Neon

1. Ve a https://console.neon.tech
2. Click en tu proyecto
3. Ve a **"SQL Editor"**
4. Ejecuta:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Paso 3: Ejecutar Migraciones

**Opción A: Desde Render Shell**
1. Render Dashboard → **Shell** tab
2. Ejecuta:
```bash
python backend/api/db_bootstrap.py
```

**Opción B: Desde Neon SQL Editor**
1. Copia el contenido de `caria_data/migrations/init.sql`
2. Pega y ejecuta en Neon SQL Editor

### Paso 4: Redeploy en Render

1. Render Dashboard → **Manual Deploy** → **Deploy latest commit**
2. O haz un commit vacío para trigger el deploy:
```bash
git commit --allow-empty -m "Trigger Render redeploy"
git push
```

### Paso 5: Verificar que Funciona

**Test Health:**
```bash
curl https://caria-api.onrender.com/health
```

**Debería retornar:**
```json
{
  "status": "ok",
  "database": "available",
  "rag": "available",
  "regime": "available",  // Debe estar disponible si el modelo está cargado
  "factors": "available"
}
```

**Test Secrets:**
```bash
curl https://caria-api.onrender.com/api/debug/secrets-status
```

## 📋 Checklist Final

- [ ] `DATABASE_URL` configurado con connection string de Neon
- [ ] `RETRIEVAL_PROVIDER` cambiado a `local` (no `llama`)
- [ ] `RETRIEVAL_EMBEDDING_MODEL` configurado
- [ ] `PORT` configurado a `8080`
- [ ] `pgvector` habilitado en Neon
- [ ] Migraciones ejecutadas
- [ ] Redeploy completado
- [ ] Health check retorna `database: available` y `rag: available`

## 🆘 Si Sigue Fallando

1. **Revisa los logs de Render** para ver el error específico
2. **Verifica que DATABASE_URL esté correcto:**
   - Debe empezar con `postgresql://`
   - Debe incluir `?sslmode=require` al final
   - No debe tener espacios extra
3. **Verifica que todas las variables estén sin espacios:**
   - `RETRIEVAL_PROVIDER=local` ✅
   - `RETRIEVAL_PROVIDER = local` ❌ (con espacios puede fallar)

