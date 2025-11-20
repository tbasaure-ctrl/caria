# Configuración Completa de Render - Pasos Concretos

## Problema Actual
Render está usando el `Dockerfile` de la raíz que busca `services/start.sh` (no existe).
Debe usar `backend/Dockerfile` que busca `backend/start.sh` (existe).

## Solución: Configurar Render Manualmente

### Paso 1: Ir a Render Dashboard
1. Abre: https://dashboard.render.com
2. Click en servicio: **caria-api**

### Paso 2: Settings → Build & Deploy
Configura estos valores exactos:

- **Environment**: `Docker`
- **Dockerfile Path**: `backend/Dockerfile` (sin `./`)
- **Docker Context**: `.` (punto)
- **Build Command**: (dejar vacío)
- **Start Command**: `/app/backend/start.sh`
- **Health Check Path**: `/health`

### Paso 3: Settings → Environment
Agrega estas variables (una por una):

#### Base de Datos (Neon)
```
DATABASE_URL = <tu-neon-connection-string>
```
Formato: `postgresql://user:password@host.neon.tech/dbname?sslmode=require`

#### API Keys
```
LLAMA_API_KEY = gsk_****************************************************
LLAMA_API_URL = https://api.groq.com/openai/v1/chat/completions
LLAMA_MODEL = llama-3.1-8b-instruct
```

#### Reddit
```
REDDIT_CLIENT_ID = <tu-reddit-client-id>
REDDIT_CLIENT_SECRET = <tu-reddit-client-secret>
REDDIT_USER_AGENT = Caria-Investment-App-v1.0
```

#### FMP
```
FMP_API_KEY = <tu-fmp-api-key>
```

#### Security
```
JWT_SECRET_KEY = <genera-uno-nuevo>
```
Generar con: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

#### App Config
```
CORS_ORIGINS = https://caria-way.com;https://caria-git-main-tomas-projects-70a0592d.vercel.app
RETRIEVAL_PROVIDER = llama
RETRIEVAL_EMBEDDING_DIM = 768
```

### Paso 4: Guardar y Deploy
1. Click **"Save Changes"**
2. Ve a **"Manual Deploy"** → **"Deploy latest commit"**
3. Espera 5-10 minutos

### Paso 5: Verificar
1. Ve a **"Logs"** tab
2. Busca: `Starting server on port 8080`
3. Si ves errores de `services/start.sh`, Render aún usa el Dockerfile viejo

## Si Sigue Fallando

### Opción 1: Eliminar Dockerfile viejo
En tu repo local:
```bash
# Renombrar el Dockerfile viejo
mv Dockerfile Dockerfile.old
git add Dockerfile.old
git commit -m "Rename old Dockerfile to avoid conflicts"
git push
```

### Opción 2: Forzar en Render
En Render Settings → Build & Deploy:
- Cambia **Dockerfile Path** a: `./notebooks/backend/Dockerfile` (si tu repo tiene estructura notebooks/)
- O simplemente: `backend/Dockerfile`

## Verificación Final

Después del deploy exitoso:

```bash
# Health check
curl https://caria-api.onrender.com/health

# Debe retornar:
{
  "status": "ok",
  "database": "available",
  "rag": "available" (si Neon tiene pgvector),
  ...
}
```

## URLs Importantes

- **Render Dashboard**: https://dashboard.render.com
- **Servicio URL**: https://caria-api.onrender.com
- **Logs**: https://dashboard.render.com → caria-api → Logs


