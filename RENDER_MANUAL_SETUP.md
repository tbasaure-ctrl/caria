# Configuración Manual de Render - Pasos Exactos

## Problema
Render está usando el `Dockerfile` viejo de la raíz. Necesitas configurarlo manualmente.

## Pasos Exactos en Render Dashboard

### 1. Ir a Settings
- URL: https://dashboard.render.com/web/caria-api
- Click en **"Settings"** (menú izquierdo)

### 2. Build & Deploy Section
Scroll hasta **"Build & Deploy"** y configura:

```
Environment: Docker
Dockerfile Path: backend/Dockerfile
Docker Context: . (punto)
Build Command: (vacío)
Start Command: /app/backend/start.sh
Health Check Path: /health
```

### 3. Environment Variables
Click en **"Environment"** tab y agrega estas variables:

**DATABASE_URL** (de Neon):
```
postgresql://[user]:[password]@[host].neon.tech/[dbname]?sslmode=require
```

**LLAMA_API_KEY**:
```
[tu-llama-api-key-de-groq]
```

**LLAMA_API_URL**:
```
https://api.groq.com/openai/v1/chat/completions
```

**LLAMA_MODEL**:
```
llama-3.1-8b-instruct
```

**REDDIT_CLIENT_ID**:
```
[tu-client-id]
```

**REDDIT_CLIENT_SECRET**:
```
[tu-client-secret]
```

**REDDIT_USER_AGENT**:
```
Caria-Investment-App-v1.0
```

**FMP_API_KEY**:
```
[tu-fmp-key]
```

**JWT_SECRET_KEY**:
```
[genera-uno-nuevo]
```

**CORS_ORIGINS**:
```
https://caria-way.com;https://caria-git-main-tomas-projects-70a0592d.vercel.app
```

**RETRIEVAL_PROVIDER**:
```
llama
```

**RETRIEVAL_EMBEDDING_DIM**:
```
768
```

### 4. Save Changes
- Click **"Save Changes"** (botón azul arriba)

### 5. Manual Deploy
- Click **"Manual Deploy"** (dropdown arriba derecha)
- Select **"Deploy latest commit"**
- Espera 5-10 minutos

### 6. Verificar Logs
- Ve a **"Logs"** tab
- Busca: `Starting server on port 8080`
- Si ves `COPY services/start.sh` → Render aún usa Dockerfile viejo

## Si Sigue Fallando

### Solución: Eliminar Dockerfile Viejo del Repo

En tu terminal local:

```bash
cd notebooks
git mv Dockerfile Dockerfile.old.backup
git commit -m "Rename old Dockerfile to prevent Render from using it"
git push origin main
```

Luego en Render:
- **Manual Deploy** → **"Deploy latest commit"** otra vez

## Verificación

Después del deploy:

```bash
curl https://caria-api.onrender.com/health
```

Debería retornar JSON con `"status": "ok"`

