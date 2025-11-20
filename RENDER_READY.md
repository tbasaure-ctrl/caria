# ✅ Render está Listo para Deploy

## Lo que ya está configurado:

1. ✅ `render.yaml` - Configuración correcta
2. ✅ `backend/Dockerfile` - Dockerfile correcto (usa `backend/start.sh`)
3. ✅ `Dockerfile` viejo renombrado a `Dockerfile.old.backup`
4. ✅ `.dockerignore` - Ignora Dockerfile viejo
5. ✅ Código actualizado para usar Llama en lugar de Gemini

## Próximo paso en Render Dashboard:

### 1. Settings → Build & Deploy
Verifica que tenga:
- **Dockerfile Path**: `backend/Dockerfile`
- **Start Command**: `/app/backend/start.sh`

### 2. Settings → Environment
Agrega estas variables (si no están):
- `DATABASE_URL` = tu Neon connection string
- `REDDIT_CLIENT_ID` = tu client ID
- `REDDIT_CLIENT_SECRET` = tu client secret  
- `FMP_API_KEY` = tu FMP key
- `JWT_SECRET_KEY` = genera uno nuevo

### 3. Manual Deploy
- Click **"Manual Deploy"** → **"Deploy latest commit"**

## Después del Deploy:

```bash
curl https://caria-api.onrender.com/health
```

Debería retornar: `{"status": "ok", ...}`

