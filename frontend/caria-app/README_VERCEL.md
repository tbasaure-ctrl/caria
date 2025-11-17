# 🚀 Deploy a Vercel - Resumen Ejecutivo

## ✅ Archivos Creados

1. **`vercel.json`** - Configuración de Vercel
2. **`.vercelignore`** - Archivos a ignorar en deploy
3. **`VERCEL_DEPLOY.md`** - Guía completa
4. **`DEPLOY_VERCEL_QUICKSTART.md`** - Guía rápida
5. **`api/vercel-functions-example.ts`** - Ejemplo de funciones serverless (opcional)

## 🎯 Pasos para Deploy

### Opción A: Dashboard (Recomendado)

1. Ve a: https://vercel.com/new
2. Conecta tu repo de GitHub/GitLab
3. **Root Directory:** `caria_data/caria-app`
4. **Framework:** Vite (auto-detectado)
5. **Variables de Entorno:**
   - `VITE_API_URL` = `https://tu-backend.com` (o `http://localhost:8000` si aún no está en producción)
6. Click **"Deploy"**

### Opción B: CLI

```bash
cd caria_data/caria-app
npm install -g vercel
vercel login
vercel
```

## 🔑 Variables de Entorno Necesarias

En Vercel Dashboard → Settings → Environment Variables:

| Variable | Desarrollo | Producción |
|----------|------------|------------|
| `VITE_API_URL` | `http://localhost:8000` | `https://tu-backend.com` |

## 📍 URLs Importantes

- **Frontend:** `https://tu-proyecto.vercel.app`
- **Firebase Functions:** Ya configuradas (no necesitas cambiarlas)
- **Backend:** Debe estar accesible públicamente para Login, Chat, Portfolio, etc.

## ⚠️ Importante

1. **Backend debe estar público:** Tu backend FastAPI debe estar accesible desde internet (no localhost)
2. **CORS:** Configura CORS en tu backend para permitir el dominio de Vercel
3. **Firebase Functions:** Ya funcionan, no necesitas cambiarlas

## 🆘 Si Algo Falla

Ver `VERCEL_DEPLOY.md` para troubleshooting detallado.

