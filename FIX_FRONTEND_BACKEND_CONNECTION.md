# 🔧 Fix Frontend-Backend Connection

## Problema Identificado

1. **Frontend (Vercel)**: Está usando `VITE_API_URL` que por defecto es `http://localhost:8000`
2. **Backend (Railway)**: Necesita tener `CORS_ORIGINS` configurado con la URL de Vercel
3. **Conexión**: No están conectados porque:
   - Vercel no tiene `VITE_API_URL` configurada con la URL de Railway
   - Railway no tiene `CORS_ORIGINS` configurado con la URL de Vercel

## Solución Paso a Paso

### Paso 1: Encontrar URL Pública de Railway

La URL interna `caria.railway.internal` no funciona para conexiones externas. Necesitas la URL pública.

**Cómo encontrarla:**
1. Ve a Railway Dashboard: https://railway.app
2. Click en tu servicio "caria"
3. Ve a la pestaña **"Settings"**
4. Busca **"Networking"** o **"Public Domain"**
5. Deberías ver algo como: `https://caria-production.up.railway.app` o similar
6. **Copia esa URL** - esa es la que necesitas

**O desde el dashboard principal:**
- Railway Dashboard → Tu Proyecto → Tu Servicio
- La URL pública debería aparecer en la parte superior del servicio
- Generalmente es: `https://[nombre-servicio]-[hash].up.railway.app`

### Paso 2: Configurar VITE_API_URL en Vercel

1. Ve a Vercel Dashboard: https://vercel.com
2. Click en tu proyecto **"caria"**
3. Ve a **"Settings"** → **"Environment Variables"**
4. Click **"Add New"**
5. Configura:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://[tu-url-railway].up.railway.app` (la URL que copiaste en Paso 1)
   - **Environments**: Marca todas (Production, Preview, Development)
6. Click **"Save"**
7. **IMPORTANTE**: Ve a **"Deployments"** → Click en el último deployment → Click en los tres puntos (⋯) → **"Redeploy"**

### Paso 3: Configurar CORS_ORIGINS en Railway

1. Ve a Railway Dashboard: https://railway.app
2. Click en tu servicio "caria"
3. Ve a la pestaña **"Variables"**
4. Busca o crea la variable:
   - **Key**: `CORS_ORIGINS`
   - **Value**: `https://caria-git-main-tomas-projects-70a0592d.vercel.app`
   - (Si tienes múltiples URLs, sepáralas con comas)
5. Click **"Add"** o **"Save"**
6. Railway redeployará automáticamente

### Paso 4: Verificar Conexión

1. Abre tu sitio de Vercel: https://caria-git-main-tomas-projects-70a0592d.vercel.app
2. Abre DevTools (F12) → Pestaña **"Network"**
3. Intenta hacer login o cualquier acción que llame al backend
4. Verifica que las peticiones vayan a la URL de Railway (no a localhost)
5. Verifica que no haya errores de CORS

## URLs Necesarias

**Frontend (Vercel):**
- URL: `https://caria-git-main-tomas-projects-70a0592d.vercel.app`

**Backend (Railway):**
- URL Interna: `caria.railway.internal` (solo funciona dentro de Railway)
- URL Pública: `https://[nombre].up.railway.app` (necesitas encontrarla)

## Checklist

- [ ] Encontré la URL pública de Railway
- [ ] Configuré `VITE_API_URL` en Vercel con la URL de Railway
- [ ] Redeployé Vercel después de configurar la variable
- [ ] Configuré `CORS_ORIGINS` en Railway con la URL de Vercel
- [ ] Verifiqué que Railway redeployó correctamente
- [ ] Probé la conexión desde el frontend
- [ ] No hay errores de CORS
- [ ] Las peticiones van al backend correcto

## Troubleshooting

### Error: "Failed to fetch" o "Network Error"
- Verifica que `VITE_API_URL` esté configurada correctamente en Vercel
- Verifica que hayas redeployado Vercel después de agregar la variable
- Verifica que la URL de Railway sea correcta (debe empezar con `https://`)

### Error: "CORS policy: No 'Access-Control-Allow-Origin' header"
- Verifica que `CORS_ORIGINS` en Railway incluya exactamente la URL de Vercel
- Debe ser `https://caria-git-main-tomas-projects-70a0592d.vercel.app` (con https, sin trailing slash)
- Verifica que Railway haya redeployado después de agregar la variable

### Error: "Connection refused" o "Cannot connect"
- Verifica que Railway esté corriendo (no crashed)
- Verifica que la URL de Railway sea correcta
- Verifica los logs de Railway para ver si hay errores

