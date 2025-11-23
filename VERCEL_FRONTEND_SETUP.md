# Vercel Frontend Setup - Configuración Correcta

## Problema Actual
Vercel está haciendo deploys solo del backend, no del frontend.

## Solución: Configurar Vercel Dashboard

### Paso 1: Verificar Proyecto de Vercel

1. **Ve a Vercel Dashboard**: https://vercel.com/dashboard
2. **Busca tu proyecto** (probablemente llamado "caria" o similar)
3. **Verifica qué está desplegando:**
   - Si ves archivos Python o `backend/` → Está configurado para backend ❌
   - Si ves `frontend/caria-app/` o archivos TypeScript → Está bien configurado ✅

### Paso 2: Configurar Root Directory en Vercel Dashboard

**IMPORTANTE:** Aunque el `vercel.json` tiene `rootDirectory`, también debes configurarlo en el dashboard:

1. **Ve a tu proyecto en Vercel Dashboard**
2. **Settings → General**
3. **Busca "Root Directory"**
4. **Configúralo como:** `frontend/caria-app`
5. **Guarda los cambios**

### Paso 3: Verificar Framework Detection

1. **Settings → General**
2. **Framework Preset:** Debe ser "Vite" o "Other"
3. **Build Command:** `npm run build` (o dejar vacío si está en vercel.json)
4. **Output Directory:** `dist` (o dejar vacío si está en vercel.json)
5. **Install Command:** `npm install` (o dejar vacío)

### Paso 4: Verificar Git Integration

1. **Settings → Git**
2. **Production Branch:** Debe ser `main` o `master`
3. **Automatic deployments:** ✅ Habilitado
4. **Branch Protection:** Verifica que `cursor/*` no esté bloqueado

### Paso 5: Crear Nuevo Proyecto (Si el actual es para backend)

Si el proyecto actual está configurado para backend, crea uno nuevo para frontend:

1. **Vercel Dashboard → Add New Project**
2. **Import Git Repository:** Selecciona tu repo
3. **Configure Project:**
   - **Framework Preset:** Vite
   - **Root Directory:** `frontend/caria-app`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. **Environment Variables:**
   - `VITE_API_URL` = `https://caria-production.up.railway.app`
   - (O tu URL de Railway backend)
5. **Deploy**

### Paso 6: Verificar que Funciona

Después de configurar:

1. **Haz un push a tu branch**
2. **Ve a Vercel Dashboard → Deployments**
3. **Deberías ver un nuevo deployment iniciándose**
4. **Revisa los logs del build:**
   - Debe ejecutar `npm install` en `frontend/caria-app/`
   - Debe ejecutar `npm run build`
   - Debe generar archivos en `dist/`

## Configuración Actual en Código

### `/vercel.json` (raíz)
```json
{
  "rootDirectory": "frontend/caria-app",
  "git": {
    "deploymentEnabled": {
      "cursor/*": true,
      "main": true,
      "master": true
    }
  }
}
```

### `/frontend/caria-app/vercel.json`
```json
{
  "framework": "vite",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "rewrites": [...],
  "headers": [...]
}
```

## Troubleshooting

### Si Vercel sigue sin detectar cambios:

1. **Verifica que el proyecto esté conectado al repo correcto**
2. **Verifica que el branch esté siendo monitoreado**
3. **Intenta un deployment manual:**
   - Vercel Dashboard → Deployments → "Create Deployment"
   - Selecciona tu branch
   - Deploy

### Si el build falla:

1. **Revisa los logs de build en Vercel**
2. **Verifica que `package.json` esté en `frontend/caria-app/`**
3. **Verifica que `node_modules` no esté en `.vercelignore`**

### Si Vercel no encuentra el proyecto:

1. **Verifica que el Root Directory esté configurado en el dashboard**
2. **El `vercel.json` en la raíz ayuda, pero el dashboard tiene prioridad**

## Checklist Final

- [ ] Root Directory configurado en Vercel Dashboard: `frontend/caria-app`
- [ ] Framework detectado como "Vite"
- [ ] Build Command: `npm run build`
- [ ] Output Directory: `dist`
- [ ] Environment Variable `VITE_API_URL` configurada
- [ ] Git integration conectada al repo correcto
- [ ] Automatic deployments habilitado
- [ ] Branch `cursor/*` no está bloqueado

## Comando Rápido para Verificar

```bash
# Ver estructura del proyecto
ls -la frontend/caria-app/

# Verificar que package.json existe
cat frontend/caria-app/package.json

# Verificar vercel.json
cat vercel.json
cat frontend/caria-app/vercel.json
```
