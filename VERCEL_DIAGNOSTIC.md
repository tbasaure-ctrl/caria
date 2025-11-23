# Vercel Diagnostic - ¿Cuál Proyecto Está Desplegando?

## Cómo Identificar el Problema

### Paso 1: Ve a Vercel Dashboard
1. https://vercel.com/dashboard
2. Lista todos tus proyectos

### Paso 2: Identifica Cada Proyecto

Para cada proyecto, verifica:

#### Proyecto de BACKEND (❌ Incorrecto para frontend):
- **Settings → General → Root Directory:** 
  - Vacío, o `backend/`, o raíz `/`
- **Build Logs muestran:**
  - `pip install`
  - `python` o `uvicorn`
  - Archivos `.py`
- **Framework:** "Other" o "Docker"

#### Proyecto de FRONTEND (✅ Correcto):
- **Settings → General → Root Directory:** 
  - `frontend/caria-app`
- **Build Logs muestran:**
  - `npm install`
  - `npm run build`
  - Archivos `.tsx`, `.ts`, `.js`
- **Framework:** "Vite" o "Next.js"

### Paso 3: Verifica los Deployments

1. **Ve a Deployments** en cada proyecto
2. **Revisa el último deployment:**
   - **Build Logs** → ¿Qué comandos ejecuta?
   - **Source** → ¿De qué branch viene?
   - **Files** → ¿Qué archivos incluye?

### Paso 4: Identifica el Problema

**Si el proyecto que debería ser frontend está desplegando backend:**
- Root Directory está mal configurado
- O estás mirando el proyecto equivocado

**Si no hay proyecto de frontend:**
- Necesitas crear uno nuevo

## Solución Rápida

### Opción A: Corregir Proyecto Existente

1. **Vercel Dashboard → Tu Proyecto**
2. **Settings → General**
3. **Root Directory:** Cambiar a `frontend/caria-app`
4. **Save**
5. **Trigger manual deployment** para probar

### Opción B: Crear Proyecto Nuevo (Recomendado)

1. **Vercel Dashboard → Add New Project**
2. **Import Git Repository:** Tu repo de GitHub
3. **Configure:**
   - **Framework:** Vite
   - **Root Directory:** `frontend/caria-app` ⚠️ IMPORTANTE
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. **Environment Variables:**
   - `VITE_API_URL` = `https://caria-production.up.railway.app`
5. **Deploy**

## Checklist de Verificación

Para el proyecto de FRONTEND en Vercel:

- [ ] **Root Directory:** `frontend/caria-app`
- [ ] **Framework:** Vite
- [ ] **Build Command:** `npm run build` (o vacío si está en vercel.json)
- [ ] **Output Directory:** `dist` (o vacío si está en vercel.json)
- [ ] **Environment Variable:** `VITE_API_URL` configurada
- [ ] **Git Integration:** Conectado al repo correcto
- [ ] **Production Branch:** `main` o `master`
- [ ] **Automatic Deployments:** ✅ Habilitado

## Cómo Verificar que Funciona

Después de configurar:

1. **Haz un pequeño cambio** (ej: agregar un comentario en un archivo)
2. **Push a GitHub**
3. **Ve a Vercel Dashboard → Deployments**
4. **Deberías ver:**
   - Nuevo deployment iniciándose automáticamente
   - Build logs mostrando `npm install` y `npm run build`
   - Deployment exitoso con URL de preview

## Preguntas para Responder

1. **¿Cuántos proyectos tienes en Vercel?**
   - Si solo uno → Probablemente está configurado para backend
   - Si dos → Uno debería ser frontend, otro backend

2. **¿Qué muestra el último deployment?**
   - Build logs con `npm` → Frontend ✅
   - Build logs con `pip` o `python` → Backend ❌

3. **¿Cuál es el Root Directory configurado?**
   - `frontend/caria-app` → Correcto ✅
   - Vacío o `backend/` → Incorrecto ❌
