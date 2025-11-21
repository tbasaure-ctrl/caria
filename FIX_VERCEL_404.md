# 🔧 Solución al Error 404 en Vercel

## Problema

El frontend está devolviendo un error **404: NOT_FOUND** porque Vercel está configurado con el **Root Directory** antiguo.

**Antes (incorrecto):**
- Root Directory: `caria_data/caria-app`

**Ahora (correcto):**
- Root Directory: `frontend/caria-app`

---

## ✅ Solución: Actualizar Root Directory en Vercel

### Paso 1: Ve a la Configuración del Proyecto

1. Ve a: https://vercel.com/dashboard
2. Selecciona tu proyecto `caria-app` (o el nombre que tengas)
3. Haz clic en **Settings**

### Paso 2: Actualizar Root Directory

1. En el menú lateral, selecciona **General**
2. Busca la sección **Root Directory**
3. Haz clic en **Edit**
4. Cambia de:
   ```
   caria_data/caria-app
   ```
   a:
   ```
   frontend/caria-app
   ```
5. Haz clic en **Save**

### Paso 3: Verificar Variables de Entorno

Asegúrate de que `VITE_API_URL` esté configurada:

1. Ve a **Settings** → **Environment Variables**
2. Verifica que exista:
   - **Key:** `VITE_API_URL`
   - **Value:** `https://caria-api-418525923468.us-central1.run.app`
   - **Environments:** Production, Preview

### Paso 4: Redeploy

1. Ve a **Deployments**
2. Haz clic en los 3 puntos (⋯) del último deployment
3. Selecciona **Redeploy**
4. O haz un nuevo commit y push (trigger automático)

---

## 🔍 Verificar que Funciona

Después del redeploy:

1. Abre tu URL de Vercel
2. Deberías ver la **Landing Page** (no el error 404)
3. Abre DevTools (F12) → Console
4. Verifica que no haya errores de conexión al backend

---

## 📋 Checklist Completo

- [ ] Root Directory actualizado a `frontend/caria-app`
- [ ] `VITE_API_URL` configurada con URL de Cloud Run
- [ ] Redeploy realizado
- [ ] Landing Page carga correctamente
- [ ] No hay errores 404
- [ ] Login funciona (conecta al backend)

---

## 🆘 Si Sigue Fallando

### Opción A: Recrear el Proyecto en Vercel

1. Ve a: https://vercel.com/new
2. Conecta tu repo de GitHub
3. Configura:
   - **Root Directory:** `frontend/caria-app`
   - **Framework Preset:** Vite
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Agrega variables de entorno:
   - `VITE_API_URL` = `https://caria-api-418525923468.us-central1.run.app`
5. Haz clic en **Deploy**

### Opción B: Usar Vercel CLI

```bash
# Instalar Vercel CLI
npm i -g vercel

# Login
vercel login

# Navegar al directorio del frontend
cd frontend/caria-app

# Deploy
vercel

# Cuando pregunte:
# ? Set up and deploy "frontend/caria-app"? [Y/n] Y
# ? Which scope? (selecciona tu cuenta)
# ? Link to existing project? [y/N] N
# ? What's your project's name? caria-app
# ? In which directory is your code located? ./
# ? Want to override the settings? [y/N] N

# Agregar variable de entorno
vercel env add VITE_API_URL
# > https://caria-api-418525923468.us-central1.run.app
# > Production, Preview

# Redeploy
vercel --prod
```

### Opción C: Verificar Estructura del Repo

Asegúrate de que la estructura en GitHub sea:

```
notebooks/
  frontend/
    caria-app/
      package.json
      vite.config.ts
      vercel.json
      index.html
      ...
```

Si el repo está en la raíz, el Root Directory debería ser solo `frontend/caria-app`.

---

## 🔗 URLs Importantes

- **Vercel Dashboard:** https://vercel.com/dashboard
- **Backend Cloud Run:** https://caria-api-418525923468.us-central1.run.app
- **Health Check Backend:** https://caria-api-418525923468.us-central1.run.app/health

---

## 📝 Notas

1. **Root Directory es relativo a la raíz del repo:** Si tu repo tiene la estructura `notebooks/frontend/caria-app`, el Root Directory debe ser `frontend/caria-app` (no `notebooks/frontend/caria-app`).

2. **Vercel detecta automáticamente Vite:** Si tienes `vite.config.ts` y `package.json` con scripts de Vite, Vercel debería detectarlo automáticamente.

3. **vercel.json ya está configurado:** El archivo `frontend/caria-app/vercel.json` ya tiene la configuración correcta para SPA (Single Page Application).

4. **Variables de entorno:** Recuerda que las variables deben empezar con `VITE_` para que Vite las exponga al frontend.

---

## ✅ Resultado Esperado

Después de corregir el Root Directory y hacer redeploy:

- ✅ La Landing Page carga correctamente
- ✅ El login funciona y conecta al backend
- ✅ No hay errores 404
- ✅ Las peticiones API van a Cloud Run





