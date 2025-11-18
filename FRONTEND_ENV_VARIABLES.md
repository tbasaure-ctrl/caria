# Variables de Entorno para el Frontend (Vercel)

## Variables Requeridas

### 1. `VITE_API_URL` ⚠️ **OBLIGATORIA**

**Descripción:** URL del backend API (Cloud Run)

**Valor actual (producción):**
```
https://caria-api-418525923468.us-central1.run.app
```

**Valor desarrollo local:**
```
http://localhost:8000
```

**Dónde se usa:**
- Todas las llamadas API del frontend
- Conexión WebSocket para chat
- Autenticación (login, registro)
- Servicios de portfolio, precios, valuación, etc.

---

## Variables Opcionales

### 2. `VITE_GEMINI_API_KEY` (Opcional)

**Descripción:** API Key de Google Gemini para funcionalidades de chat/IA

**Cuándo necesitas esto:**
- Solo si el frontend hace llamadas directas a Gemini
- Si el backend maneja todo, **NO es necesario** en el frontend

**Valor:**
```
tu-gemini-api-key-aqui
```

---

## Cómo Configurar en Vercel

### Método 1: Dashboard de Vercel (Recomendado)

1. **Ve a tu proyecto en Vercel:**
   - https://vercel.com/dashboard
   - Selecciona tu proyecto `caria-app` (o el nombre que tengas)

2. **Accede a Environment Variables:**
   - Ve a **Settings** → **Environment Variables**

3. **Agrega/Edita variables:**

   **Para `VITE_API_URL`:**
   - **Key:** `VITE_API_URL`
   - **Value:** `https://caria-api-418525923468.us-central1.run.app`
   - **Environments:** Marca todas las que necesites:
     - ✅ Production
     - ✅ Preview
     - ✅ Development (opcional)

   **Para `VITE_GEMINI_API_KEY` (si es necesario):**
   - **Key:** `VITE_GEMINI_API_KEY`
   - **Value:** `tu-api-key`
   - **Environments:** Production, Preview

4. **Guarda y redeploy:**
   - Haz clic en **Save**
   - Ve a **Deployments**
   - Haz clic en los 3 puntos (⋯) del último deployment
   - Selecciona **Redeploy**

### Método 2: CLI de Vercel

```bash
# Instalar Vercel CLI (si no lo tienes)
npm i -g vercel

# Login
vercel login

# Agregar variable de entorno
vercel env add VITE_API_URL

# Te pedirá:
# ? What’s the value of VITE_API_URL? 
# > https://caria-api-418525923468.us-central1.run.app
# 
# ? Add VITE_API_URL to which Environments (select multiple)?
# > Production, Preview, Development

# Si necesitas Gemini API Key
vercel env add VITE_GEMINI_API_KEY
```

### Método 3: Archivo `.env` (Solo desarrollo local)

Crea un archivo `.env` en `frontend/caria-app/`:

```env
VITE_API_URL=http://localhost:8000
VITE_GEMINI_API_KEY=tu-api-key-local
```

**⚠️ IMPORTANTE:**
- Agrega `.env` a `.gitignore` para no subir credenciales
- Este archivo solo funciona en desarrollo local
- Para producción, usa el Dashboard de Vercel

---

## Verificar Configuración

### 1. Verificar en Vercel Dashboard

1. Ve a **Settings** → **Environment Variables**
2. Verifica que `VITE_API_URL` esté configurada
3. Verifica que el valor sea correcto (URL de Cloud Run)

### 2. Verificar en el código

El frontend usa `VITE_API_URL` en:
- `frontend/caria-app/services/apiService.ts`:
  ```typescript
  export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  ```

### 3. Probar en el navegador

1. Abre tu app en Vercel
2. Abre DevTools (F12) → Console
3. Ejecuta:
   ```javascript
   console.log(import.meta.env.VITE_API_URL)
   ```
4. Debería mostrar: `https://caria-api-418525923468.us-central1.run.app`

### 4. Probar conexión al backend

1. Abre DevTools → Network
2. Intenta hacer login
3. Verifica que las peticiones vayan a:
   - ✅ `https://caria-api-418525923468.us-central1.run.app/api/auth/login`
   - ❌ NO a `http://localhost:8000`

---

## Troubleshooting

### ❌ Error: "Failed to connect to server"

**Causa:** `VITE_API_URL` no está configurada o apunta a localhost

**Solución:**
1. Verifica en Vercel Dashboard que `VITE_API_URL` esté configurada
2. Verifica que el valor sea la URL de Cloud Run (no localhost)
3. Haz redeploy después de cambiar variables

### ❌ Error: "CORS policy: No 'Access-Control-Allow-Origin'"

**Causa:** El backend no tiene configurado CORS para el dominio de Vercel

**Solución:**
1. Verifica que en Cloud Run, la variable `CORS_ORIGINS` incluya tu dominio de Vercel
2. Ejemplo: `https://tu-app.vercel.app,https://tu-dominio.com`

### ❌ Las variables no se actualizan después de cambiarlas

**Causa:** Vercel necesita un redeploy para aplicar nuevas variables

**Solución:**
1. Ve a **Deployments**
2. Haz clic en los 3 puntos (⋯) del último deployment
3. Selecciona **Redeploy**
4. O haz un nuevo commit y push (trigger automático)

### ❌ Variable no funciona (undefined)

**Causa:** Variables en Vite deben empezar con `VITE_`

**Solución:**
- ✅ Correcto: `VITE_API_URL`
- ❌ Incorrecto: `API_URL`, `REACT_APP_API_URL`

---

## Resumen Rápido

### Para Producción (Vercel):

```env
VITE_API_URL=https://caria-api-418525923468.us-central1.run.app
```

### Para Desarrollo Local:

```env
VITE_API_URL=http://localhost:8000
```

### Configuración Mínima Requerida:

1. ✅ `VITE_API_URL` = URL de Cloud Run
2. ⚠️ `VITE_GEMINI_API_KEY` = Solo si el frontend llama directamente a Gemini

---

## Checklist de Deployment

- [ ] `VITE_API_URL` configurada en Vercel Dashboard
- [ ] Valor apunta a Cloud Run (no localhost)
- [ ] Variable disponible en Production y Preview
- [ ] Redeploy realizado después de configurar
- [ ] Verificado en DevTools que las peticiones van al backend correcto
- [ ] Login funciona correctamente
- [ ] Chat funciona correctamente
- [ ] Portfolio carga datos correctamente

---

## URLs Importantes

- **Backend Cloud Run:** https://caria-api-418525923468.us-central1.run.app
- **Vercel Dashboard:** https://vercel.com/dashboard
- **Health Check Backend:** https://caria-api-418525923468.us-central1.run.app/health

---

## Notas Importantes

1. **Variables deben empezar con `VITE_`:** Vite solo expone variables que empiezan con este prefijo al frontend por seguridad.

2. **Redeploy necesario:** Después de cambiar variables en Vercel, necesitas hacer redeploy para que se apliquen.

3. **URLs absolutas:** El frontend usa URLs absolutas (con `https://`), no relativas.

4. **CORS:** Asegúrate de que el backend tenga configurado CORS para permitir peticiones desde tu dominio de Vercel.


