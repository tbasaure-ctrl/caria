# Variables de Entorno para el Frontend (Vercel)

La app en Vercel se conecta al backend que vive en Railway/Neon. Solo necesitamos exponer una variable a Vite para que todas las llamadas vayan a la API correcta.

## Variables Requeridas

### 1. `VITE_API_URL` ⚠️ **OBLIGATORIA**

- **Descripción:** URL base del backend (FastAPI + Socket.IO) desplegado en Railway.
- **Valor actual (producción):**
  ```
  https://caria-production.up.railway.app
  ```
- **Valor desarrollo local:**
  ```
  http://localhost:8000
  ```
- **Dónde se usa:**
  - Todas las peticiones REST (`fetchWithAuth`)
  - Conexión WebSocket del chat (se deriva eliminando `/api`)
  - Login, registro y refresco de tokens
  - Servicios de comunidad, Arena, análisis, precios, valuación, etc.

> Nota: Si el backend cambia de dominio en Railway, actualiza este valor y redeploy.

---

## Cómo Configurar en Vercel

### Método 1: Dashboard de Vercel (Recomendado)

1. Ingresa a [https://vercel.com/dashboard](https://vercel.com/dashboard) y abre tu proyecto.
2. Ve a **Settings → Environment Variables**.
3. Agrega o edita la variable:
   - **Key:** `VITE_API_URL`
   - **Value:** `https://caria-production.up.railway.app`
   - **Environments:** Production, Preview y Development.
4. Guarda y ejecuta un **Redeploy** para que la variable se propague.

### Método 2: CLI de Vercel

```bash
npm i -g vercel
vercel login
vercel env add VITE_API_URL
# Ingresa https://caria-production.up.railway.app y marca Production/Preview/Development
```

### Método 3: Archivo `.env` (solo local)

En `frontend/caria-app/.env`:

```env
VITE_API_URL=http://localhost:8000
```

---

## Verificar Configuración

1. **Dashboard:** Confirmar que `VITE_API_URL` aparece en Settings → Environment Variables.
2. **Código:** `frontend/caria-app/services/apiService.ts` usa esta variable.
3. **Console del navegador:** `console.log(import.meta.env.VITE_API_URL)` debe mostrar la URL de Railway.
4. **Network tab:** las llamadas deben apuntar a `https://caria-production.up.railway.app/...` y no a localhost.

---

## Troubleshooting

- **"Failed to connect to server":** Revisa que `VITE_API_URL` esté definido y el backend esté activo en Railway.
- **CORS issues:** agrega tu dominio de Vercel en la variable `CORS_ORIGINS` del backend (Railway) o usa la regex de `*.vercel.app` ya soportada.
- **Variables no actualizan:** cada cambio en Vercel requiere un redeploy (manual o nuevo commit).
- **`undefined` en Vite:** recuerda que todas las variables deben iniciar con `VITE_`.

---

## Checklist de Deployment

- [ ] `VITE_API_URL` configurada en Vercel (Production/Preview/Development)
- [ ] Valor apunta a Railway (no a Cloud Run ni localhost)
- [ ] Redeploy realizado después del cambio
- [ ] DevTools confirma que las peticiones usan la URL correcta
- [ ] Login, chat y widgets cargan datos reales del backend en Railway

---

## URLs Importantes

- **Backend Railway:** https://caria-production.up.railway.app
- **Health Check:** https://caria-production.up.railway.app/health
- **Vercel Dashboard:** https://vercel.com/dashboard

---

## Notas

1. Solo `VITE_API_URL` es necesaria actualmente; el backend maneja Groq Llama, embeddings y Neon.
2. Mantén el valor actualizado si cambias de Railway Space o dominio personalizado.
3. Para pruebas locales, levanta el backend (`uvicorn api.app:socketio_app --reload`) y usa `http://localhost:8000`.





