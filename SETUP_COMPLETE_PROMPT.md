# Prompt: Configurar y Probar Caria Backend + Frontend

## Objetivo
Configurar todos los secrets de API, verificar deployments en Cloud Run y Vercel, y probar que todos los endpoints funcionen correctamente.

## ⚠️ Si Tienes Problemas con Timeout en Cloud Run

**Lee primero:** `TEST_CLOUD_RUN_BACKEND.md` - Guía completa para diagnosticar y solucionar timeouts.

**Prueba rápida:**
```bash
# Health check simple
curl https://caria-api-418525923468.us-central1.run.app/health

# Ver logs
gcloud run services logs read caria-api --region=us-central1 --project=caria-backend --limit=50
```

## Tareas

### 1. Configurar Secrets en Google Cloud Secret Manager

**URL:** https://console.cloud.google.com/security/secret-manager?project=caria-backend

Crear los siguientes secrets (si no existen):

1. **reddit-client-id**
   - Valor: `1eIYr0z6slzt62EXy1KQ6Q`

2. **reddit-client-secret**
   - Valor: `p53Yud4snfuadHAvgva_6vWkj0eXcw`

3. **gemini-api-key**
   - Valor: [Obtener de Google AI Studio: https://aistudio.google.com/apikey]

4. **fmp-api-key**
   - Valor: [Obtener de Financial Modeling Prep: https://site.financialmodelingprep.com/developer/docs/]

5. **postgres-password**
   - Valor: `SimplePass123` (o el password actual de PostgreSQL)

6. **jwt-secret-key**
   - Valor: [Generar string aleatorio seguro, ej: `openssl rand -hex 32`]

**Método rápido (gcloud CLI):**
```bash
gcloud config set project caria-backend
echo -n '1eIYr0z6slzt62EXy1KQ6Q' | gcloud secrets create reddit-client-id --data-file=-
echo -n 'p53Yud4snfuadHAvgva_6vWkj0eXcw' | gcloud secrets create reddit-client-secret --data-file=-
# Repetir para los demás secrets
```

### 2. Verificar Cloud Run Deployment

**URL:** https://console.cloud.google.com/run?project=caria-backend

1. Seleccionar servicio `caria-api`
2. Verificar última revisión está activa
3. Ir a "VARIABLES & SECRETS" y confirmar que aparecen:
   - `GEMINI_API_KEY` → `gemini-api-key:latest`
   - `FMP_API_KEY` → `fmp-api-key:latest`
   - `REDDIT_CLIENT_ID` → `reddit-client-id:latest`
   - `REDDIT_CLIENT_SECRET` → `reddit-client-secret:latest`
   - `POSTGRES_PASSWORD` → `postgres-password:latest`
   - `JWT_SECRET_KEY` → `jwt-secret-key:latest`

4. Si faltan, hacer redeploy:
   - "EDIT & DEPLOY NEW REVISION" → "DEPLOY" (sin cambios)

**URL del backend:** `https://caria-api-418525923468.us-central1.run.app`

### 3. Verificar Frontend en Vercel

**URL:** https://vercel.com/dashboard

1. Seleccionar proyecto `caria-app` (o nombre del proyecto)
2. Ir a "Settings" → "Environment Variables"
3. Verificar que existe:
   - `VITE_API_URL` = `https://caria-api-418525923468.us-central1.run.app`
4. Si falta o está incorrecta, agregarla y hacer redeploy

### 4. Trigger GitHub Actions Deployment (si es necesario)

Si hiciste cambios en secrets, trigger un nuevo deployment:

```bash
git commit --allow-empty -m "Trigger: Redeploy after secrets configuration"
git push origin main
```

Monitorear en: https://github.com/tbasaure-ctrl/caria/actions

### 5. Probar Backend

Ejecutar script de diagnóstico:

```bash
cd notebooks
python diagnose_api_connection.py
```

**Resultados esperados:**
- ✅ Health Check: Backend corriendo
- ✅ CORS: Configurado correctamente
- ✅ FMP API: Funciona (puede requerir auth)
- ✅ Reddit API: Funciona (no debe dar 401)
- ✅ Fear & Greed: Funciona (puede requerir auth)

### 6. Probar Frontend

1. Abrir la app en Vercel
2. Verificar en DevTools (F12) → Console:
   ```javascript
   console.log(import.meta.env.VITE_API_URL)
   // Debe mostrar: https://caria-api-418525923468.us-central1.run.app
   ```
3. Probar funcionalidades:
   - Login/Registro
   - Ver portfolio
   - Ver Fear & Greed Index
   - Ver Reddit Sentiment (debe cargar datos, no error)
   - Ver precios en tiempo real

### 7. Verificar Logs si hay Problemas

**Cloud Run logs:**
```bash
gcloud run services logs read caria-api --region=us-central1 --limit=50
```

**Buscar errores:**
- "Secret not found" → Secret no existe o no tiene permisos
- "401 Unauthorized" (Reddit) → Credenciales incorrectas
- "API key invalid" → API key incorrecta o expirada

## Checklist Final

- [ ] Todos los secrets creados en Secret Manager
- [ ] Cloud Run muestra todos los secrets en "VARIABLES & SECRETS"
- [ ] Vercel tiene `VITE_API_URL` configurada
- [ ] `diagnose_api_connection.py` muestra todos los endpoints funcionando
- [ ] Frontend carga y muestra datos (no errores en console)
- [ ] Reddit Sentiment muestra datos (no error 401)
- [ ] Precios se actualizan correctamente
- [ ] Fear & Greed Index se carga

## Comandos Rápidos

```bash
# Verificar secrets
gcloud secrets list --project=caria-backend

# Ver logs de Cloud Run
gcloud run services logs read caria-api --region=us-central1 --limit=50

# Probar backend
python diagnose_api_connection.py

# Health check manual
curl https://caria-api-418525923468.us-central1.run.app/health
```

## Enlaces Importantes

- **Secret Manager:** https://console.cloud.google.com/security/secret-manager?project=caria-backend
- **Cloud Run:** https://console.cloud.google.com/run?project=caria-backend
- **Vercel Dashboard:** https://vercel.com/dashboard
- **GitHub Actions:** https://github.com/tbasaure-ctrl/caria/actions
- **Backend URL:** https://caria-api-418525923468.us-central1.run.app

## Tiempo Estimado

- Configurar secrets: 10-15 min
- Verificar deployments: 5 min
- Probar endpoints: 5 min
- **Total: ~25 minutos**

---

**Objetivo:** Tener backend y frontend completamente funcionales con todas las APIs conectadas.

