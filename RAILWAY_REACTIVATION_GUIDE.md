# 🚂 Guía de Reactivación de Railway

## Problema
Railway dejó de funcionar porque la suscripción expiró o fue removida. Necesitamos reactivar el servicio.

## Pasos para Reactivar

### 1. Verificar Estado de la Cuenta
1. Ve a https://railway.app
2. Inicia sesión con tu cuenta de GitHub
3. Verifica que tu proyecto "caria-backend" o "caria" esté visible
4. Si el proyecto aparece como "paused" o "offline", continúa con los siguientes pasos

### 2. Verificar Configuración del Servicio

#### A. Root Directory
1. En Railway, ve a tu proyecto
2. Click en el servicio "caria" (o el nombre que tenga)
3. Ve a la pestaña **Settings**
4. En la sección **"Build & Deploy"**, verifica:
   - **Root Directory**: Debe estar **VACÍO** o ser `.` (punto)
   - Si está configurado con otro valor, cámbialo a vacío y guarda

#### B. Build Configuration
Verifica que Railway esté usando el `railway.json`:
- Railway debería detectar automáticamente el Dockerfile en `backend/Dockerfile`
- Si no lo detecta, en Settings → Build, configura:
  - **Builder**: Dockerfile
  - **Dockerfile Path**: `backend/Dockerfile`

#### C. Start Command
En Settings → Deploy, verifica:
- **Start Command**: `/app/backend/start.sh`
- O puede estar vacío si Railway usa el CMD del Dockerfile (que está bien)

### 3. Verificar Variables de Entorno

Ve a Settings → Variables y verifica que estas variables estén configuradas:

#### Variables Requeridas:
```bash
# Database (Railway las proporciona automáticamente cuando agregas PostgreSQL)
POSTGRES_HOST=<railway-provided>
POSTGRES_PORT=5432
POSTGRES_USER=<railway-provided>
POSTGRES_PASSWORD=<railway-provided>
POSTGRES_DB=railway

# CORS - IMPORTANTE: Agrega tu dominio de Vercel
CORS_ORIGINS=https://caria-way.com,https://www.caria-way.com,https://*.vercel.app

# API Keys
GEMINI_API_KEY=<tu-gemini-key>
FMP_API_KEY=<tu-fmp-api-key>
FRED_API_KEY=<tu-fred-api-key>
ALPHA_VANTAGE_API_KEY=<tu-alpha-vantage-key>

# JWT Secret (mínimo 32 caracteres)
JWT_SECRET_KEY=<tu-secret-key-min-32-chars>

# Caria Settings
CARIA_SETTINGS_PATH=/app/caria_data/configs/base.yaml

# Port (Railway lo configura automáticamente, pero puedes verificar)
PORT=8080
```

### 4. Verificar Base de Datos PostgreSQL

1. En Railway, verifica que tengas un servicio PostgreSQL agregado
2. Si no lo tienes:
   - Click en "+ New" en tu proyecto
   - Selecciona "Database" → "Add PostgreSQL"
   - Railway creará automáticamente las variables de entorno de conexión

3. Si ya tienes PostgreSQL pero está pausado:
   - Click en el servicio PostgreSQL
   - Debería reactivarse automáticamente cuando reactives el servicio principal

### 5. Reactivar el Servicio

#### Opción A: Desde el Dashboard
1. Ve a tu servicio "caria"
2. Si aparece un botón "Restart" o "Deploy", haz click
3. Railway debería iniciar un nuevo deployment

#### Opción B: Forzar Nuevo Deployment
1. Ve a la pestaña **Deployments**
2. Click en el botón **"Deploy"** o **"Redeploy"**
3. O haz un pequeño cambio en el código y haz push a GitHub:
   ```bash
   git commit --allow-empty -m "Trigger Railway redeploy"
   git push origin main
   ```

### 6. Verificar Logs

1. Ve a la pestaña **Logs** en Railway
2. Busca errores comunes:

#### Error: "Cannot find module 'caria'"
- **Solución**: Verifica que Root Directory esté vacío
- Verifica que PYTHONPATH esté configurado en start.sh

#### Error: "Connection refused" o "Database connection failed"
- **Solución**: Verifica que las variables POSTGRES_* estén configuradas
- Verifica que el servicio PostgreSQL esté activo

#### Error: "Port already in use" o "Address already in use"
- **Solución**: Railway maneja esto automáticamente, pero si persiste:
  - Verifica que PORT esté configurado correctamente
  - El start.sh usa `PORT=${PORT:-8080}` que debería funcionar

#### Error: "ModuleNotFoundError" o "ImportError"
- **Solución**: Verifica que caria-lib esté siendo copiado en el Dockerfile
- Revisa los logs de build para ver si hay errores al copiar archivos

### 7. Verificar Health Check

Railway está configurado para usar `/health` como healthcheck:
- Verifica que el endpoint `/health` esté funcionando
- Puedes probarlo manualmente: `https://tu-proyecto.up.railway.app/health`

Si el healthcheck falla, Railway puede marcar el servicio como offline.

### 8. Verificar URL del Servicio

Una vez que el servicio esté activo:
1. Ve a Settings → Networking
2. Copia la URL pública (algo como `https://caria-production.up.railway.app`)
3. Actualiza tu frontend en Vercel con esta URL:
   - Ve a Vercel → Tu Proyecto → Settings → Environment Variables
   - Actualiza `VITE_API_URL` con la nueva URL de Railway

## Comandos Útiles para Debugging

### Ver logs en tiempo real:
```bash
# Railway CLI (si lo tienes instalado)
railway logs
```

### Verificar que el servicio responde:
```bash
curl https://tu-proyecto.up.railway.app/health
```

### Verificar variables de entorno:
En Railway → Settings → Variables, todas deberían estar visibles.

## Si el Problema Persiste

1. **Revisa los logs completos** en Railway → Logs
2. **Verifica el último commit** que se desplegó exitosamente
3. **Compara la configuración** con la última vez que funcionó
4. **Contacta soporte de Railway** si el problema es con la cuenta/suscripción

## Checklist de Reactivación

- [ ] Cuenta de Railway activa y con créditos
- [ ] Proyecto visible en el dashboard
- [ ] Root Directory configurado correctamente (vacío o `.`)
- [ ] Dockerfile detectado en `backend/Dockerfile`
- [ ] Variables de entorno configuradas (especialmente POSTGRES_* y CORS_ORIGINS)
- [ ] Servicio PostgreSQL activo
- [ ] Deployment iniciado o servicio reiniciado
- [ ] Logs muestran que el servidor inició correctamente
- [ ] Health check responde en `/health`
- [ ] URL pública accesible
- [ ] Frontend actualizado con la nueva URL

## Próximos Pasos Después de Reactivar

1. Verifica que el backend responda: `curl https://tu-proyecto.up.railway.app/health`
2. Actualiza Vercel con la nueva URL si cambió
3. Prueba el chat WebSocket desde el frontend
4. Verifica que las APIs principales funcionen

