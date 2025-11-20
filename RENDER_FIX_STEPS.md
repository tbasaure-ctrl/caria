# Pasos para Arreglar el Error de Build en Render

## Error Actual
```
COPY services/start.sh /app/services/start.sh
Exited with status 1
```

## Solución: Pasos Concretos

### Paso 1: Verificar que el Dockerfile correcto existe
- ✅ El Dockerfile correcto está en: `backend/Dockerfile`
- ✅ Usa: `backend/start.sh` (no `services/start.sh`)

### Paso 2: Actualizar render.yaml
Ya está actualizado con:
- `dockerfilePath: backend/Dockerfile`
- `dockerContext: .`

### Paso 3: En Render Dashboard

1. **Ve a tu servicio**: https://dashboard.render.com → caria-api
2. **Settings** → **Build & Deploy**
3. **Dockerfile Path**: Debe ser `backend/Dockerfile`
4. **Root Directory**: Debe estar vacío (o `.`)
5. **Save Changes**

### Paso 4: Verificar Variables de Entorno

En **Environment** tab, asegúrate de tener:

```
DATABASE_URL=<tu-neon-connection-string>
LLAMA_API_KEY=gsk_****************************************************
REDDIT_CLIENT_ID=<tu-client-id>
REDDIT_CLIENT_SECRET=<tu-client-secret>
FMP_API_KEY=<tu-fmp-key>
JWT_SECRET_KEY=<genera-uno-nuevo>
```

### Paso 5: Manual Deploy

1. Click **"Manual Deploy"** dropdown
2. Select **"Deploy latest commit"**
3. Espera a que termine el build

### Paso 6: Verificar Logs

Si falla, revisa los logs:
- Busca errores de `COPY` o `services/start.sh`
- Debe usar `backend/start.sh` en su lugar

## Si Sigue Fallando

### Opción A: Eliminar Dockerfile viejo de la raíz
Si hay un `Dockerfile` en la raíz del repo que usa `services/`, elimínalo o renómbralo.

### Opción B: Forzar uso del Dockerfile correcto
En Render Settings → Build & Deploy:
- **Dockerfile Path**: `backend/Dockerfile` (sin `./`)
- **Build Command**: Dejar vacío
- **Start Command**: `/app/backend/start.sh`


