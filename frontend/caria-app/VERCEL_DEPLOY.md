# 🚀 Guía de Despliegue en Vercel

Guía completa para desplegar Wise Adviser en Vercel.

## 📋 Prerequisitos

1. **Cuenta de Vercel**: Crea una en [vercel.com](https://vercel.com)
2. **Vercel CLI** (opcional, para deploy desde terminal):
   ```bash
   npm install -g vercel
   ```

---

## 🎯 Opción 1: Deploy desde Vercel Dashboard (Más Fácil)

### Paso 1: Conectar tu Repositorio

1. Ve a [Vercel Dashboard](https://vercel.com/dashboard)
2. Click en **"Add New..."** → **"Project"**
3. Conecta tu repositorio de GitHub/GitLab/Bitbucket
4. Selecciona el proyecto `wise_adviser_cursor_context`

### Paso 2: Configurar el Proyecto

**Root Directory:**
```
caria_data/caria-app
```

**Framework Preset:**
```
Vite
```

**Build Command:**
```
npm run build
```

**Output Directory:**
```
dist
```

**Install Command:**
```
npm install
```

### Paso 3: Configurar Variables de Entorno

En la sección **"Environment Variables"**, agrega:

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `VITE_API_URL` | `https://tu-backend.com` o `http://localhost:8000` | URL de tu backend FastAPI |
| `VITE_GEMINI_API_KEY` | `tu-gemini-api-key` | API Key de Gemini (opcional, si lo usas directamente) |

**Nota:** Para producción, usa la URL de tu backend desplegado. Si usas Firebase Functions, puedes dejar `VITE_API_URL` apuntando a tu backend tradicional.

### Paso 4: Deploy

Click en **"Deploy"** y espera a que termine.

---

## 🎯 Opción 2: Deploy desde CLI

### Paso 1: Instalar Vercel CLI

```bash
npm install -g vercel
```

### Paso 2: Login

```bash
vercel login
```

### Paso 3: Navegar al Directorio

```bash
cd caria_data/caria-app
```

### Paso 4: Deploy

```bash
# Deploy a preview (staging)
vercel

# Deploy a producción
vercel --prod
```

### Paso 5: Configurar Variables de Entorno

```bash
# Configurar variables
vercel env add VITE_API_URL
# Ingresa: https://tu-backend.com

vercel env add VITE_GEMINI_API_KEY
# Ingresa: tu-gemini-api-key
```

---

## 🔧 Configuración de Variables de Entorno

### Desarrollo Local

Crea `.env.local`:

```env
VITE_API_URL=http://localhost:8000
VITE_GEMINI_API_KEY=tu-api-key-local
```

### Producción en Vercel

Configura en Vercel Dashboard → Settings → Environment Variables:

- **Production:**
  - `VITE_API_URL` = `https://tu-backend-produccion.com`
  - `VITE_GEMINI_API_KEY` = `tu-api-key-produccion`

- **Preview:**
  - `VITE_API_URL` = `http://localhost:8000` (o tu backend de staging)
  - `VITE_GEMINI_API_KEY` = `tu-api-key-staging`

---

## 🌐 Configurar Dominio Personalizado (Opcional)

1. Ve a Vercel Dashboard → Tu Proyecto → Settings → Domains
2. Agrega tu dominio (ej: `wiseadviser.com`)
3. Sigue las instrucciones para configurar DNS

---

## 🔄 Integración con Firebase Functions

Tu frontend ya está configurado para usar Firebase Functions para el Analysis Tool. Las URLs están hardcodeadas en `firebaseFunctionsService.ts`:

```typescript
export const FIREBASE_FUNCTIONS = {
  CHALLENGE_THESIS: 'https://us-central1-caria-9b633.cloudfunctions.net/challenge_thesis',
  ANALYZE_WITH_GEMINI: 'https://us-central1-caria-9b633.cloudfunctions.net/analyze_with_gemini',
};
```

**Esto seguirá funcionando** desde Vercel sin cambios.

---

## 📝 Actualizar URLs para Producción

Si tu backend está en producción, actualiza `VITE_API_URL` en Vercel:

1. Vercel Dashboard → Tu Proyecto → Settings → Environment Variables
2. Edita `VITE_API_URL`
3. Cambia a: `https://tu-backend-produccion.com`
4. Redeploy

---

## 🧪 Probar el Deploy

Después del deploy, tu app estará en:
- **Preview:** `https://tu-proyecto-vercel.vercel.app`
- **Producción:** `https://tu-dominio.com` (si configuraste dominio)

---

## 🆘 Troubleshooting

### Error: "Build failed"
- Verifica que `package.json` tenga el script `build`
- Revisa los logs en Vercel Dashboard

### Error: "Environment variable not found"
- Verifica que las variables empiecen con `VITE_`
- Vercel solo expone variables que empiezan con `VITE_` al frontend

### CORS Errors
- Verifica que tu backend tenga configurado CORS para el dominio de Vercel
- Agrega `https://tu-proyecto.vercel.app` a `CORS_ORIGINS` en tu backend

### WebSocket no funciona
- WebSockets requieren que el backend esté accesible públicamente
- Verifica que tu backend esté desplegado y accesible desde internet

---

## 📚 Recursos

- [Vercel Documentation](https://vercel.com/docs)
- [Vite + Vercel](https://vercel.com/docs/frameworks/vite)
- [Environment Variables](https://vercel.com/docs/concepts/projects/environment-variables)

---

## ✅ Checklist de Deploy

- [ ] Cuenta de Vercel creada
- [ ] Repositorio conectado
- [ ] Variables de entorno configuradas
- [ ] Build exitoso
- [ ] Dominio configurado (opcional)
- [ ] Backend accesible públicamente
- [ ] CORS configurado en backend
- [ ] Firebase Functions funcionando

---

## 🎉 ¡Listo!

Una vez desplegado, tendrás:
- ✅ Frontend en Vercel (CDN global, SSL automático)
- ✅ Firebase Functions para Analysis Tool
- ✅ Backend tradicional para Login, Chat, Portfolio, etc.

