# ⚡ Quick Start: Deploy a Vercel en 5 Minutos

## 🚀 Método Rápido (Dashboard)

### 1. Ve a Vercel
https://vercel.com/new

### 2. Conecta tu Repo
- Click "Import Git Repository"
- Selecciona tu repo
- **Root Directory:** `caria_data/caria-app`
- **Framework:** Vite (auto-detectado)

### 3. Configura Variables
En "Environment Variables", agrega:
- `VITE_API_URL` = `https://tu-backend.com` (o deja `http://localhost:8000` si aún no tienes backend en producción)

### 4. Deploy
Click "Deploy" y espera 2-3 minutos.

**¡Listo!** Tu app estará en `https://tu-proyecto.vercel.app`

---

## 🔧 Método CLI (Si Prefieres Terminal)

```bash
# 1. Instalar Vercel CLI
npm install -g vercel

# 2. Login
vercel login

# 3. Ir al directorio del frontend
cd caria_data/caria-app

# 4. Deploy
vercel

# 5. Configurar variables (cuando te pregunte)
# VITE_API_URL: https://tu-backend.com
```

---

## ✅ Verificar que Funciona

1. Abre la URL que te da Vercel
2. Prueba el login
3. Prueba el Analysis Tool (usa Firebase Functions)
4. Prueba otros widgets (usan tu backend)

---

## 🔄 Actualizar Variables Después

```bash
vercel env add VITE_API_URL
# Selecciona: Production, Preview, Development
# Ingresa el valor
```

O desde Dashboard: Settings → Environment Variables

---

## 📝 Notas Importantes

- ✅ Firebase Functions seguirán funcionando (URLs hardcodeadas)
- ✅ El backend debe estar accesible públicamente para que funcionen Login, Chat, Portfolio, etc.
- ✅ CORS debe estar configurado en tu backend para permitir el dominio de Vercel

---

## 🆘 Problemas Comunes

**"Build failed"**
→ Verifica que `package.json` tenga `"build": "vite build"`

**"CORS error"**
→ Agrega tu dominio de Vercel a `CORS_ORIGINS` en tu backend

**"WebSocket no funciona"**
→ Tu backend debe estar accesible públicamente (no localhost)

---

¡Eso es todo! 🎉

