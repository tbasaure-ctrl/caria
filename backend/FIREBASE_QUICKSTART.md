# 🚀 Firebase Quick Start - Guía Rápida

Guía rápida para empezar con Firebase en 5 minutos.

## ⚡ Setup Rápido (5 minutos)

### 1. Crear Proyecto Firebase (2 min)

1. Ve a [Firebase Console](https://console.firebase.google.com/)
2. Click **"Add project"**
3. Nombre: `wise-adviser`
4. Desactiva Analytics (opcional)
5. Click **"Create project"**
6. **Copia el Project ID** que aparece

### 2. Configurar `.firebaserc` (30 seg)

Edita `services/functions/.firebaserc`:

```json
{
  "projects": {
    "default": "TU-PROYECTO-ID-AQUI"
}
```

### 3. Instalar Firebase CLI (1 min)

```bash
npm install -g firebase-tools
firebase login
```

### 4. Configurar Variables (1 min)

```bash
cd services/functions
firebase functions:config:set gemini.api_key="TU_GEMINI_API_KEY"
firebase functions:config:set backend.url="http://localhost:8000"
```

### 5. Desplegar (30 seg)

```bash
firebase deploy --only functions
```

**¡Listo!** Tus funciones estarán en:
- `https://us-central1-TU-PROYECTO-ID.cloudfunctions.net/challengeThesis`

---

## 📱 Frontend (Opcional - si quieres Firebase Auth)

### 1. Instalar SDK

```bash
cd caria_data/caria-app
npm install firebase
```

### 2. Configurar

1. Firebase Console → Project Settings → General
2. Scroll hasta "Your apps" → Click `</>` (Web)
3. Copia la configuración
4. Pégala en `src/firebase/config.ts`

### 3. Usar

```typescript
import { loginWithEmail, getIdToken } from './firebase';

// Login
const userCredential = await loginWithEmail(email, password);
const token = await getIdToken();

// Usar token para llamar a tu backend
```

---

## 📚 Guías Completas

- **[FIREBASE_SETUP.md](./FIREBASE_SETUP.md)** - Guía completa paso a paso
- **[FIREBASE_FRONTEND_SETUP.md](./FIREBASE_FRONTEND_SETUP.md)** - Configuración detallada del frontend
- **[FIREBASE_BACKEND_AUTH.md](./FIREBASE_BACKEND_AUTH.md)** - Integración con tu backend

---

## ✅ Checklist Mínimo

- [ ] Proyecto Firebase creado
- [ ] Project ID en `.firebaserc`
- [ ] Firebase CLI instalado y logueado
- [ ] Variables de entorno configuradas
- [ ] Funciones desplegadas

---

## 🆘 Problemas Comunes

**"Project not found"**
→ Verifica el Project ID en `.firebaserc`

**"Functions require Blaze plan"**
→ Ve a Firebase Console → Billing → Upgrade a Blaze (gratis hasta cierto límite)

**"Permission denied"**
→ Ejecuta `firebase login` nuevamente

---

## 🎯 Próximos Pasos

1. ✅ Cloud Functions desplegadas → **Listo para usar**
2. 🔄 Frontend configurado → Ver `FIREBASE_FRONTEND_SETUP.md`
3. 🔐 Backend integrado → Ver `FIREBASE_BACKEND_AUTH.md`

