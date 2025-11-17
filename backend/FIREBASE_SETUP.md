# 🔥 Guía Completa de Configuración Firebase para Wise Adviser

Esta guía te llevará paso a paso para integrar Firebase en tu proyecto.

## 📋 Tabla de Contenidos

1. [Paso 1: Crear Proyecto Firebase](#paso-1-crear-proyecto-firebase)
2. [Paso 2: Configurar Productos Firebase](#paso-2-configurar-productos-firebase)
3. [Paso 3: Registrar tu App](#paso-3-registrar-tu-app)
4. [Paso 4: Conectar tu Código](#paso-4-conectar-tu-código)
5. [Paso 5: Configurar Variables de Entorno](#paso-5-configurar-variables-de-entorno)

---

## Paso 1: Crear Proyecto Firebase

### 1.1 Crear proyecto en Firebase Console

1. Ve a [Firebase Console](https://console.firebase.google.com/)
2. Click en **"Add project"** o **"Crear proyecto"**
3. Ingresa el nombre del proyecto: `wise-adviser` (o el que prefieras)
4. Click en **"Continue"**
5. **Desactiva** Google Analytics (opcional, puedes activarlo después)
6. Click en **"Create project"**
7. Espera a que se cree el proyecto (30-60 segundos)
8. Click en **"Continue"**

### 1.2 Obtener Project ID

1. En la página principal del proyecto, verás el **Project ID**
2. **Copia este ID** - lo necesitarás para `.firebaserc`

---

## Paso 2: Configurar Productos Firebase

### 2.1 Firebase Authentication 🔐

1. En el menú lateral, click en **"Authentication"**
2. Click en **"Get started"**
3. Ve a la pestaña **"Sign-in method"**
4. Habilita los métodos que necesites:

   **Email/Password (Recomendado para empezar):**
   - Click en **"Email/Password"**
   - Activa **"Enable"**
   - Click en **"Save"**

   **Google Sign-In (Opcional):**
   - Click en **"Google"**
   - Activa **"Enable"**
   - Ingresa tu email de soporte
   - Click en **"Save"**

   **Otros métodos (Opcional):**
   - GitHub, Facebook, Twitter, etc.

### 2.2 Cloud Functions for Firebase ⚡

Ya está configurado en `services/functions/`, pero necesitas:

1. En el menú lateral, click en **"Functions"**
2. Si es la primera vez, acepta los términos
3. Firebase configurará automáticamente Cloud Functions

**Nota:** Necesitas tener un plan Blaze (pay-as-you-go) para Cloud Functions, pero el tier gratuito es muy generoso.

### 2.3 Firebase Cloud Messaging (FCM) 📱

1. En el menú lateral, click en **"Cloud Messaging"**
2. Si es la primera vez, acepta los términos
3. Ve a la pestaña **"Cloud Messaging API (Legacy)"**
4. Anota el **Server key** (lo necesitarás para el backend)

---

## Paso 3: Registrar tu App

### 3.1 Registrar Web App (Frontend React)

1. En la página principal del proyecto, click en el ícono **`</>`** (Web)
2. Ingresa un **App nickname**: `wise-adviser-web`
3. **NO** marques "Also set up Firebase Hosting" (por ahora)
4. Click en **"Register app"**
5. **Copia la configuración de Firebase** - se verá así:

```javascript
const firebaseConfig = {
  apiKey: "AIza...",
  authDomain: "tu-proyecto.firebaseapp.com",
  projectId: "tu-proyecto-id",
  storageBucket: "tu-proyecto.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abc123"
};
```

6. **Guarda esta configuración** - la necesitarás para el frontend
7. Click en **"Continue to console"**

### 3.2 Configurar Dominios Autorizados (CORS)

1. Ve a **Authentication** → **Settings** → **Authorized domains**
2. Agrega tus dominios:
   - `localhost` (ya está por defecto)
   - Tu dominio de producción (ej: `wiseadviser.com`)

---

## Paso 4: Conectar tu Código

### 4.1 Actualizar `.firebaserc`

Edita `services/functions/.firebaserc`:

```json
{
  "projects": {
    "default": "TU-PROYECTO-ID-AQUI"
}
```

Reemplaza `TU-PROYECTO-ID-AQUI` con el Project ID que copiaste en el Paso 1.2.

### 4.2 Instalar Firebase CLI

```bash
npm install -g firebase-tools
```

### 4.3 Login a Firebase

```bash
firebase login
```

Esto abrirá tu navegador para autenticarte.

### 4.4 Inicializar Firebase Functions (si no lo has hecho)

```bash
cd services/functions
firebase init functions
```

**Selecciona:**
- ✅ Use an existing project
- ✅ Python
- ✅ Install dependencies now? → Yes

---

## Paso 5: Configurar Variables de Entorno

### 5.1 Configurar Variables para Cloud Functions

```bash
cd services/functions

# Gemini API
firebase functions:config:set gemini.api_key="TU_GEMINI_API_KEY"
firebase functions:config:set gemini.api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Backend tradicional (para RAG)
firebase functions:config:set backend.url="http://localhost:8000"  # Desarrollo
# En producción: "https://tu-backend.com"

# Llama (opcional)
firebase functions:config:set llama.api_key="TU_LLAMA_API_KEY"
firebase functions:config:set llama.api_url="TU_LLAMA_API_URL"
firebase functions:config:set llama.model_name="llama-3.1-70b-instruct"
```

### 5.2 Verificar Configuración

```bash
firebase functions:config:get
```

Deberías ver todas tus variables configuradas.

---

## 🚀 Desplegar Cloud Functions

```bash
cd services/functions
firebase deploy --only functions
```

Esto desplegará tus funciones y te dará las URLs:
- `https://us-central1-TU-PROYECTO-ID.cloudfunctions.net/challengeThesis`
- `https://us-central1-TU-PROYECTO-ID.cloudfunctions.net/analyzeWithGemini`

---

## 📱 Configurar Frontend (React)

### 5.1 Instalar Firebase SDK

```bash
cd caria_data/caria-app
npm install firebase
```

### 5.2 Crear archivo de configuración

Crea `caria_data/caria-app/src/firebase/config.ts` (ver siguiente sección)

---

## ✅ Checklist de Configuración

- [ ] Proyecto Firebase creado
- [ ] Project ID copiado y actualizado en `.firebaserc`
- [ ] Firebase Authentication habilitado (Email/Password)
- [ ] Cloud Functions inicializado
- [ ] Variables de entorno configuradas
- [ ] Firebase CLI instalado y logueado
- [ ] Funciones desplegadas
- [ ] Frontend configurado (siguiente paso)

---

## 🔗 Próximos Pasos

1. [Configurar Firebase en Frontend](./FIREBASE_FRONTEND_SETUP.md)
2. [Integrar Firebase Auth en Backend](./FIREBASE_BACKEND_AUTH.md)
3. [Configurar FCM para Notificaciones](./FIREBASE_FCM_SETUP.md)

---

## 🆘 Troubleshooting

### Error: "Project not found"
- Verifica que el Project ID en `.firebaserc` sea correcto
- Ejecuta `firebase projects:list` para ver tus proyectos

### Error: "Permission denied"
- Ejecuta `firebase login` nuevamente
- Verifica que tengas permisos en el proyecto

### Error: "Functions require Blaze plan"
- Ve a Firebase Console → Billing
- Actualiza a plan Blaze (pay-as-you-go)
- El tier gratuito es muy generoso, no te preocupes por costos iniciales

### Variables de entorno no funcionan
- Usa `firebase functions:config:get` para verificar
- En Firebase Functions v2, las variables se configuran diferente (usa secrets)

---

## 📚 Recursos

- [Firebase Documentation](https://firebase.google.com/docs)
- [Cloud Functions Documentation](https://firebase.google.com/docs/functions)
- [Firebase Authentication](https://firebase.google.com/docs/auth)

