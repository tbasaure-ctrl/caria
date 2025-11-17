# 📱 Configuración de Firebase en el Frontend (React)

Guía para integrar Firebase Authentication y Cloud Messaging en tu aplicación React.

## 📦 Paso 1: Instalar Dependencias

```bash
cd caria_data/caria-app
npm install firebase
```

## ⚙️ Paso 2: Configurar Firebase

### 2.1 Obtener Configuración de Firebase

1. Ve a [Firebase Console](https://console.firebase.google.com/)
2. Selecciona tu proyecto
3. Click en el ícono de configuración ⚙️ → **Project settings**
4. Scroll hasta **"Your apps"**
5. Si no tienes una web app, click en **`</>`** para agregar una
6. Copia la configuración que se ve así:

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

### 2.2 Actualizar `src/firebase/config.ts`

Edita `caria_data/caria-app/src/firebase/config.ts` y reemplaza los valores:

```typescript
const firebaseConfig = {
  apiKey: "TU_API_KEY_AQUI",           // ← Reemplaza aquí
  authDomain: "TU-PROYECTO-ID.firebaseapp.com",
  projectId: "TU-PROYECTO-ID",
  storageBucket: "TU-PROYECTO-ID.appspot.com",
  messagingSenderId: "TU_SENDER_ID",
  appId: "TU_APP_ID"
};
```

## 🔐 Paso 3: Usar Firebase Authentication

### Ejemplo: Componente de Login

```typescript
import { useState } from 'react';
import { loginWithEmail, loginWithGoogle, getIdToken } from '../firebase';

function LoginComponent() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleEmailLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const userCredential = await loginWithEmail(email, password);
      const token = await getIdToken();
      
      // Enviar token a tu backend para verificación
      const response = await fetch('http://localhost:8000/api/auth/firebase', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ firebase_token: token })
      });
      
      const data = await response.json();
      // Guardar token de tu backend
      localStorage.setItem('access_token', data.access_token);
    } catch (error) {
      console.error('Error de login:', error);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      const userCredential = await loginWithGoogle();
      const token = await getIdToken();
      // Similar al login con email
    } catch (error) {
      console.error('Error con Google:', error);
    }
  };

  return (
    <div>
      <form onSubmit={handleEmailLogin}>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
        />
        <button type="submit">Login</button>
      </form>
      
      <button onClick={handleGoogleLogin}>
        Login con Google
      </button>
    </div>
  );
}
```

### Ejemplo: Hook para Estado de Autenticación

```typescript
// hooks/useAuth.ts
import { useEffect, useState } from 'react';
import { onAuthChange, getCurrentUser, getIdToken } from '../firebase';
import type { User } from 'firebase/auth';

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthChange((user) => {
      setUser(user);
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  const getToken = async () => {
    return await getIdToken();
  };

  return { user, loading, getToken };
}
```

## 📱 Paso 4: Configurar Cloud Messaging (FCM)

### 4.1 Obtener VAPID Key

1. Firebase Console → **Cloud Messaging**
2. Scroll hasta **"Web configuration"**
3. Click en **"Generate key pair"** (si no tienes uno)
4. Copia el **Key pair** (VAPID key)

### 4.2 Actualizar `config.ts`

Edita `src/firebase/config.ts` y reemplaza:

```typescript
const token = await getToken(messaging, {
  vapidKey: 'TU_VAPID_KEY_AQUI' // ← Pega tu VAPID key aquí
});
```

### 4.3 Crear Service Worker

Crea `public/firebase-messaging-sw.js`:

```javascript
// public/firebase-messaging-sw.js
importScripts('https://www.gstatic.com/firebasejs/10.7.1/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.7.1/firebase-messaging-compat.js');

firebase.initializeApp({
  apiKey: "TU_API_KEY",
  authDomain: "TU-PROYECTO-ID.firebaseapp.com",
  projectId: "TU-PROYECTO-ID",
  storageBucket: "TU-PROYECTO-ID.appspot.com",
  messagingSenderId: "TU_SENDER_ID",
  appId: "TU_APP_ID"
});

const messaging = firebase.messaging();

messaging.onBackgroundMessage((payload) => {
  console.log('Mensaje recibido en background:', payload);
  
  const notificationTitle = payload.notification.title;
  const notificationOptions = {
    body: payload.notification.body,
    icon: '/icon-192x192.png' // Ajusta la ruta de tu ícono
  };

  self.registration.showNotification(notificationTitle, notificationOptions);
});
```

### 4.4 Registrar Service Worker

En tu `main.tsx` o `App.tsx`:

```typescript
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/firebase-messaging-sw.js')
    .then((registration) => {
      console.log('Service Worker registrado:', registration);
    })
    .catch((error) => {
      console.error('Error registrando Service Worker:', error);
    });
}
```

## 🔄 Paso 5: Integrar con tu Backend Actual

Si quieres mantener tu sistema de autenticación actual pero agregar Firebase como opción:

### Opción A: Usar Firebase solo para Cloud Functions

No necesitas cambiar tu autenticación actual. Solo usa las Firebase Functions para análisis.

### Opción B: Híbrido - Firebase Auth + Tu Backend

1. Usuario se autentica con Firebase
2. Obtienes el Firebase ID token
3. Envías el token a tu backend
4. Tu backend verifica el token con Firebase Admin SDK
5. Tu backend genera su propio JWT (opcional)

Ver: `FIREBASE_BACKEND_AUTH.md` para más detalles.

## ✅ Checklist

- [ ] Firebase SDK instalado (`npm install firebase`)
- [ ] Configuración actualizada en `src/firebase/config.ts`
- [ ] Componente de login creado
- [ ] Hook `useAuth` implementado
- [ ] VAPID key configurado (si usas FCM)
- [ ] Service Worker creado y registrado (si usas FCM)

## 🚀 Próximos Pasos

1. [Integrar Firebase Auth en Backend](./FIREBASE_BACKEND_AUTH.md)
2. [Configurar FCM para Notificaciones](./FIREBASE_FCM_SETUP.md)

