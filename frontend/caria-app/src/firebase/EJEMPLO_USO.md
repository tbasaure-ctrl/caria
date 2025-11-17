# 📖 Ejemplo de Uso de Firebase en tu App React

Ya tienes Firebase configurado. Aquí te muestro cómo usarlo:

## ✅ Lo que ya está hecho

- ✅ Firebase instalado (`npm install firebase`)
- ✅ Configuración en `src/firebase/config.ts` con tus credenciales
- ✅ Helpers de autenticación en `src/firebase/auth.ts`
- ✅ Analytics inicializado automáticamente

---

## 🚀 Ejemplo 1: Usar Firebase en un Componente

### Componente de Login

```typescript
// src/components/Login.tsx
import { useState } from 'react';
import { loginWithEmail, loginWithGoogle, getIdToken } from '../firebase';

export function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleEmailLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      // Login con Firebase
      const userCredential = await loginWithEmail(email, password);
      console.log('Usuario logueado:', userCredential.user);
      
      // Obtener token para enviar a tu backend
      const token = await getIdToken();
      console.log('Firebase token:', token);
      
      // Opcional: Enviar token a tu backend
      // const response = await fetch('http://localhost:8000/api/auth/firebase/verify', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ firebase_token: token })
      // });
      
    } catch (error) {
      console.error('Error de login:', error);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      const userCredential = await loginWithGoogle();
      console.log('Usuario logueado con Google:', userCredential.user);
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

---

## 🚀 Ejemplo 2: Hook para Estado de Autenticación

```typescript
// src/hooks/useAuth.ts
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

    return unsubscribe; // Cleanup
  }, []);

  const getToken = async () => {
    return await getIdToken();
  };

  return { user, loading, getToken };
}
```

### Usar el Hook

```typescript
// src/components/App.tsx
import { useAuth } from '../hooks/useAuth';

function App() {
  const { user, loading, getToken } = useAuth();

  if (loading) {
    return <div>Cargando...</div>;
  }

  if (!user) {
    return <Login />;
  }

  return (
    <div>
      <h1>Bienvenido, {user.email}</h1>
      <button onClick={async () => {
        const token = await getToken();
        // Usar token para llamar a tu backend
      }}>
        Obtener Token
      </button>
    </div>
  );
}
```

---

## 🚀 Ejemplo 3: Llamar a Firebase Functions

```typescript
import { getIdToken } from '../firebase';

async function challengeThesis(thesis: string, ticker?: string) {
  // Obtener token de Firebase
  const firebaseToken = await getIdToken();
  
  // Llamar a tu Firebase Function
  const response = await fetch(
    'https://us-central1-caria-9b633.cloudfunctions.net/challengeThesis',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${firebaseToken}` // Si necesitas autenticación
      },
      body: JSON.stringify({
        thesis,
        ticker,
        top_k: 5
      })
    }
  );
  
  const data = await response.json();
  return data;
}
```

---

## 📱 Ejemplo 4: Usar Analytics

Analytics ya está inicializado automáticamente. Si quieres trackear eventos personalizados:

```typescript
import { analytics } from '../firebase';
import { logEvent } from 'firebase/analytics';

// Trackear evento personalizado
if (analytics) {
  logEvent(analytics, 'thesis_challenged', {
    ticker: 'AAPL',
    thesis_length: thesis.length
  });
}
```

---

## 🔐 Ejemplo 5: Proteger Rutas

```typescript
// src/components/ProtectedRoute.tsx
import { useAuth } from '../hooks/useAuth';
import { Navigate } from 'react-router-dom'; // Si usas React Router

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();

  if (loading) {
    return <div>Cargando...</div>;
  }

  if (!user) {
    return <Navigate to="/login" />;
  }

  return <>{children}</>;
}
```

---

## 📝 Resumen de Funciones Disponibles

Desde `src/firebase/index.ts` puedes importar:

```typescript
import {
  // Configuración
  app,
  auth,
  messaging,
  analytics,
  
  // Autenticación
  registerWithEmail,
  loginWithEmail,
  loginWithGoogle,
  logout,
  resetPassword,
  getIdToken,
  onAuthChange,
  getCurrentUser,
  
  // Messaging
  getFCMToken
} from '../firebase';
```

---

## ✅ Próximos Pasos

1. **Crear componente de Login** usando `loginWithEmail` o `loginWithGoogle`
2. **Usar el hook `useAuth`** para proteger rutas
3. **Llamar a tus Firebase Functions** desde componentes
4. **Configurar FCM** si quieres notificaciones push (ver `FIREBASE_FCM_SETUP.md`)

---

## 🆘 Troubleshooting

**Error: "Firebase: Error (auth/network-request-failed)"**
- Verifica tu conexión a internet
- Verifica que las credenciales en `config.ts` sean correctas

**Error: "Firebase: Error (auth/popup-closed-by-user)"**
- El usuario cerró la ventana de popup de Google
- No es un error crítico, solo informa al usuario

**Analytics no funciona**
- Analytics solo funciona en producción o con HTTPS
- En desarrollo local puede no funcionar correctamente

