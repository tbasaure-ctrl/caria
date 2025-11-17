# 🔥 Cómo Usar el Login con Firebase

He creado un nuevo componente `LoginModalFirebase` que integra Firebase Authentication. Aquí te explico cómo usarlo:

## 📁 Archivos Creados

1. **`components/LoginModalFirebase.tsx`** - Componente de login con Firebase
2. **`src/hooks/useAuth.ts`** - Hook para manejar estado de autenticación

## 🚀 Opción 1: Reemplazar el Login Actual

Si quieres usar Firebase Authentication completamente, reemplaza `LoginModal` por `LoginModalFirebase` en `App.tsx`:

```typescript
// App.tsx
import { LoginModalFirebase } from './components/LoginModalFirebase';

// ... en el JSX:
{isLoginModalOpen && (
  <LoginModalFirebase 
    onClose={() => setLoginModalOpen(false)} 
    onSuccess={handleLoginSuccess}
    onSwitchToRegister={handleShowRegister}
  />
)}
```

## 🚀 Opción 2: Usar Ambos (Híbrido)

Puedes mantener ambos modales y dejar que el usuario elija:

```typescript
// App.tsx
import { LoginModal } from './components/LoginModal';
import { LoginModalFirebase } from './components/LoginModalFirebase';

const [loginMethod, setLoginMethod] = useState<'traditional' | 'firebase'>('traditional');

// En el JSX:
{isLoginModalOpen && (
  <>
    {loginMethod === 'traditional' ? (
      <LoginModal 
        onClose={() => setLoginModalOpen(false)} 
        onSuccess={handleLoginSuccess}
        onSwitchToRegister={handleShowRegister}
      />
    ) : (
      <LoginModalFirebase 
        onClose={() => setLoginModalOpen(false)} 
        onSuccess={handleLoginSuccess}
        onSwitchToRegister={handleShowRegister}
      />
    )}
    <button onClick={() => setLoginMethod(loginMethod === 'traditional' ? 'firebase' : 'traditional')}>
      Cambiar método de login
    </button>
  </>
)}
```

## 🔐 Usar el Hook useAuth

El hook `useAuth` te permite verificar el estado de autenticación en cualquier componente:

```typescript
import { useAuth } from '../src/hooks/useAuth';

function MyComponent() {
  const { user, loading, isAuthenticated, getToken } = useAuth();

  if (loading) {
    return <div>Cargando...</div>;
  }

  if (!isAuthenticated) {
    return <div>No estás logueado</div>;
  }

  return (
    <div>
      <p>Bienvenido, {user?.email}</p>
      <button onClick={async () => {
        const token = await getToken();
        console.log('Firebase token:', token);
      }}>
        Obtener Token
      </button>
    </div>
  );
}
```

## 🔄 Flujo de Autenticación

El componente `LoginModalFirebase` hace lo siguiente:

1. **Usuario se autentica con Firebase** (Email/Password o Google)
2. **Obtiene token de Firebase**
3. **Intenta enviar token a tu backend** (`/api/auth/firebase/verify`)
4. **Si el backend responde**, guarda el token JWT de tu backend
5. **Si el backend no responde**, usa solo el token de Firebase

### ¿Qué significa esto?

- **Si tienes el endpoint `/api/auth/firebase/verify` en tu backend**: El usuario obtendrá un token JWT de tu backend (compatible con tu sistema actual)
- **Si NO tienes el endpoint**: El usuario usará solo el token de Firebase (funciona igual, pero necesitarás adaptar tu backend)

## 📝 Crear el Endpoint en el Backend (Opcional)

Si quieres que el backend verifique tokens de Firebase, crea el endpoint siguiendo la guía:
- `services/FIREBASE_BACKEND_AUTH.md`

## ✅ Características del LoginModalFirebase

- ✅ Login con Email/Password
- ✅ Login con Google (un solo click)
- ✅ Manejo de errores específicos de Firebase
- ✅ Loading states
- ✅ Compatible con tu diseño actual (mismo estilo)
- ✅ Fallback si el backend no tiene endpoint de Firebase
- ✅ Soporte para tecla Escape para cerrar

## 🎨 Personalización

El componente usa las mismas clases CSS que tu `LoginModal` original, así que se verá igual. Si quieres cambiar el estilo, edita las clases en `LoginModalFirebase.tsx`.

## 🆘 Troubleshooting

**Error: "Firebase: Error (auth/popup-blocked)"**
- El navegador bloqueó el popup de Google
- Pide al usuario que permita popups para tu sitio

**Error: "Backend no tiene endpoint de Firebase"**
- Es normal si no has creado el endpoint aún
- El componente funcionará solo con Firebase tokens

**No se muestra el botón de Google**
- Verifica que Google Sign-In esté habilitado en Firebase Console
- Ve a Authentication → Sign-in method → Google → Enable

---

¿Necesitas ayuda para integrarlo? ¡Dime qué opción prefieres!

