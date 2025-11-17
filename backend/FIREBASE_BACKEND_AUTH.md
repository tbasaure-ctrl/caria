# 🔐 Integración de Firebase Authentication en Backend

Guía para verificar tokens de Firebase Authentication en tu backend FastAPI.

## 📋 Opciones de Integración

Tienes dos opciones principales:

### Opción 1: Solo Verificar Tokens (Recomendado para empezar)
- Usuario se autentica con Firebase
- Backend solo verifica que el token sea válido
- Mantienes tu sistema de usuarios actual

### Opción 2: Migración Completa
- Usuario se autentica con Firebase
- Backend crea usuario en tu DB si no existe
- Usas Firebase como única fuente de verdad

Esta guía cubre la **Opción 1** (más simple y menos invasiva).

---

## 📦 Paso 1: Instalar Firebase Admin SDK

```bash
cd services/api
pip install firebase-admin
```

O agrega a `requirements.txt`:

```txt
firebase-admin>=6.0.0
```

---

## 🔑 Paso 2: Obtener Service Account Key

1. Firebase Console → **Project Settings** → **Service accounts**
2. Click en **"Generate new private key"**
3. Descarga el archivo JSON
4. **IMPORTANTE:** Guárdalo en un lugar seguro y **NO** lo subas a Git

### Ubicación recomendada:

```
services/api/
├── firebase_service_account.json  # ← Este archivo (en .gitignore)
└── ...
```

Agrega a `.gitignore`:

```gitignore
firebase_service_account.json
*.json
!package.json
```

---

## ⚙️ Paso 3: Configurar Firebase Admin

Crea `services/api/firebase_admin.py`:

```python
"""
Firebase Admin SDK Configuration
"""
import os
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, auth

# Inicializar Firebase Admin (solo una vez)
_firebase_app = None

def initialize_firebase_admin():
    """
    Inicializa Firebase Admin SDK.
    Debe llamarse una vez al inicio de la aplicación.
    """
    global _firebase_app
    
    if _firebase_app is not None:
        return _firebase_app
    
    # Buscar archivo de service account
    current_dir = Path(__file__).parent
    service_account_path = current_dir / "firebase_service_account.json"
    
    # También buscar en variable de entorno
    env_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    if env_path:
        service_account_path = Path(env_path)
    
    if not service_account_path.exists():
        raise FileNotFoundError(
            f"Firebase service account file not found at {service_account_path}. "
            "Download it from Firebase Console → Project Settings → Service accounts"
        )
    
    cred = credentials.Certificate(str(service_account_path))
    _firebase_app = firebase_admin.initialize_app(cred)
    
    return _firebase_app


def verify_firebase_token(id_token: str) -> dict:
    """
    Verifica un token de Firebase Authentication.
    
    Args:
        id_token: Token ID de Firebase (obtenido del frontend)
    
    Returns:
        dict: Información decodificada del token (uid, email, etc.)
    
    Raises:
        ValueError: Si el token es inválido
    """
    try:
        initialize_firebase_admin()
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        raise ValueError(f"Token inválido: {str(e)}")


def get_user_by_uid(uid: str) -> dict:
    """
    Obtiene información de usuario desde Firebase Auth.
    
    Args:
        uid: Firebase User ID
    
    Returns:
        dict: Información del usuario
    """
    try:
        initialize_firebase_admin()
        user = auth.get_user(uid)
        return {
            "uid": user.uid,
            "email": user.email,
            "display_name": user.display_name,
            "email_verified": user.email_verified,
        }
    except Exception as e:
        raise ValueError(f"Usuario no encontrado: {str(e)}")
```

---

## 🔌 Paso 4: Crear Endpoint para Verificar Tokens

Crea `services/api/routes/firebase_auth.py`:

```python
"""
Firebase Authentication Integration
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from api.firebase_admin import verify_firebase_token
from api.dependencies import get_auth_service
from caria.services.auth_service import AuthService

router = APIRouter(prefix="/api/auth/firebase", tags=["Firebase Auth"])


class FirebaseTokenRequest(BaseModel):
    firebase_token: str


class FirebaseAuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict


@router.post("/verify", response_model=FirebaseAuthResponse)
async def verify_firebase_auth(
    request: FirebaseTokenRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Verifica un token de Firebase y genera tokens JWT de tu backend.
    
    Flujo:
    1. Verifica que el token de Firebase sea válido
    2. Busca o crea usuario en tu base de datos
    3. Genera tokens JWT de tu backend
    """
    try:
        # 1) Verificar token de Firebase
        decoded_token = verify_firebase_token(request.firebase_token)
        firebase_uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token de Firebase no contiene email"
            )
        
        # 2) Buscar usuario en tu DB por email o crear uno nuevo
        # (Adapta esto a tu lógica de negocio)
        try:
            user = auth_service.get_user_by_email(email)
        except ValueError:
            # Usuario no existe, crear uno nuevo
            # Usa el display_name de Firebase si está disponible
            display_name = decoded_token.get("name") or email.split("@")[0]
            user = auth_service.register_user(
                email=email,
                username=email.split("@")[0],  # O genera un username único
                password=None,  # No hay password con Firebase Auth
                full_name=display_name
            )
        
        # 3) Generar tokens JWT de tu backend
        from caria.services.auth_service import AuthService
        access_token = AuthService.create_access_token(user)
        refresh_token = AuthService.create_refresh_token(user)
        auth_service.store_refresh_token(user.id, refresh_token)
        
        return FirebaseAuthResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            user={
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando autenticación: {str(e)}"
        )
```

---

## 🔗 Paso 5: Registrar Router en App Principal

En `services/api/app.py`, agrega:

```python
from api.routes.firebase_auth import router as firebase_auth_router

# ... después de otros routers ...
app.include_router(firebase_auth_router)
```

---

## 🧪 Paso 6: Probar la Integración

### Desde el Frontend:

```typescript
import { getIdToken } from './firebase';

async function loginWithFirebase() {
  // 1. Usuario se autentica con Firebase (ya lo tienes)
  const firebaseToken = await getIdToken();
  
  // 2. Enviar token a tu backend
  const response = await fetch('http://localhost:8000/api/auth/firebase/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ firebase_token: firebaseToken })
  });
  
  const data = await response.json();
  
  // 3. Guardar tokens de tu backend
  localStorage.setItem('access_token', data.access_token);
  localStorage.setItem('refresh_token', data.refresh_token);
  
  // Ahora puedes usar estos tokens para llamar a tus endpoints protegidos
}
```

---

## 🔒 Paso 7: Middleware para Endpoints Protegidos (Opcional)

Si quieres que algunos endpoints acepten tokens de Firebase directamente:

```python
from fastapi import Depends, HTTPException, status
from api.firebase_admin import verify_firebase_token

async def get_current_user_firebase(
    authorization: str = Header(None)
) -> dict:
    """
    Dependency que acepta tokens de Firebase o JWT de tu backend.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token requerido"
        )
    
    token = authorization.split(" ")[1]
    
    # Intentar verificar como token de Firebase
    try:
        decoded = verify_firebase_token(token)
        return {"firebase_user": decoded}
    except ValueError:
        # Si falla, intentar como JWT de tu backend (tu lógica actual)
        # ...
        pass
```

---

## ✅ Checklist

- [ ] Firebase Admin SDK instalado
- [ ] Service Account Key descargado y guardado
- [ ] `firebase_admin.py` creado
- [ ] Endpoint `/api/auth/firebase/verify` creado
- [ ] Router registrado en `app.py`
- [ ] Frontend actualizado para usar el nuevo endpoint
- [ ] Probado con un usuario real

---

## 🆘 Troubleshooting

### Error: "Service account file not found"
- Verifica que el archivo esté en la ubicación correcta
- O configura `FIREBASE_SERVICE_ACCOUNT_PATH` como variable de entorno

### Error: "Token inválido"
- Verifica que el token no haya expirado (tokens de Firebase expiran después de 1 hora)
- Asegúrate de obtener un token fresco con `getIdToken(true)` en el frontend

### Error: "Permission denied"
- Verifica que el Service Account tenga permisos de "Firebase Authentication Admin"

---

## 📚 Recursos

- [Firebase Admin SDK Python](https://firebase.google.com/docs/admin/setup)
- [Verificar Tokens ID](https://firebase.google.com/docs/auth/admin/verify-id-tokens)

