# ¿Por qué el límite de 72 bytes en las contraseñas?

## La Limitación de bcrypt

El límite de **72 bytes** no es una decisión arbitraria de Caria, sino una **limitación técnica de bcrypt**, la librería que usamos para hashear contraseñas de forma segura.

### ¿Por qué bcrypt tiene este límite?

1. **Diseño original**: bcrypt fue diseñado en 1999 cuando las contraseñas largas no eran comunes
2. **Optimización de rendimiento**: Limitar a 72 bytes permite optimizaciones en el algoritmo
3. **Estándar de la industria**: Es el límite estándar que usan la mayoría de sistemas

### ¿Qué significa 72 bytes?

- **Caracteres ASCII simples** (a-z, A-Z, 0-9, símbolos básicos): **72 caracteres**
- **Caracteres acentuados** (á, é, ñ, etc.): **~36 caracteres** (cada uno ocupa 2 bytes)
- **Emojis**: **~18-24 caracteres** (cada uno ocupa 3-4 bytes)

## ¿Es suficiente 72 bytes?

**Para la mayoría de casos, SÍ:**

- Una contraseña de 20-30 caracteres con mayúsculas, minúsculas, números y símbolos es muy segura
- Ejemplo: `MyP@ssw0rd!2024#Secure` = 24 caracteres = 24 bytes ✅
- Ejemplo: `EstaEsMiContraseña123!@#` = 27 caracteres = 27 bytes ✅

**Problemas comunes:**

- Contraseñas generadas por gestores que incluyen emojis o caracteres especiales complejos
- Copiar/pegar que agrega caracteres invisibles
- Contraseñas muy largas (más de 50-60 caracteres)

## Alternativas si necesitas contraseñas más largas

### Opción 1: Usar Argon2 (Recomendado para nuevos proyectos)

Argon2 es más moderno y **NO tiene límite de longitud**:

```python
# Cambiar de bcrypt a argon2
import argon2

def hash_password(password: str) -> str:
    ph = argon2.PasswordHasher()
    return ph.hash(password)  # Sin límite de longitud
```

**Ventajas:**
- ✅ Sin límite de longitud
- ✅ Más seguro contra ataques modernos
- ✅ Ganador del Password Hashing Competition 2015

**Desventajas:**
- ⚠️ Requiere migrar contraseñas existentes
- ⚠️ Cambio en la base de datos

### Opción 2: Pre-hashear con SHA-256 antes de bcrypt

```python
import hashlib
import bcrypt

def hash_password(password: str) -> str:
    # Pre-hashear con SHA-256 para eliminar límite práctico
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    # Luego hashear con bcrypt (siempre será 64 bytes, dentro del límite)
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_hash.encode('utf-8'), salt)
    return hashed.decode('utf-8')
```

**Ventajas:**
- ✅ Compatible con bcrypt existente
- ✅ Sin límite práctico de longitud
- ✅ No requiere migración

**Desventajas:**
- ⚠️ Menos seguro que Argon2
- ⚠️ Dos pasos de hashing

### Opción 3: Truncar contraseñas largas (NO recomendado)

```python
def hash_password(password: str) -> str:
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]  # Truncar
    # ... resto del código
```

**Desventajas:**
- ❌ Reduce seguridad (contraseñas diferentes pueden tener el mismo hash)
- ❌ Confuso para usuarios

## Recomendación para Caria

**Opción A: Mantener bcrypt con validación (Actual)**
- ✅ Simple y estándar
- ✅ Compatible con la mayoría de sistemas
- ✅ Suficiente para 99% de usuarios
- ⚠️ Limita a 72 bytes

**Opción B: Migrar a Argon2 (Futuro)**
- ✅ Sin límites
- ✅ Más seguro
- ⚠️ Requiere migración de contraseñas existentes

**Opción C: Pre-hashear con SHA-256 (Compromiso)**
- ✅ Sin límite práctico
- ✅ Compatible con bcrypt
- ⚠️ Menos seguro que Argon2

## ¿Qué hacer ahora?

1. **Para usuarios**: Usar contraseñas de 20-50 caracteres sin emojis
2. **Para desarrollo**: Considerar migrar a Argon2 en el futuro si muchos usuarios necesitan contraseñas muy largas

## Referencias

- [bcrypt Wikipedia](https://en.wikipedia.org/wiki/Bcrypt)
- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [Argon2 Specification](https://github.com/P-H-C/phc-winner-argon2)

