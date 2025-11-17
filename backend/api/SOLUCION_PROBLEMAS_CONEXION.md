# 🔧 Solución a Problemas de Conexión

## Problemas Detectados

### 1. ❌ Password demasiado largo (273 bytes)

**Error**: `Password is too long. Maximum length is 72 bytes when encoded`

**Solución**: Ya corregido en el script. Ahora usa `Test123!` (8 caracteres, ~8 bytes)

### 2. ❌ Login da error 500

**Posibles causas**:
- El usuario existe pero la contraseña es diferente
- Hay un error en el servidor (revisa los logs)
- Problema con la base de datos

**Solución**:

#### Opción A: Crear usuario nuevo con contraseña conocida

```bash
# Usa este script simple para probar solo el login
python test_login_simple.py
```

#### Opción B: Eliminar usuario existente y recrearlo

Si tienes acceso a PostgreSQL:

```sql
-- Conectarte a la base de datos
psql -U caria_user -d caria

-- Eliminar el usuario
DELETE FROM users WHERE username = 'testuser';

-- Salir
\q
```

Luego ejecuta el script de nuevo.

#### Opción C: Usar otro usuario

Edita `test_api_connection.py` y cambia:

```python
TEST_USER = {
    "email": "test2@caria.com",
    "username": "testuser2",  # Cambia esto
    "password": "Test123!",
    "full_name": "Test User 2"
}
```

### 3. ❌ API no responde (Connection refused)

**Solución**: Inicia la API primero:

```bash
cd services/api
python start_api.py
```

O verifica que esté corriendo:

```bash
curl http://localhost:8000/health
```

### 4. ✅ Regime funciona correctamente

El endpoint de régimen está funcionando bien, así que la API básica está OK.

## 🚀 Pasos para Resolver

### Paso 1: Verificar que la API esté corriendo

```bash
# En una terminal
cd services/api
python start_api.py
```

Deberías ver:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Paso 2: Probar login simple

En otra terminal:

```bash
cd services/api
python test_login_simple.py
```

Este script te dirá exactamente qué está pasando con el login.

### Paso 3: Revisar logs del servidor

Si el login da error 500, revisa los logs en la terminal donde corre la API. Busca errores en rojo.

### Paso 4: Crear usuario manualmente (si es necesario)

Si el usuario no existe o tiene problemas, créalo manualmente:

```bash
# Usa curl o Insomnia para hacer register
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@caria.com",
    "username": "testuser",
    "password": "Test123!",
    "full_name": "Test User"
  }'
```

## 📋 Credenciales Actuales del Script

El script ahora usa:
- **Username**: `testuser`
- **Password**: `Test123!` (8 caracteres, seguro y corto)

## 🔍 Debugging

### Ver qué usuario existe en la BD

```sql
psql -U caria_user -d caria
SELECT username, email, created_at FROM users;
```

### Ver logs detallados

En la terminal donde corre la API, deberías ver errores detallados si algo falla.

### Probar endpoints individualmente

```bash
# Health
curl http://localhost:8000/health

# Regime (sin auth)
curl http://localhost:8000/api/regime/current

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=Test123!"
```

## 💡 Si Nada Funciona

1. **Reinicia la API**: Detén y vuelve a iniciar `python start_api.py`
2. **Verifica variables de entorno**: Asegúrate de que `.env` esté configurado
3. **Verifica PostgreSQL**: Asegúrate de que la base de datos esté corriendo
4. **Usa Insomnia manualmente**: Crea los requests manualmente siguiendo `CREAR_REQUESTS_MANUAL.md`












