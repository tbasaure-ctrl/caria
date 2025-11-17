# 🔍 Diagnóstico Rápido de Problemas

## Problema Principal: API No Responde o Da Errores

### ✅ Paso 1: Verificar que la API esté corriendo

Abre una **nueva terminal** y ejecuta:

```bash
cd services/api
python start_api.py
```

**Deberías ver**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Si ves errores, cópialos y compártelos.

### ✅ Paso 2: Probar Health Check Manualmente

En otra terminal (o en el navegador):

```bash
# Opción A: Con curl
curl http://localhost:8000/health

# Opción B: En el navegador
# Abre: http://localhost:8000/health
```

**Deberías ver**:
```json
{
  "status": "ok",
  "database": "available",
  "auth": "available"
}
```

### ✅ Paso 3: Probar Login Manualmente

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=Test123!"
```

**Si funciona**, deberías ver un `access_token`.

**Si da error 500**, revisa los logs de la API para ver el error específico.

## 🔧 Soluciones Comunes

### Problema: "Connection refused" o timeout

**Causa**: La API no está corriendo

**Solución**:
1. Ve a la terminal donde debería estar corriendo la API
2. Si no está corriendo, ejecuta: `python start_api.py`
3. Espera a ver "Application startup complete"
4. Prueba de nuevo

### Problema: Error 500 en Login

**Causa**: Error en el servidor (puede ser BD, contraseña incorrecta, etc.)

**Solución**:
1. **Revisa los logs** de la API (en la terminal donde corre)
2. Busca el error específico (aparecerá en rojo)
3. Posibles causas:
   - Usuario no existe → Regístralo primero
   - Contraseña incorrecta → Usa la contraseña correcta
   - Error de BD → Verifica que PostgreSQL esté corriendo

### Problema: Usuario ya existe pero login falla

**Solución**: El usuario existe pero con otra contraseña. Opciones:

**Opción A**: Eliminar y recrear el usuario
```sql
psql -U caria_user -d caria
DELETE FROM users WHERE username = 'testuser';
```

**Opción B**: Usar otro usuario
- Cambia `testuser` por `testuser2` en los scripts

**Opción C**: Resetear contraseña (si tienes acceso a la BD)
```sql
-- Necesitarías el hash de la contraseña, mejor elimina y recrea
```

## 📋 Credenciales Correctas

Para todos los scripts y ejemplos, usa:

- **Username**: `testuser`
- **Password**: `Test123!` (8 caracteres)

## 🚀 Crear Request en Insomnia (Mientras Arreglamos)

Mientras diagnosticamos, puedes crear requests manualmente en Insomnia:

### Request 1: Health Check

1. Click en **"+"** → **"HTTP Request"**
2. **Method**: `GET`
3. **URL**: `http://localhost:8000/health`
4. Click **"Send"**

### Request 2: Login

1. Click en **"+"** → **"HTTP Request"**
2. **Method**: `POST`
3. **URL**: `http://localhost:8000/api/auth/login`
4. **Body** tab → **Form URL Encoded**:
   - `username`: `testuser`
   - `password`: `Test123!`
5. Click **"Send"**

Si funciona, copia el `access_token` de la respuesta.

## 📞 Próximos Pasos

1. **Verifica que la API esté corriendo** (Paso 1 arriba)
2. **Prueba Health Check** (Paso 2 arriba)
3. **Si Health Check funciona pero Login no**, revisa los logs de la API
4. **Comparte los errores** que veas en los logs para ayudarte mejor

## 💡 Script de Diagnóstico Completo

Ejecuta este script para ver qué está pasando:

```bash
python test_login_simple.py
```

Este script te dirá exactamente dónde está el problema.












