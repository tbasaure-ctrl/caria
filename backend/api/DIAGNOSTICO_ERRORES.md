# Diagnóstico de Errores de Autenticación

## Cómo identificar el error real

### 1. Abre la Consola del Navegador

1. Abre tu navegador
2. Presiona `F12` o `Ctrl+Shift+I` (Windows) / `Cmd+Option+I` (Mac)
3. Ve a la pestaña **Console**
4. Intenta registrarte o hacer login
5. Mira los mensajes de error que aparecen

### 2. Revisa la Pestaña Network

1. En las herramientas de desarrollador, ve a la pestaña **Network**
2. Intenta registrarte o hacer login
3. Busca la request a `/api/auth/register` o `/api/auth/login`
4. Haz clic en ella
5. Ve a la pestaña **Response** para ver qué responde el servidor

### 3. Tipos de Errores Comunes

#### Error: "Failed to fetch"
**Causa**: No puede conectar al servidor
**Soluciones**:
- Verifica que la API esté corriendo (`python start_api.py`)
- Verifica que la URL sea correcta (por defecto: `http://localhost:8000`)
- Verifica que no haya problemas de CORS

#### Error: "Password is too long. Maximum length is 72 bytes..."
**Causa**: Contraseña excede 72 bytes cuando se codifica
**Soluciones**:
- Usa una contraseña más corta (máximo ~50 caracteres si usas caracteres simples)
- Evita emojis y caracteres especiales complejos
- Si usas caracteres acentuados, reduce la longitud

#### Error: "Password must be at least 8 characters long"
**Causa**: Contraseña muy corta
**Soluciones**:
- Usa al menos 8 caracteres

#### Error: "Username already exists" o "Email already registered"
**Causa**: El usuario o email ya está registrado
**Soluciones**:
- Usa un username o email diferente
- O intenta hacer login en lugar de registrarte

### 4. Verificar que la API está funcionando

```bash
# En otra terminal, prueba el endpoint directamente:
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","username":"testuser","password":"test123456"}'
```

Si esto funciona, el problema está en el frontend.
Si esto falla, el problema está en el backend.

### 5. Verificar logs de la API

Mira la terminal donde está corriendo la API. Deberías ver:
- Requests entrantes
- Errores detallados
- Stack traces si hay excepciones

## Mensajes de Error Mejorados

Ahora los mensajes de error son más claros:

- ✅ "Password must be at least 8 characters long" - Contraseña muy corta
- ✅ "Password is too long. Maximum length is 72 bytes..." - Contraseña muy larga o con caracteres especiales
- ✅ "Unable to connect to the server..." - Problema de conexión
- ✅ Mensajes específicos del servidor - Errores de validación o del servidor

## Próximos Pasos

1. **Abre la consola del navegador** (F12)
2. **Intenta registrarte/login**
3. **Copia el error exacto** que aparece
4. **Comparte el error** para que pueda ayudarte mejor

