# 🎯 Crear Requests Manualmente en Insomnia - Paso a Paso

## ✅ Request 1: Health Check (El Más Fácil - Sin Token)

### Paso 1: Crear Request
1. Click en el botón **"+"** grande (arriba a la izquierda)
2. Selecciona **"HTTP Request"**

### Paso 2: Configurar
1. En el campo **"Name"** (arriba), escribe: `Health Check`
2. En el dropdown de método (izquierda de la URL), selecciona: **GET**
3. En el campo **URL**, escribe: `http://localhost:8000/health`

### Paso 3: Enviar
1. Click en el botón **"Send"** (botón morado/púrpura a la derecha de la URL)
2. Deberías ver una respuesta abajo con el estado de la API

---

## ✅ Request 2: Login (Para Obtener Token)

### Paso 1: Crear Request
1. Click en **"+"** → **"HTTP Request"**

### Paso 2: Configurar Método y URL
1. **Name**: `Login`
2. **Method**: `POST` (cambia el dropdown de GET a POST)
3. **URL**: `http://localhost:8000/api/auth/login`

### Paso 3: Configurar Body
1. Click en el tab **"Body"** (arriba, junto a Params, Headers, etc.)
2. Selecciona **"Form URL Encoded"** (no JSON)
3. Agrega estos campos uno por uno:
   - Click en **"+ Add"** o el campo vacío
   - **Name**: `username`
   - **Value**: `testuser`
   - Click en **"+ Add"** otra vez
   - **Name**: `password`
   - **Value**: `TestPassword123!`

Debería verse así:
```
username: testuser
password: TestPassword123!
```

### Paso 4: Enviar y Copiar Token
1. Click en **"Send"**
2. En la respuesta (abajo), busca:
   ```json
   {
     "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
   }
   ```
3. **COPIA TODO EL TOKEN** (desde `eyJ` hasta el final)
4. Guárdalo en un archivo de texto o memorízalo

---

## ✅ Request 3: Get Prices (Requiere Token)

### Paso 1: Crear Request
1. Click en **"+"** → **"HTTP Request"**

### Paso 2: Configurar Método y URL
1. **Name**: `Get Prices`
2. **Method**: `POST`
3. **URL**: `http://localhost:8000/api/prices/realtime`

### Paso 3: Configurar Headers
1. Click en el tab **"Header"**
2. Agrega un nuevo header:
   - Click en **"+ Add"** o el campo vacío
   - **Name**: `Authorization`
   - **Value**: `Bearer TU_TOKEN_AQUI` (reemplaza TU_TOKEN_AQUI con el token que copiaste)
   - Ejemplo: `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

### Paso 4: Configurar Body
1. Click en el tab **"Body"**
2. Selecciona **"JSON"** (no Form URL Encoded)
3. En el editor JSON, pega esto:
   ```json
   {
     "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"]
   }
   ```

### Paso 5: Enviar
1. Click en **"Send"**
2. Deberías ver los precios en tiempo real

---

## ✅ Request 4: Get Holdings (Requiere Token)

### Paso 1: Crear Request
1. Click en **"+"** → **"HTTP Request"**

### Paso 2: Configurar
1. **Name**: `Get Holdings`
2. **Method**: `GET`
3. **URL**: `http://localhost:8000/api/holdings`

### Paso 3: Agregar Token
1. Click en tab **"Header"**
2. Agrega:
   - **Name**: `Authorization`
   - **Value**: `Bearer TU_TOKEN_AQUI` (usa tu token real)

### Paso 4: Enviar
1. Click en **"Send"**
2. Verás tus holdings (o lista vacía si no tienes ninguno)

---

## ✅ Request 5: Create Holding (Requiere Token)

### Paso 1: Crear Request
1. Click en **"+"** → **"HTTP Request"**

### Paso 2: Configurar Método y URL
1. **Name**: `Create Holding`
2. **Method**: `POST`
3. **URL**: `http://localhost:8000/api/holdings`

### Paso 3: Configurar Headers
1. Click en tab **"Header"**
2. Agrega DOS headers:
   - **Header 1**:
     - Name: `Authorization`
     - Value: `Bearer TU_TOKEN_AQUI`
   - **Header 2**:
     - Name: `Content-Type`
     - Value: `application/json`

### Paso 4: Configurar Body
1. Click en tab **"Body"**
2. Selecciona **"JSON"**
3. Pega esto:
   ```json
   {
     "ticker": "AAPL",
     "quantity": 10,
     "average_cost": 150.0,
     "notes": "Apple Inc. - Long term hold"
   }
   ```
4. Puedes cambiar `ticker`, `quantity`, `average_cost` y `notes` según quieras

### Paso 5: Enviar
1. Click en **"Send"**
2. Deberías ver el holding creado en la respuesta

---

## ✅ Request 6: Get Holdings with Prices (Requiere Token)

### Paso 1: Crear Request
1. Click en **"+"** → **"HTTP Request"**

### Paso 2: Configurar
1. **Name**: `Get Holdings with Prices`
2. **Method**: `GET`
3. **URL**: `http://localhost:8000/api/holdings/with-prices`

### Paso 3: Agregar Token
1. Click en tab **"Header"**
2. Agrega:
   - **Name**: `Authorization`
   - **Value**: `Bearer TU_TOKEN_AQUI`

### Paso 4: Enviar
1. Click en **"Send"**
2. Verás tus holdings con precios en tiempo real y métricas calculadas

---

## ✅ Request 7: Get Regime (Sin Token - Fácil)

### Paso 1: Crear Request
1. Click en **"+"** → **"HTTP Request"**

### Paso 2: Configurar
1. **Name**: `Get Regime`
2. **Method**: `GET`
3. **URL**: `http://localhost:8000/api/regime/current`

### Paso 3: Enviar
1. Click en **"Send"** (no necesita headers ni body)
2. Verás el régimen macro actual

---

## ✅ Request 8: Quick Valuation (Requiere Token)

### Paso 1: Crear Request
1. Click en **"+"** → **"HTTP Request"**

### Paso 2: Configurar Método y URL
1. **Name**: `Quick Valuation`
2. **Method**: `POST`
3. **URL**: `http://localhost:8000/api/valuation/AAPL`
   - Puedes cambiar `AAPL` por otro ticker (ej: `MSFT`, `GOOGL`)

### Paso 3: Configurar Headers
1. Click en tab **"Header"**
2. Agrega DOS headers:
   - **Authorization**: `Bearer TU_TOKEN_AQUI`
   - **Content-Type**: `application/json`

### Paso 4: Configurar Body
1. Click en tab **"Body"**
2. Selecciona **"JSON"**
3. Pega:
   ```json
   {
     "ticker": "AAPL"
   }
   ```
   - Cambia `AAPL` si cambiaste la URL

### Paso 5: Enviar
1. Click en **"Send"**
2. Verás la valuación de la empresa

---

## 📋 Resumen de URLs y Métodos

| Request | Method | URL | ¿Necesita Token? |
|---------|--------|-----|-------------------|
| Health Check | GET | `http://localhost:8000/health` | ❌ No |
| Login | POST | `http://localhost:8000/api/auth/register` | ❌ No |
| Register | POST | `http://localhost:8000/api/auth/login` | ❌ No |
| Get Prices | POST | `http://localhost:8000/api/prices/realtime` | ✅ Sí |
| Get Holdings | GET | `http://localhost:8000/api/holdings` | ✅ Sí |
| Create Holding | POST | `http://localhost:8000/api/holdings` | ✅ Sí |
| Holdings with Prices | GET | `http://localhost:8000/api/holdings/with-prices` | ✅ Sí |
| Get Regime | GET | `http://localhost:8000/api/regime/current` | ❌ No |
| Quick Valuation | POST | `http://localhost:8000/api/valuation/AAPL` | ✅ Sí |

## 💡 Tips Importantes

1. **Token**: Después del login, copia el `access_token` completo (es muy largo)
2. **Headers**: Para requests con token, siempre agrega `Authorization: Bearer TU_TOKEN`
3. **Body JSON**: Asegúrate de seleccionar "JSON" en el tab Body, no "Form URL Encoded"
4. **Content-Type**: Para POST con JSON, agrega header `Content-Type: application/json`

## 🐛 Si Algo No Funciona

1. **Verifica que la API esté corriendo**: `python start_api.py`
2. **Verifica la URL**: Debe ser `http://localhost:8000`
3. **Verifica el token**: Debe ser completo y empezar con `eyJ`
4. **Verifica los headers**: Deben estar en el tab "Header", no en "Auth"

## 🚀 Orden Recomendado para Probar

1. ✅ Health Check (sin token)
2. ✅ Register (sin token)
3. ✅ Login (sin token) → **COPIA EL TOKEN**
4. ✅ Get Regime (sin token)
5. ✅ Get Prices (con token)
6. ✅ Get Holdings (con token)
7. ✅ Create Holding (con token)
8. ✅ Get Holdings with Prices (con token)
9. ✅ Quick Valuation (con token)












