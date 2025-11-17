# 🚀 Crear Request Rápido en Insomnia (Sin Importar)

Si no encuentras los requests importados, aquí tienes cómo crear uno rápidamente:

## ✅ Crear Health Check (Sin Token)

1. Click en el botón **"+"** grande (arriba a la izquierda)
2. Selecciona **"HTTP Request"**
3. Configura:
   - **Name**: `Health Check`
   - **Method**: `GET` (dropdown a la izquierda de la URL)
   - **URL**: `http://localhost:8000/health`
4. Click en **"Send"** (botón morado/púrpura)
5. Deberías ver una respuesta con el estado de la API

## ✅ Crear Login (Para Obtener Token)

1. Click en **"+"** → **"HTTP Request"**
2. Configura:
   - **Name**: `Login`
   - **Method**: `POST`
   - **URL**: `http://localhost:8000/api/auth/login`
3. Click en el tab **"Body"**
4. Selecciona **"Form URL Encoded"**
5. Agrega estos campos:
   - `username` = `testuser`
   - `password` = `TestPassword123!`
6. Click en **"Send"**
7. **Copia el `access_token`** de la respuesta

## ✅ Crear Request con Token

Después de obtener el token:

1. Click en **"+"** → **"HTTP Request"**
2. Configura:
   - **Name**: `Get Prices`
   - **Method**: `POST`
   - **URL**: `http://localhost:8000/api/prices/realtime`
3. Click en el tab **"Body"**
4. Selecciona **"JSON"**
5. Pega esto:
   ```json
   {
     "tickers": ["AAPL", "MSFT", "GOOGL"]
   }
   ```
6. Click en el tab **"Header"**
7. Agrega un nuevo header:
   - **Name**: `Authorization`
   - **Value**: `Bearer TU_TOKEN_AQUI` (reemplaza TU_TOKEN_AQUI con tu token)
8. Click en **"Send"**

## 📋 Requests Completos para Copiar

### Health Check
```
GET http://localhost:8000/health
```

### Login
```
POST http://localhost:8000/api/auth/login
Body (Form URL Encoded):
  username: testuser
  password: TestPassword123!
```

### Get Prices (Requiere Token)
```
POST http://localhost:8000/api/prices/realtime
Header:
  Authorization: Bearer TU_TOKEN_AQUI
Body (JSON):
{
  "tickers": ["AAPL", "MSFT", "GOOGL"]
}
```

### Get Holdings (Requiere Token)
```
GET http://localhost:8000/api/holdings
Header:
  Authorization: Bearer TU_TOKEN_AQUI
```

### Create Holding (Requiere Token)
```
POST http://localhost:8000/api/holdings
Header:
  Authorization: Bearer TU_TOKEN_AQUI
  Content-Type: application/json
Body (JSON):
{
  "ticker": "AAPL",
  "quantity": 10,
  "average_cost": 150.0,
  "notes": "Apple Inc."
}
```

### Get Regime (Sin Token)
```
GET http://localhost:8000/api/regime/current
```

## 💡 Organizar Requests

Puedes crear folders para organizar:

1. Click derecho en el sidebar → **"New Folder"**
2. Nombra el folder (ej: "Auth", "Prices", "Holdings")
3. Arrastra los requests a los folders












