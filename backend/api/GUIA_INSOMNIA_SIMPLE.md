# Guía Rápida de Insomnia - Sin Environment

Si no encuentras el "Environment" en Insomnia, aquí tienes una guía alternativa más simple.

## 📥 Opción 1: Importar Colección (Recomendado)

### Paso 1: Abrir Insomnia
1. Abre Insomnia
2. Si es la primera vez, crea un nuevo "Design Document" o "Collection"

### Paso 2: Importar
1. Click en el botón **"+"** o **"Create"** (arriba a la izquierda)
2. Selecciona **"Import"** → **"From File"**
3. Busca y selecciona `insomnia_collection.json`
4. Click en **"Import"**

### Paso 3: Configurar URL Manualmente

Si no ves el environment, simplemente edita cada request manualmente:

1. Abre cualquier request (ej: "Health Check")
2. En la barra de URL, verás: `{{ _.base_url }}/health`
3. Reemplaza `{{ _.base_url }}` con `http://localhost:8000`
4. La URL debería quedar: `http://localhost:8000/health`

## 🔐 Configurar Autenticación Manualmente

### Paso 1: Hacer Login

1. Abre **"Authentication"** → **"Login"**
2. En la URL, reemplaza `{{ _.base_url }}` con `http://localhost:8000`
3. En el body, asegúrate de tener:
   ```
   username: testuser
   password: TestPassword123!
   ```
4. Click en **"Send"**
5. **Copia el `access_token`** de la respuesta

### Paso 2: Agregar Token a Otros Requests

Para cada request que necesite autenticación:

1. Abre el request
2. Ve al tab **"Auth"** (o **"Header"**)
3. Selecciona **"Bearer Token"** (si está disponible)
4. Pega tu token en el campo

O manualmente en **"Header"**:
1. Click en **"Header"**
2. Agrega un nuevo header:
   - Name: `Authorization`
   - Value: `Bearer TU_TOKEN_AQUI` (reemplaza TU_TOKEN_AQUI con el token que copiaste)

## 📋 Lista de Endpoints con URLs Completas

Aquí tienes todos los endpoints listos para copiar y pegar:

### Health Check (Sin autenticación)
```
GET http://localhost:8000/health
```

### Register
```
POST http://localhost:8000/api/auth/register
Content-Type: application/json

{
  "email": "test@example.com",
  "username": "testuser",
  "password": "TestPassword123!",
  "full_name": "Test User"
}
```

### Login
```
POST http://localhost:8000/api/auth/login
Content-Type: application/x-www-form-urlencoded

username=testuser&password=TestPassword123!
```

### Get Current User (Requiere token)
```
GET http://localhost:8000/api/auth/me
Authorization: Bearer TU_TOKEN_AQUI
```

### Get Realtime Prices (Batch)
```
POST http://localhost:8000/api/prices/realtime
Content-Type: application/json
Authorization: Bearer TU_TOKEN_AQUI

{
  "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"]
}
```

### Get Realtime Price (Single)
```
GET http://localhost:8000/api/prices/realtime/AAPL
Authorization: Bearer TU_TOKEN_AQUI
```

### List Holdings
```
GET http://localhost:8000/api/holdings
Authorization: Bearer TU_TOKEN_AQUI
```

### Create Holding
```
POST http://localhost:8000/api/holdings
Content-Type: application/json
Authorization: Bearer TU_TOKEN_AQUI

{
  "ticker": "AAPL",
  "quantity": 10,
  "average_cost": 150.0,
  "notes": "Apple Inc."
}
```

### Get Holdings with Prices
```
GET http://localhost:8000/api/holdings/with-prices
Authorization: Bearer TU_TOKEN_AQUI
```

### Delete Holding
```
DELETE http://localhost:8000/api/holdings/1
Authorization: Bearer TU_TOKEN_AQUI
```
(Reemplaza `1` con el ID real del holding)

### Quick Valuation
```
POST http://localhost:8000/api/valuation/AAPL
Content-Type: application/json
Authorization: Bearer TU_TOKEN_AQUI

{
  "ticker": "AAPL"
}
```

### Monte Carlo Valuation
```
POST http://localhost:8000/api/valuation/AAPL/monte-carlo
Content-Type: application/json
Authorization: Bearer TU_TOKEN_AQUI

{
  "ticker": "AAPL",
  "n_paths": 10000,
  "country_risk": "low"
}
```

### Get Current Regime (Sin autenticación)
```
GET http://localhost:8000/api/regime/current
```

### Factor Screen
```
POST http://localhost:8000/api/factors/screen
Content-Type: application/json
Authorization: Bearer TU_TOKEN_AQUI

{
  "top_n": 10,
  "regime": null,
  "date": null,
  "page": 1,
  "page_size": 10
}
```

### Challenge Thesis
```
POST http://localhost:8000/api/analysis/challenge
Content-Type: application/json
Authorization: Bearer TU_TOKEN_AQUI

{
  "thesis": "Creo que NVIDIA va a seguir subiendo porque la IA está en auge",
  "ticker": "NVDA"
}
```

## 🎯 Crear Requests Manualmente en Insomnia

Si prefieres crear los requests desde cero:

1. Click en **"+"** o **"Create"** → **"HTTP Request"**
2. Cambia el método (GET, POST, etc.)
3. Pega la URL completa
4. En **"Body"**, selecciona el tipo (JSON, form-urlencoded, etc.)
5. Pega el body correspondiente
6. En **"Header"**, agrega los headers necesarios
7. Click en **"Send"**

## 💡 Tips Rápidos

1. **Guarda el token**: Después del login, guarda el token en un archivo de texto para copiarlo fácilmente
2. **Duplica requests**: Click derecho en un request → "Duplicate" para crear variaciones rápidamente
3. **Organiza con folders**: Crea folders para organizar (Auth, Prices, Holdings, etc.)
4. **Usa el script de Python**: Si Insomnia te da problemas, usa `test_api_connection.py` que ya está configurado

## 🐛 Si Insomnia No Funciona

Usa el script de Python en su lugar:

```bash
cd services/api
python test_api_connection.py
```

Este script prueba todos los endpoints automáticamente y muestra los resultados con colores.












