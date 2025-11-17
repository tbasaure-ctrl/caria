# Guía de Uso de Insomnia para CARIA API

Esta guía te ayudará a configurar Insomnia para probar todos los endpoints de CARIA.

## 📥 Importar Colección

### Paso 1: Abrir Insomnia

1. Abre Insomnia (descárgalo desde https://insomnia.rest si no lo tienes)
2. Crea un nuevo workspace o usa uno existente

### Paso 2: Importar Colección

1. Click en **"Create"** → **"Import"** → **"From File"**
2. Selecciona el archivo `insomnia_collection.json` desde `services/api/`
3. La colección "CARIA API" aparecerá en tu workspace

## ⚙️ Configurar Variables de Entorno

### Paso 1: Abrir Environment

1. Click en el dropdown de environments (arriba a la derecha)
2. Selecciona **"Base Environment"**

### Paso 2: Configurar Variables

Edita las siguientes variables:

- **`base_url`**: `http://localhost:8000` (o la URL donde corre tu API)
- **`access_token`**: Déjalo vacío por ahora, se llenará automáticamente después del login
- **`holding_id`**: ID de un holding (se actualizará después de crear uno)

### Paso 3: Guardar

Click en **"Done"** para guardar los cambios.

## 🔐 Flujo de Autenticación

### 1. Registrar Usuario (Opcional)

Si es la primera vez:
- Abre **"Authentication"** → **"Register User"**
- Click en **"Send"**
- Si el usuario ya existe, puedes saltar este paso

### 2. Login

1. Abre **"Authentication"** → **"Login"**
2. Ajusta las credenciales si es necesario:
   - `username`: `testuser`
   - `password`: `TestPassword123!`
3. Click en **"Send"**
4. **IMPORTANTE**: Copia el `access_token` de la respuesta
5. Ve a **"Base Environment"** y pega el token en `access_token`
6. Guarda el environment

### 3. Verificar Token

1. Abre **"Authentication"** → **"Get Current User"**
2. Click en **"Send"**
3. Deberías ver la información del usuario

## 📋 Probar Endpoints

### Health Check (Sin autenticación)

1. Abre **"Health"** → **"Health Check"**
2. Click en **"Send"**
3. Deberías ver el estado de la API

### Precios en Tiempo Real

1. Abre **"Prices"** → **"Get Realtime Prices (Batch)"**
2. Puedes modificar los tickers en el body: `["AAPL", "MSFT", "GOOGL"]`
3. Click en **"Send"**
4. Deberías ver los precios actuales

### Holdings

1. **List Holdings**: Ver todos tus holdings
2. **Create Holding**: Crear un nuevo holding
   - Modifica el body con tu ticker, cantidad, costo promedio
3. **Get Holdings with Prices**: Ver holdings con precios en tiempo real y métricas
4. **Delete Holding**: Eliminar un holding (actualiza `holding_id` en el environment primero)

### Valuación

1. **Quick Valuation**: Valuación rápida de una empresa
   - Cambia el ticker en la URL y en el body
2. **Monte Carlo Valuation**: Simulación Monte Carlo
   - Puede tardar unos segundos
   - Ajusta `n_paths` para pruebas más rápidas (1000 en lugar de 10000)

### Régimen Macro

1. Abre **"Regime"** → **"Get Current Regime"**
2. Click en **"Send"**
3. Verás las probabilidades de régimen actual

### Factor Screening

1. Abre **"Factors"** → **"Factor Screen"**
2. Ajusta `top_n` para ver más/menos empresas
3. Click en **"Send"**

### Análisis RAG

1. Abre **"Analysis"** → **"Challenge Thesis"**
2. Modifica el `thesis` con tu propia tesis de inversión
3. Asegúrate de incluir un `ticker` válido
4. Click en **"Send"**
5. Puede tardar unos segundos (RAG con LLM)

## 🔄 Automatizar Token Refresh

Insomnia puede actualizar automáticamente el token después del login:

1. Abre **"Authentication"** → **"Login"**
2. Click en el tab **"Tests"** (abajo)
3. Agrega este código:

```javascript
const data = JSON.parse(response.body);
if (data.access_token) {
    insomnia.environment.set('access_token', data.access_token);
} else if (data.token && data.token.access_token) {
    insomnia.environment.set('access_token', data.token.access_token);
}
```

Ahora cada vez que hagas login, el token se actualizará automáticamente.

## 🎨 Personalizar Requests

Puedes duplicar cualquier request y modificarlo:

1. Click derecho en un request
2. Selecciona **"Duplicate"**
3. Modifica el nombre, URL, body, etc.
4. Guarda

## 📊 Ver Respuestas

Insomnia muestra:
- **Status Code**: Código HTTP de la respuesta
- **Time**: Tiempo de respuesta
- **Body**: Respuesta completa (JSON formateado)
- **Headers**: Headers de respuesta

## 🐛 Troubleshooting

### Error: "Unauthorized" (401)

- Verifica que el token esté configurado en el environment
- Haz login nuevamente y actualiza el token

### Error: "Connection refused"

- Verifica que la API esté corriendo en `http://localhost:8000`
- O cambia `base_url` en el environment a la URL correcta

### Error: "CORS policy"

- Esto es normal si pruebas desde el navegador
- Insomnia no tiene problemas de CORS
- Si pruebas desde la UI, asegúrate de que CORS esté configurado correctamente

### Token expirado

- Simplemente haz login nuevamente
- O configura el refresh automático (ver arriba)

## 💡 Tips

1. **Usa variables**: Puedes usar `{{ _.base_url }}` y `{{ _.access_token }}` en cualquier request
2. **Organiza con folders**: Ya están organizados por categoría
3. **Duplica requests**: Para crear variaciones rápidamente
4. **Guarda responses**: Click derecho → "Save Response" para guardar ejemplos
5. **Exporta collection**: Para compartir con tu equipo

## 📚 Recursos

- [Documentación de Insomnia](https://docs.insomnia.rest/)
- [API Docs de CARIA](http://localhost:8000/docs) (cuando la API esté corriendo)

