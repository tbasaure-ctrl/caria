# Firebase Functions para Wise Adviser

Migración de endpoints que usan Gemini API a funciones serverless con integración RAG.

## 🚀 Setup inicial

### 1. Instalar Firebase CLI

```bash
npm install -g firebase-tools
```

### 2. Login a Firebase

```bash
firebase login
```

### 3. Inicializar proyecto Firebase

```bash
cd services/functions
firebase init functions
```

**Opciones a seleccionar:**
- ✅ Use an existing project (o crea uno nuevo)
- ✅ Python como lenguaje
- ✅ Acepta las opciones por defecto

### 4. Configurar variables de entorno

```bash
# API Keys de Gemini
firebase functions:config:set gemini.api_key="TU_GEMINI_API_KEY"
firebase functions:config:set gemini.api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Llama (opcional - fallback)
firebase functions:config:set llama.api_key="TU_LLAMA_API_KEY"
firebase functions:config:set llama.api_url="TU_LLAMA_API_URL"
firebase functions:config:set llama.model_name="llama-3.1-70b-instruct"

# URL del backend tradicional (para RAG)
firebase functions:config:set backend.url="http://localhost:8000"  # Desarrollo local
# En producción: "https://tu-backend.com" o la IP de tu servidor
```

**Nota:** En Firebase Functions v2, las variables de entorno se configuran diferente. Si usas v2, usa:

```bash
firebase functions:secrets:set GEMINI_API_KEY
firebase functions:secrets:set BACKEND_URL
```

### 5. Desplegar funciones

```bash
firebase deploy --only functions
```

## 📍 Endpoints disponibles

### POST `/challengeThesis`

Reemplaza: `POST /api/analysis/challenge`

**Request:**
```json
{
  "thesis": "Mi tesis de inversión sobre Apple...",
  "ticker": "AAPL",
  "top_k": 5
}
```

**Response:**
```json
{
  "thesis": "...",
  "retrieved_chunks": [
    {
      "id": "...",
      "score": 0.85,
      "title": "...",
      "source": "...",
      "content": "...",
      "metadata": {...}
    }
  ],
  "critical_analysis": "Análisis crítico detallado...",
  "identified_biases": ["Confirmation bias", "..."],
  "recommendations": ["Recomendación 1", "..."],
  "confidence_score": 0.8
}
```

### POST `/analyzeWithGemini`

Análisis directo con Gemini (sin RAG)

**Request:**
```json
{
  "prompt": "Analiza esta inversión..."
}
```

**Response:**
```json
{
  "raw_text": "Respuesta del modelo..."
}
```

## 🌐 URLs después del deploy

Las funciones estarán disponibles en:
- `https://us-central1-TU-PROYECTO-ID.cloudfunctions.net/challengeThesis`
- `https://us-central1-TU-PROYECTO-ID.cloudfunctions.net/analyzeWithGemini`

Reemplaza `TU-PROYECTO-ID` con el ID de tu proyecto de Firebase.

## 🧪 Desarrollo local

### Opción 1: Emulador de Firebase

```bash
firebase emulators:start --only functions
```

Las funciones estarán en:
- `http://localhost:5001/TU-PROYECTO-ID/us-central1/challengeThesis`
- `http://localhost:5001/TU-PROYECTO-ID/us-central1/analyzeWithGemini`

**Importante:** Asegúrate de que tu backend tradicional esté corriendo en `http://localhost:8000` para que RAG funcione.

### Opción 2: Variables de entorno locales

Crea un archivo `.env.local` en `services/functions/`:

```env
GEMINI_API_KEY=tu-api-key
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent
BACKEND_URL=http://localhost:8000
LLAMA_API_KEY=opcional
LLAMA_API_URL=opcional
LLAMA_MODEL_NAME=llama-3.1-70b-instruct
```

## 🔄 Flujo de trabajo

1. **Frontend** → Llama a Firebase Function `/challengeThesis`
2. **Firebase Function** → Llama a backend tradicional `/api/analysis/wisdom` para obtener chunks RAG
3. **Firebase Function** → Construye prompt con chunks RAG
4. **Firebase Function** → Llama a Gemini API (o Llama como fallback)
5. **Firebase Function** → Devuelve respuesta completa al frontend

## 📝 Actualizar frontend

Una vez desplegado, actualiza tu frontend para usar las nuevas URLs:

```typescript
// Antes:
const response = await fetch('http://localhost:8000/api/analysis/challenge', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ thesis, ticker, top_k })
});

// Después:
const response = await fetch('https://us-central1-TU-PROYECTO-ID.cloudfunctions.net/challengeThesis', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ thesis, ticker, top_k })
});
```

## 🔒 Seguridad

- Las API keys están almacenadas como variables de entorno seguras en Firebase
- CORS está configurado para permitir solo tus dominios
- El backend tradicional debe estar accesible desde Firebase (considera usar una IP pública o VPN)

## 💰 Costos

Firebase Functions tiene un tier gratuito generoso:
- **2 millones de invocaciones/mes** gratis
- **400,000 GB-segundos** de tiempo de cómputo gratis
- **200,000 GB-segundos** de tiempo de CPU gratis

Después de eso, pagas solo por lo que uses.

## 🐛 Troubleshooting

### Error: "BACKEND_URL no configurada"
Asegúrate de configurar la variable de entorno `backend.url` o `BACKEND_URL`.

### Error: "No se pudo acceder a ningún modelo LLM"
Verifica que `GEMINI_API_KEY` esté configurada correctamente.

### RAG no funciona
- Verifica que tu backend tradicional esté corriendo y accesible
- Verifica que el endpoint `/api/analysis/wisdom` funcione correctamente
- Revisa los logs de Firebase Functions para ver errores específicos

### CORS errors
Asegúrate de agregar tu dominio de frontend a `cors_origins` en `main.py`.

