# Quick Start - Proyecto Caria

## Iniciar el Sistema (5 minutos)

### 1. Configurar LLM (Elige UNO)

#### Opción A: Llama (GRATIS, LOCAL) ⭐ RECOMENDADO
```bash
# 1. Instalar Ollama
# Windows: https://ollama.ai/download
# Mac: brew install ollama
# Linux: curl https://ollama.ai/install.sh | sh

# 2. Descargar modelo
ollama pull llama3.2

# 3. Instalar cliente
cd notebooks
poetry add ollama

# 4. Verificar
poetry run python -c "import ollama; print('✅ Ollama OK')"
```

#### Opción B: Gemini (API GRATIS)
```bash
# 1. Obtener API Key: https://makersuite.google.com/app/apikey

# 2. Configurar
export GEMINI_API_KEY="tu_api_key_aqui"

# 3. Instalar
cd notebooks
poetry add google-generativeai

# 4. Verificar
poetry run python -c "import os; print('✅ Gemini OK' if os.getenv('GEMINI_API_KEY') else '❌ No API key')"
```

### 2. Levantar API
```bash
cd notebooks/services
poetry run uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 3. Probar Healthcheck
Abrir en navegador: http://localhost:8000/health

Deberías ver:
```json
{
  "status": "ok",
  "rag": "available",
  "regime": "available",
  "factors": "available",
  "valuation": "available"
}
```

## Probar Endpoints

### 1. Régimen Macro Actual
```bash
curl http://localhost:8000/api/regime/current
```

**Respuesta**:
```json
{
  "regime": "expansion",
  "probabilities": {
    "expansion": 0.45,
    "slowdown": 0.25,
    "recession": 0.15,
    "stress": 0.15
  },
  "confidence": 0.45
}
```

### 2. Top Acciones (Screening)
```bash
curl -X POST http://localhost:8000/api/factors/screen \
  -H "Content-Type: application/json" \
  -d '{
    "top_n": 10,
    "regime": "expansion"
  }'
```

### 3. Valuación de Empresa
```bash
curl -X POST http://localhost:8000/api/valuation/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "method": "auto"
  }'
```

### 4. Challenge Tesis (RAG + LLM)
```bash
curl -X POST http://localhost:8000/api/analysis/challenge \
  -H "Content-Type: application/json" \
  -d '{
    "thesis": "Creo que Tesla está infravalorada porque tiene ventaja tecnológica",
    "ticker": "TSLA"
  }'
```

## Conectar con Google Studio

### URLs de la API
```
Base URL: http://tu-servidor:8000

Endpoints:
- GET  /api/regime/current         → MODEL OUTLOOK
- POST /api/factors/screen         → IDEAL PORTFOLIO
- POST /api/valuation/analyze      → VALUACIÓN
- POST /api/analysis/challenge     → CHALLENGE THESIS
```

### Ejemplo de Integración
1. En Google Studio → Agregar Data Source
2. Tipo: "Custom API" o "Google Sheets"
3. Si usas Sheets, crea un Apps Script:

```javascript
function updateData() {
  // Obtener régimen
  var response = UrlFetchApp.fetch("http://tu-servidor:8000/api/regime/current");
  var data = JSON.parse(response.getContentText());

  // Actualizar sheet
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Data");
  sheet.getRange("A2").setValue(data.regime);
  sheet.getRange("B2").setValue(data.probabilities.expansion);
}

// Ejecutar cada hora con Triggers
```

## Troubleshooting

### Error: "No LLM provider available"
- **Solución**: Instala Ollama O configura GEMINI_API_KEY

### Error: "Regime service unavailable"
- **Causa**: Modelo HMM no encontrado
- **Solución**:
```bash
cd notebooks/caria_data
poetry run python train_hmm_simple.py
```

### Error: "RAG disabled"
- **Causa**: PostgreSQL/pgvector no configurado
- **Impacto**: Solo afecta "Challenge Thesis", resto funciona
- **Solución (opcional)**:
```bash
docker run -d --name caria-postgres \
  -e POSTGRES_PASSWORD=Theolucas7 \
  -e POSTGRES_USER=caria_user \
  -e POSTGRES_DB=caria \
  -p 5432:5432 \
  ankane/pgvector
```

## Verificación Final

```bash
# Test completo
cd notebooks/caria_data
poetry run python -c "
from caria.services.llm_service import LLMService
from caria.models.regime.hmm_regime_detector import HMMRegimeDetector

# 1. LLM
llm = LLMService.auto_detect()
print(f'✅ LLM: {llm.provider}')

# 2. HMM
detector = HMMRegimeDetector.load('models/regime_hmm_model.pkl')
print(f'✅ HMM: {len(detector.feature_names)} features')

# 3. Valuación
from caria.models.valuation.multiples_valuator import MultiplesValuator
valuator = MultiplesValuator()
print('✅ Valuación: Múltiplos disponibles')

print('\n🎉 Todo listo!')
"
```

## Siguientes Pasos

1. ✅ Sistema funcionando
2. 🔄 Conectar con Google Studio
3. 📊 Crear dashboards visuales
4. 🚀 Deploy a producción (opcional)

## Links Útiles

- **Ollama**: https://ollama.ai/
- **Gemini API**: https://makersuite.google.com/
- **FastAPI Docs**: http://localhost:8000/docs
- **Repo**: C:\key\wise_adviser_cursor_context\notebooks\

---

**¿Problemas?** Revisa `IMPLEMENTACION_COMPLETA.md` para documentación detallada.
