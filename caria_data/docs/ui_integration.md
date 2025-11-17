# Integración de Sistemas con UI

Este documento describe cómo conectar los 4 sistemas especializados de Caria con la UI existente.

## Mapeo de Endpoints a UI

### 1. MODEL OUTLOOK (Gauge de Sentimiento)
**Endpoint**: `GET /api/regime/current`

**Respuesta**:
```json
{
  "regime": "expansion",
  "probabilities": {
    "expansion": 0.65,
    "slowdown": 0.20,
    "recession": 0.10,
    "stress": 0.05
  },
  "confidence": 0.65
}
```

**Mapeo UI**:
- `regime` → Determina color del gauge (expansion=verde, recession=rojo, etc.)
- `probabilities.expansion` → Porcentaje mostrado
- `confidence` → Opacidad o intensidad del indicador

### 2. IDEAL CARIA PORTFOLIO
**Endpoints combinados**:
- `POST /api/factors/screen` (top N empresas)
- `GET /api/valuation/{ticker}` (para cada empresa)

**Flujo**:
1. Llamar `/api/factors/screen` con `top_n=20`
2. Para cada empresa en resultados, llamar `/api/valuation/{ticker}`
3. Combinar scores de factores con valuación
4. Mostrar lista rankeada con explicaciones

### 3. TOP MOVERS
**Endpoint**: `POST /api/factors/screen`

**Parámetros**:
- `top_n=10`
- `regime`: Opcional, se detecta automáticamente

**Respuesta**: Lista de empresas rankeadas por composite score

### 4. Challenge Your Thesis
**Endpoint**: `POST /api/analysis/challenge`

**Request**:
```json
{
  "thesis": "Buy NVDA because AI is the future",
  "ticker": "NVDA",
  "top_k": 5
}
```

**Respuesta**:
```json
{
  "thesis": "Buy NVDA because AI is the future",
  "retrieved_chunks": [...],
  "critical_analysis": "Análisis crítico...",
  "identified_biases": ["confirmation bias", "overconfidence"],
  "recommendations": ["Revisar múltiples perspectivas..."],
  "confidence_score": 0.75
}
```

## Ejemplos de Integración

### Frontend React/TypeScript

```typescript
// Obtener régimen actual para MODEL OUTLOOK
async function getCurrentRegime() {
  const response = await fetch('/api/regime/current');
  const data = await response.json();
  return {
    regime: data.regime,
    probability: data.probabilities[data.regime],
    confidence: data.confidence
  };
}

// Obtener portfolio ideal
async function getIdealPortfolio() {
  // 1. Screenear empresas
  const screenResponse = await fetch('/api/factors/screen', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ top_n: 20 })
  });
  const companies = await screenResponse.json().companies;
  
  // 2. Valuar cada una
  const portfolio = await Promise.all(
    companies.map(async (company) => {
      const valuation = await fetch(`/api/valuation/${company.ticker}`);
      return {
        ...company,
        valuation: await valuation.json()
      };
    })
  );
  
  return portfolio;
}

// Challenge thesis
async function challengeThesis(thesis: string, ticker?: string) {
  const response = await fetch('/api/analysis/challenge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thesis, ticker, top_k: 5 })
  });
  return await response.json();
}
```

## Estados de Servicios

Todos los endpoints verifican disponibilidad de servicios. Si un servicio no está disponible, retorna:
- `503 Service Unavailable` con mensaje descriptivo
- El healthcheck (`GET /health`) muestra estado de cada servicio

## Notas de Implementación

1. **Régimen**: El Sistema I (HMM) debe estar entrenado antes de usar endpoints que dependen de régimen
2. **Factores**: Requiere datos de fundamentals y técnicos en `silver/`
3. **Valuación**: DCF requiere FCF y shares outstanding; Scorecard requiere datos cualitativos (actualmente placeholders)
4. **RAG**: Requiere embeddings en pgvector y opcionalmente LLM local para análisis crítico

