# 🚀 SETUP COMPLETO - Conectar con Google AI Studio

**Objetivo**: Tener API funcionando en 15 minutos

---

## PASO 1: Entrenar Modelos (5 min)

```bash
cd C:\key\wise_adviser_cursor_context\notebooks

# Instalar dependencias
poetry add xgboost scikit-learn joblib fastapi uvicorn

# Entrenar modelos
poetry run python scripts/train_models_simple.py
```

**Output esperado**:
```
✅ Guardado: models/quality_model.pkl
✅ Guardado: models/valuation_model.pkl
✅ Guardado: models/momentum_model.pkl
```

---

## PASO 2: Iniciar API (1 min)

```bash
# Terminal 1: API FastAPI
cd services/api
poetry run uvicorn main:app --reload --port 8000

# Output:
# INFO: Uvicorn running on http://0.0.0.0:8000
```

**Test**:
```bash
# Terminal 2: Test endpoint
curl http://localhost:8000/

# Output esperado:
# {"service":"Caria API","version":"1.0.0","status":"running"}
```

---

## PASO 3: Configurar Gemini API (2 min)

```bash
# Agregar a .env
echo "GEMINI_API_KEY=tu_api_key_aqui" >> .env
```

**Obtener API key**:
1. Ve a: https://makersuite.google.com/app/apikey
2. Crea proyecto
3. Generate API key
4. Copia a .env

---

## PASO 4: Conectar con Google AI Studio (5 min)

### En Google AI Studio:

**1. Crear Function Calling**:
```json
{
  "name": "analyze_ticker",
  "description": "Analiza un ticker detectando sesgos cognitivos en el proceso de pensamiento del usuario",
  "parameters": {
    "type": "object",
    "properties": {
      "ticker": {
        "type": "string",
        "description": "Símbolo del ticker (ej: AAPL, NVDA, MSFT)"
      },
      "user_query": {
        "type": "string",
        "description": "Query del usuario para detectar sesgos"
      }
    },
    "required": ["ticker"]
  }
}
```

**2. System Instruction (Prompt)**:
```
Eres Caria, un asistente de análisis de inversiones que ayuda a usuarios a tomar decisiones racionales exponiendo sesgos cognitivos.

PRINCIPIOS:
1. NO des scores numéricos (88/100, etc)
2. NO cites inversores gratuitamente ("Buffett dice...")
3. SÍ detecta sesgos (anchoring, FOMO, loss aversion, etc)
4. SÍ usa contexto histórico relevante
5. BALANCE: Responde directo + 2-3 preguntas clave (no abrumar)

SESGOS A DETECTAR:
- Anchoring: Fijar precio en punto irrelevante
- Social Proof: "Todos compran"
- FOMO: Urgencia injustificada
- Loss Aversion: No vender perdedoras
- Recency: Proyectar tendencia reciente al futuro
- Overconfidence: "Sé más que el mercado"
- Confirmation: Buscar solo info que confirme

FORMATO DE RESPUESTA:
1. Análisis directo (moat, valuación, context)
2. 🔍 Sesgo detectado (si aplica)
3. 2-3 preguntas clave
4. Contexto histórico (si relevante)

EJEMPLO:
Usuario: "NVDA subió 300%, todos dicen que seguirá, ¿compro?"

Tu respuesta:
"NVDA tiene moat fuerte (CUDA ecosystem, 80% market share AI chips).

Valuación: P/E 60x vs histórico 35x. Premium por growth esperado de AI, pero asume demanda sostenida.

🔍 Detecto 2 sesgos:
- Recency: "Subió 300%" proyecta pasado al futuro
- Social proof: "Todos dicen" → Decision por rebaño

Preguntas clave:
1. ¿Qué pasa si demanda de GPUs cae 50% (como crypto 2022)?
2. ¿Por qué NVDA vs AMD / custom chips de hyperscalers?
3. ¿Qué % de portfolio? ¿Toleras -50% drawdown?

Contexto: Similar a Cisco 2000 (P/E 100, "internet monopoly"). Funcionó 2 años, luego colapsó. Diferencia: ¿NVDA tiene moat más profundo?"
```

**3. Configurar Webhook**:
- URL: `http://tu-ip:8000/analyze`
- Method: POST
- Headers: `Content-Type: application/json`

---

## PASO 5: Test End-to-End (2 min)

**En Google AI Studio chat**:
```
Usuario: "Quiero comprar AAPL, está barata"

Esperado:
- Llama a analyze_ticker(ticker="AAPL", user_query="está barata")
- Detecta anchoring bias (sin contexto de por qué "barata")
- Retorna análisis + preguntas socráticas
```

---

## 📂 ESTRUCTURA DE ARCHIVOS

```
notebooks/
├── services/api/
│   └── main.py                    ← API FastAPI ✅
├── scripts/
│   └── train_models_simple.py     ← Entrenamiento ✅
├── models/
│   ├── quality_model.pkl          ← Generado tras training
│   ├── valuation_model.pkl
│   └── momentum_model.pkl
├── docs/
│   ├── MUNGER_25_BIASES.md        ← 25 sesgos ✅
│   └── VISION_ACTUALIZADA.md      ← Filosofía ✅
└── .env                           ← GEMINI_API_KEY
```

---

## 🐛 TROUBLESHOOTING

### Error: "ModuleNotFoundError: xgboost"
```bash
poetry add xgboost scikit-learn
```

### Error: "Models not found"
```bash
# Entrenar primero
poetry run python scripts/train_models_simple.py
```

### Error: "Connection refused" en Google AI Studio
```bash
# Tu IP debe ser pública O usar ngrok:
ngrok http 8000

# Usar URL de ngrok en Google AI Studio
# https://xxxx-xx-xxx-xx-xx.ngrok.io/analyze
```

### API no detecta sesgos
- Los sesgos se detectan por keywords en `user_query`
- Ver función `detect_biases_in_query()` en main.py
- Agregar más patterns si necesario

---

## 🚀 DEPLOYMENT (Futuro)

### Opción 1: Railway.app (más fácil)
```bash
# 1. Crear cuenta en railway.app
# 2. Connect GitHub repo
# 3. Deploy automático
# 4. Railway te da URL pública
```

### Opción 2: Google Cloud Run
```bash
# 1. Crear Dockerfile
# 2. Build imagen
# 3. Push a GCR
# 4. Deploy Cloud Run
```

### Opción 3: Render.com (gratis)
```bash
# 1. Connect GitHub
# 2. Select repo
# 3. Deploy
# URL: https://caria-api.onrender.com
```

---

## ✅ CHECKLIST FINAL

- [ ] Modelos entrenados (train_models_simple.py)
- [ ] API corriendo (localhost:8000)
- [ ] GEMINI_API_KEY en .env
- [ ] Google AI Studio configurado (Function + System Prompt)
- [ ] Test: Usuario → Gemini → API → Respuesta con sesgos

**Cuando tengas esto → FUNCIONANDO 🎉**

---

**Siguiente**: Mejorar API con datos reales de tickers (ahora es placeholder)
