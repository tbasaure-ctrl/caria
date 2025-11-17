# Caria - Nueva Arquitectura: Análisis de Valuación + Detección de Sesgos + RAG

**Fecha**: 2025-11-09
**Objetivo**: Ayudar a usuarios a tomar decisiones racionales exponiendo sus sesgos cognitivos y proveyendo contexto histórico

---

## 🎯 VISIÓN ACTUALIZADA

### De Predicción → A Evaluación Contextual

**ANTES (Incorrecto):**
```
Input: Fundamentals trimestrales
Output: Predicción de return a 1 año
Problema: Imposible predecir con fundamentals lagging
```

**AHORA (Correcto):**
```
Input: Multi-modal features (fundamentals + technical + macro + sentiment + historical patterns)
Output: Framework de análisis cualitativo
  1. Moats Analysis: ¿Qué ventajas competitivas duraderas tiene?
  2. Valuation Context: ¿Cómo se compara históricamente?
  3. Macro Context: ¿Qué régimen estamos viviendo?
  4. Bias Detection: ¿Qué sesgos cognitivos estás mostrando?
  5. Historical Parallels: ¿Qué situaciones similares existieron?

Goal: Proceso de pensamiento estructurado que revele tus propios sesgos
NO: "Score 88/100" SÍ: "Estás mostrando anchoring bias al fijarte en el precio de entrada anterior"
```

---

## 📊 ARQUITECTURA DE SCORING

### 1. Quality Score (0-100)
**¿Qué mide?** Ventajas competitivas duraderas (moats)

**Features clave:**
- **Rentabilidad persistente**: ROIC > 15% por 10+ años
- **Poder de pricing**: Gross margin expansion
- **Capital efficiency**: FCF/Revenue ratio
- **Reinvestment**: R&D intensity + CapEx patterns
- **Management quality**: Insider ownership, capital allocation history
- **Network effects**: Customer retention, switching costs
- **Brand strength**: Pricing premium vs competitors

**Outliers históricos** (Quality > 90):
- Coca-Cola (1920s-1990s): Brand moat
- Berkshire Hathaway (1965-2024): Capital allocation
- Apple (2000-2024): Ecosystem lock-in
- Visa/Mastercard (2000-2024): Network effects

**Modelo:**
```python
# Gradient Boosting (XGBoost) - mejor para tabular features
# Input: 50+ fundamental features (10 years history)
# Output: Quality percentile (0-100)
# Loss: Ranking loss (pairwise comparisons)
```

### 2. Valuation Score (0-100)
**¿Qué mide?** Attractive vs expensive relative to intrinsic value

**Features clave:**
- **Multiples**: P/E, P/B, EV/EBITDA, P/FCF
- **Relative valuation**: vs sector, vs history, vs growth
- **DCF-based**: Implied growth rate, margin of safety
- **Shiller PE**: Cyclically-adjusted earnings
- **Earnings quality**: Accruals, cash conversion

**Framework:**
```
Score = 100 - (Current_Valuation_Percentile * 100)

Example:
- P/E = 10 (10th percentile historically) → Score = 90 (muy barata)
- P/E = 40 (90th percentile historically) → Score = 10 (muy cara)
```

### 3. Momentum Score (0-100)
**¿Qué mide?** Technical setup + price action

**Features clave:**
- **Trend**: SMA 50/200, ADX
- **Momentum**: RSI, MACD, ROC
- **Volume**: OBV, Chaikin Money Flow
- **Volatility**: Bollinger Bands, ATR
- **Relative strength**: vs S&P 500, vs sector

**Uso:**
- High momentum (>70): Confirma quality pick
- Low momentum (<30): Wait for better entry

### 4. Context Score (0-100)
**¿Qué mide?** Macro timing + regime awareness

**Features clave:**
- **Interest rates**: Fed Funds, 10Y Treasury, yield curve
- **Inflation**: CPI, PCE, breakevens
- **Growth**: GDP, Employment, PMIs
- **Credit**: IG/HY spreads, default rates
- **Market regime**: Bull/Bear/Crisis classification
- **Sector rotation**: Which sectors outperform in current regime

**Examples:**
- Tech stocks: Score high when rates falling + growth accelerating
- Value stocks: Score high when rates rising + inflation high
- Defensives: Score high when recession probability > 30%

---

## 🧠 MODELO ENSEMBLE

### Architecture
```
┌─────────────────────────────────────────────────────┐
│                  INPUT FEATURES                       │
├──────────────┬──────────────┬──────────┬─────────────┤
│ Fundamentals │  Technicals  │   Macro  │  Sentiment  │
│  (40 feats)  │  (25 feats)  │ (15 fts) │  (10 feats) │
└──────┬───────┴──────┬───────┴────┬─────┴──────┬──────┘
       │              │            │            │
       v              v            v            v
┌──────────────────────────────────────────────────────┐
│               SPECIALIZED SCORERS                     │
├───────────────┬──────────────┬──────────┬────────────┤
│   XGBoost     │   XGBoost    │ XGBoost  │  XGBoost   │
│   Quality     │  Valuation   │ Momentum │  Context   │
│  (50 trees)   │  (50 trees)  │(30 trees)│ (30 trees) │
└───────┬───────┴──────┬───────┴────┬─────┴──────┬─────┘
        │              │            │            │
        v              v            v            v
┌──────────────────────────────────────────────────────┐
│                  SCORE OUTPUTS                        │
├───────────────┬──────────────┬──────────┬────────────┤
│  Quality      │  Valuation   │ Momentum │  Context   │
│   (0-100)     │   (0-100)    │  (0-100) │   (0-100)  │
└───────┬───────┴──────┬───────┴────┬─────┴──────┬─────┘
        │              │            │            │
        └──────────────┴────────────┴────────────┘
                       │
                       v
            ┌─────────────────────┐
            │  META-LEARNER       │
            │  (Weighted Average) │
            └──────────┬──────────┘
                       │
                       v
            ┌─────────────────────┐
            │ OVERALL SCORE       │
            │   (0-100)           │
            └─────────────────────┘
```

### Weighted Combination
```python
overall_score = (
    0.40 * quality_score +      # Most important: moats
    0.30 * valuation_score +    # Price matters
    0.20 * context_score +      # Timing matters
    0.10 * momentum_score       # Confirmation
)
```

---

## 🔍 OUTLIER DETECTION

### Definition
**Outlier** = Empresa con características excepcionales que crean alpha a largo plazo

### Criteria
```python
is_outlier = (
    quality_score > 85 AND
    (
        (valuation_score > 60 AND context_score > 60) OR  # Good price + timing
        (quality_score > 95)  # So exceptional that price doesn't matter
    )
)
```

### Historical Outliers Dataset
Crear dataset de empresas excepcionales desde 1900:
- Ford (1900s): Manufacturing revolution
- Coca-Cola (1920s): Brand building
- IBM (1950s): Computing monopoly
- Walmart (1970s): Retail disruption
- Microsoft (1990s): Software moat
- Apple (2000s): Ecosystem lock-in
- Amazon (2000s): Platform economics
- Google (2000s): Search monopoly
- Berkshire (1965-2024): Capital allocation

**Features to extract:**
- Founding story + key innovations
- Moat development timeline
- Management characteristics
- Capital allocation patterns
- Crisis responses
- Reinvestment strategies

---

## 💬 RAG SYSTEM INTEGRATION

### Query Examples

**Query 1**: "Why is AAPL scored 88?"
```
Retrieved context:
- Warren Buffett on moats (1995): "The key to investing is...competitive advantage"
- Peter Lynch on great businesses: "Own what you know...ecosystem stickiness"
- Historical: Apple iPhone launch (2007) created switching costs

Generated response:
"Apple scores 88 primarily due to its exceptional Quality Score (94).

KEY MOATS:
1. Ecosystem Lock-in: 2B+ devices create switching costs ~$2000/user
2. Brand Premium: Gross margins 44% vs Samsung 35%
3. Services Revenue: Recurring $85B/year (35% margins)

HISTORICAL PRECEDENT:
Similar to Coca-Cola in 1950s - built brand moat that allowed pricing power
for decades. Buffett: 'Give me a business so good that an idiot can run it,
because eventually one will.'

VALUATION CONTEXT:
Current P/E 28x vs 10-year avg 18x → Valuation Score only 45.
Not a screaming buy, but quality justifies premium.

MACRO TIMING:
Fed cutting rates + Services growth accelerating → Context Score 72.
Better entry may come if recession materializes."
```

**Query 2**: "Compare NVDA vs AMD"
```
Retrieved context:
- Historical: Intel monopoly (1990s) vs AMD underdog
- Clayton Christensen: Disruptive innovation theory
- Current AI revolution analogies to PC revolution (1980s)

Generated response:
"OUTLIER COMPARISON:

NVIDIA (Overall: 91)
- Quality: 96 (AI chip monopoly, 80% market share)
- Valuation: 35 (P/E 65x, extended)
- Momentum: 88 (strong uptrend)
- Historical analog: Intel in 1990s (dominated CPUs)

AMD (Overall: 68)
- Quality: 72 (good products, but follower)
- Valuation: 65 (P/E 35x, reasonable)
- Momentum: 55 (consolidating)
- Historical analog: AMD in 2000s (catching up)

WISDOM:
Buffett: 'It's better to buy a wonderful company at a fair price than a fair
company at a wonderful price.'

NVDA has the moat (CUDA ecosystem = switching costs). AMD is cheaper but
lacks the ecosystem lock-in. Similar to Coca-Cola vs Pepsi in 1960s -
both good businesses, but one had the deeper moat.

DECISION:
- For quality-focused investor: NVDA despite high valuation
- For value-focused: Wait for NVDA correction or nibble AMD
- Historical: Intel monopoly lasted 15 years before ARM disruption"
```

---

## 📁 IMPLEMENTATION PLAN

### Phase 1: Core Scoring Models (Week 1-2)
1. ✅ Create quality_scorer.py (XGBoost for quality)
2. ✅ Create valuation_scorer.py (XGBoost for valuation)
3. ✅ Create momentum_scorer.py (XGBoost for momentum)
4. ✅ Create context_scorer.py (XGBoost for macro context)
5. ✅ Create ensemble.py (Meta-learner)

### Phase 2: Outlier Detection (Week 3)
1. Build historical outliers dataset (manual curation)
2. Extract common patterns (factor analysis)
3. Train outlier classifier (binary: outlier vs normal)
4. Validate against known cases (AAPL, MSFT, BRK, GOOGL)

### Phase 3: RAG Integration (Week 4)
1. ✅ Setup pgvector (done)
2. Ingest wisdom corpus (29MB, 35 books)
3. Build query engine (retrieve + rank + generate)
4. Test Q&A: "Why score X?", "Compare A vs B"

### Phase 4: API & UI (Week 5-6)
1. FastAPI endpoints:
   - POST /score/{ticker} → Returns all 5 scores
   - POST /compare → Compare 2+ tickers
   - POST /outliers → Find outliers in universe
   - POST /explain → RAG-powered explanation
2. Simple UI dashboard (Streamlit):
   - Ticker input → Score breakdown
   - Historical chart with regime coloring
   - Outlier list (sortable)
   - Chat interface for RAG

---

## 📈 SUCCESS METRICS

### Model Performance
- **Quality Score**: AUC > 0.80 in identifying top decile performers
- **Valuation Score**: Correlation with forward 3Y returns > 0.3
- **Outlier Detection**: Precision > 70% on historical cases

### User Value
- **Decisioning**: Users can articulate WHY they're buying (not just FOMO)
- **Emotional control**: RAG provides historical context to reduce panic
- **Learning**: Users understand business quality frameworks

---

## 🔗 FILES TO CREATE

```
src/caria/scoring/
├── quality_scorer.py       ← XGBoost for quality (ROIC, moats, etc)
├── valuation_scorer.py     ← XGBoost for valuation (P/E, DCF, etc)
├── momentum_scorer.py      ← XGBoost for technicals
├── context_scorer.py       ← XGBoost for macro timing
├── ensemble.py             ← Meta-learner combining all
└── outlier_detector.py     ← Binary classifier for outliers

src/caria/rag/
├── query_engine.py         ← Retrieve + Generate
├── context_retriever.py    ← pgvector search
└── prompt_templates.py     ← Templates for explanations

data/manual/
└── historical_outliers.csv ← Curated dataset (Ford, Coca-Cola, etc)

configs/pipelines/
└── scoring_pipeline.yaml   ← Run all scorers + ensemble
```

---

## 💡 KEY INSIGHTS

1. **Quality > Price**: Buffett was right - wonderful companies compound
2. **Moats are everything**: Without moats, competition erodes returns
3. **Context matters**: Even AAPL was a bad buy in 2000 (too early)
4. **Psychology > Math**: RAG helps control emotions with historical wisdom
5. **Outliers are rare**: ~5% of companies create 90% of wealth

---

**Next Action**: Implement quality_scorer.py with XGBoost
