# Plan de Implementación Revisado: Thesis Arena, Regime Testing, y Community Feed

## Cambios Principales Basados en Feedback

### 1. Thesis Arena → Comunidades en lugar de Agentes Individuales
- **Value Community**: Icono que evoque a Warren Buffett (sin imagen real)
- **Finance-Crypto Bro Community**: Millennials con retornos extraordinarios pero novatos
- **Growth Community**: Combina filosofías de Lynch, Druckenmiller, Miller, etc.
- **Contrarian Community**: Siempre va contra el pensamiento de la mayoría
- Cada comunidad tiene icono con tooltip hover (sin click) que muestra descripción

### 2. Regime Map → Simplificado a "Probar Escenarios"
- Eliminar drag & drop complejo
- Solo opción "Test según régimen" con holdings actuales
- Pasa por Monte Carlo simulation
- Visualización de protección: exposición y drawdown proyectado a 12 meses

### 3. Portfolio Ideal → Sistema de Validación y Reentrenamiento
- Exponer modelo a elegir outliers o portfolios con 10-20 holdings
- Tracking de performance en tiempo real vs benchmarks
- Sistema de reentrenamiento basado en performance

---

## Phase 1: Regime Testing Tool (Simplificado - Week 1)

### Backend Changes

**1.1 Regime Testing Endpoint** (`backend/api/routes/regime_testing.py`)
- POST `/api/portfolio/regime-test`
- Input: `{regime: "expansion"|"recession"|"slowdown"|"stress", holdings: [{ticker, allocation}]}`
- Process:
  1. Obtener holdings actuales del usuario (o usar los proporcionados)
  2. Clasificar cada holding por régimen suitability
  3. Calcular exposición actual vs régimen objetivo
  4. Ejecutar Monte Carlo simulation a 12 meses
  5. Calcular drawdown estimado, percentiles
- Return: `{exposure_score: 0-100, drawdown_estimate: {...}, monte_carlo_results: {...}, protection_level: "high"|"medium"|"low"}`

**1.2 Monte Carlo Integration** (`backend/api/routes/monte_carlo.py`)
- Extender endpoint existente para aceptar régimen específico
- POST `/api/portfolio/monte-carlo/regime-test`
- Usa holdings reales del usuario + régimen objetivo
- Retorna resultados específicos para visualización de protección

**1.3 Asset Regime Classification** (`backend/api/services/asset_regime_service.py`)
- Clasificar holdings por régimen suitability
- Usar datos históricos: cómo se comportó cada activo en cada régimen
- Return: `{ticker, regime_suitability: {expansion: score, recession: score, ...}}`

### Frontend Changes

**1.4 RegimeTestWidget Component** (`frontend/caria-app/components/widgets/RegimeTestWidget.tsx`)
- Botón principal: "Test según Régimen"
- Dropdown para seleccionar régimen: Expansion, Recesión, Slowdown, Stress
- Muestra holdings actuales del usuario (desde Portfolio widget)
- Al hacer click:
  1. Llama a `/api/portfolio/regime-test`
  2. Muestra loading state
  3. Renderiza resultados:
     - **Gráfico de protección**: Gauge o barra mostrando nivel de protección (0-100%)
     - **Drawdown estimado**: "Drawdown máximo esperado: -X%"
     - **Exposición**: "Tu portfolio está X% expuesto a este régimen"
     - **Monte Carlo visualization**: Gráfico de distribución de retornos a 12 meses
     - **Recomendaciones**: "Para mejorar protección, considera agregar Y% de [activo]"

**1.5 ProtectionVisualization Component** (`frontend/caria-app/components/widgets/ProtectionVisualization.tsx`)
- Gauge circular o barra horizontal mostrando nivel de protección
- Colores: Verde (alta protección), Amarillo (media), Rojo (baja)
- Tooltip con detalles: exposición actual, drawdown esperado

**1.6 RegimeTestResults Component** (`frontend/caria-app/components/widgets/RegimeTestResults.tsx`)
- Muestra resultados del test
- Integra con MonteCarloSimulation existente para visualización
- Botones: "Ver detalles", "Comparar con otros regímenes"

### Integration
- Conectar con `Portfolio` widget para obtener holdings actuales
- Reutilizar `MonteCarloSimulation` para visualización
- Usar `ModelOutlook` para régimen actual como default

---

## Phase 2: Thesis Arena con Comunidades (Week 2)

### Backend Changes

**2.1 Community Prompts System** (`backend/api/routes/thesis_arena.py`)
- Crear 4 comunidades con prompts específicos:
  
  **Value Community**:
  - Prompt: Enfoque en margin of safety, P/E ratios, moats, valor intrínseco
  - Filosofía: "Solo compra cuando hay margen de seguridad significativo"
  - Icono: 💎 (diamante evoca valor/calidad)
  
  **Finance-Crypto Bro Community**:
  - Prompt: Enfoque en retornos extraordinarios, momentum, FOMO, riesgo alto
  - Filosofía: "YOLO, pero con datos. Retornos 100%+ son posibles si..."
  - Icono: 🚀 (cohete evoca crecimiento explosivo)
  
  **Growth Community**:
  - Prompt: Combina filosofías de Lynch (buy what you know), Druckenmiller (macro + growth), Miller (contrarian growth)
  - Filosofía: "Crecimiento sostenible + ventaja competitiva + timing macro"
  - Icono: 📈 (gráfico ascendente)
  
  **Contrarian Community**:
  - Prompt: Siempre desafía consenso, busca oportunidades donde otros ven riesgo
  - Filosofía: "Cuando todos compran, vende. Cuando todos venden, compra (pero con fundamentos)"
  - Icono: 🔄 (flechas opuestas evoca contrarian)
  
- Store prompts en `backend/prompts/communities/` directory

**2.2 Thesis Challenge Endpoint** (`backend/api/routes/thesis_arena.py`)
- POST `/api/thesis/arena/challenge`
- Input: `{thesis_text, ticker?, conviction: 0-10, community_preferences?: ["value", "growth", "crypto", "contrarian"]}`
- Process: 
  - Si `community_preferences` no se especifica, desafía contra todas las comunidades
  - Envía tesis a cada comunidad seleccionada (LLM calls en paralelo)
  - Cada comunidad responde con su perspectiva
- Return: `{communities: [{name, icon, response, conviction_impact: -X to +X, description}], new_conviction: X}`

**2.3 Conviction Calculator** (`backend/api/services/conviction_service.py`)
- Similar a plan original
- Peso por comunidad puede variar según tipo de tesis

**2.4 Arena Thread System** (`backend/api/routes/thesis_arena.py`)
- POST `/api/thesis/arena/respond` - Usuario responde a desafío de comunidad
- Threaded conversations por comunidad
- Store en `thesis_arena_threads` table

### Frontend Changes

**2.5 ThesisArena Component** (`frontend/caria-app/components/widgets/ThesisArena.tsx`)
- Top: Editable thesis textarea
- Conviction meter: Slider 0-10 con barra visual
- **Grid 2x2 de comunidades**:
  - Cada card muestra:
    - **Icono** (emoji o SVG custom): 💎, 🚀, 📈, 🔄
    - **Nombre**: "Value Community", "Finance-Crypto Bro", etc.
    - **Tooltip hover** (sin click): Descripción de la comunidad
    - **Response text**: Respuesta generada (5-10 líneas)
    - **Conviction impact**: "+2.3" o "-1.5" con color
  - Loading state: Skeleton mientras LLM genera
- Footer buttons:
  - Red: "Responder y contraatacar" (abre thread modal)
  - Green: "Aceptar y refinar tesis" (actualiza conviction)
  - Gray: "Publicar en Feed" (link a Community Feed)

**2.6 CommunityCard Component** (`frontend/caria-app/components/widgets/CommunityCard.tsx`)
- Card reutilizable para cada comunidad
- Props: `{icon, name, description, response, convictionImpact, onHover}`
- Tooltip aparece en hover (no click)
- Descripción muestra características del oponente

**2.7 CommunityTooltip Component** (`frontend/caria-app/components/widgets/CommunityTooltip.tsx`)
- Tooltip que aparece en hover sobre icono
- Muestra descripción de la comunidad
- Ejemplo: "Value Community: Enfocada en margin of safety, P/E ratios razonables, y moats duraderos. Te desafiará si tu tesis ignora valoración."

**2.8 ArenaThreadModal Component** (`frontend/caria-app/components/modals/ArenaThreadModal.tsx`)
- Modal para conversaciones multi-round
- Muestra historial por comunidad
- Usuario puede responder a comunidad específica

### Iconos y Descripciones

**Value Community** 💎
- Descripción: "Enfocada en margin of safety, valoraciones razonables (P/E, P/B), y moats competitivos duraderos. Te desafiará si tu tesis ignora el precio que pagas."
- Características: Conservadora, fundamentos sólidos, largo plazo

**Finance-Crypto Bro Community** 🚀
- Descripción: "Millennials con experiencia en retornos extraordinarios pero novatos en inversión tradicional. Enfocada en momentum, tendencias, y oportunidades de alto riesgo/alto retorno."
- Características: Optimista, momentum-driven, tolerancia alta al riesgo

**Growth Community** 📈
- Descripción: "Combina filosofías de Lynch ('buy what you know'), Druckenmiller (macro + growth), y Miller (contrarian growth). Busca crecimiento sostenible con ventaja competitiva y timing macro."
- Características: Balanceada, fundamentos + macro, crecimiento sostenible

**Contrarian Community** 🔄
- Descripción: "Siempre va contra el consenso. Busca oportunidades donde otros ven riesgo, pero con fundamentos sólidos. Te desafiará si tu tesis sigue la mayoría."
- Características: Contrarian, busca oportunidades no obvias, fundamentos sólidos

### Integration
- Reemplazar `StartAnalysisCTA` onClick para abrir ThesisArena
- Conectar con Gemini service existente
- Usar auth system para tracking de usuario

---

## Phase 3: Enhanced Community Feed (Week 3)

### Backend Changes

**3.1 Enhanced Posts Schema** (`backend/api/routes/community.py`)
- Agregar campos a `community_posts`:
  - `arena_conviction_pre: float`
  - `arena_conviction_post: float`
  - `arena_communities_challenged: string[]` (qué comunidades desafió)
  - `arena_survived: boolean`
  - `arena_wins: int`
  - `arena_losses: int`
- Migration script

**3.2 Feed Rankings** (`backend/api/routes/community.py`)
- Endpoint `/api/community/rankings`
- Returns: Top comunidades más desafiantes, Hot theses, Survivors

**3.3 Post Filtering** (`backend/api/routes/community.py`)
- Usar LLM para filtrar low-quality posts
- Endpoint `/api/community/posts/validate`

**3.4 Arena Integration** (`backend/api/routes/community.py`)
- POST `/api/community/posts` - Al publicar desde Arena, incluir metadata
- Link posts a arena threads
- Mostrar badge "Survived Arena x3" si aplica

### Frontend Changes

**3.5 Enhanced CommunityFeed Component** (`frontend/caria-app/components/widgets/CommunityFeed.tsx`)
- Reemplazar/enhance `CommunityIdeas` existente
- Search bar: "Buscar por ticker o tema"
- Post cards muestran:
  - Badge de Arena si sobrevivió desafíos
  - Pre/post conviction scores
  - "Entrar Arena" button (link a ThesisArena con tesis pre-filled)
  - "Rematch" button para posts que perdieron
- Sidebar: Rankings widget
- Footer: Botón grande "Publicar Mi Tesis"

**3.6 RankingsWidget Component** (`frontend/caria-app/components/widgets/RankingsWidget.tsx`)
- Muestra top comunidades más desafiantes
- Hot theses, survivors
- Updates en tiempo real

**3.7 ThesisEditor Component** (`frontend/caria-app/components/modals/ThesisEditorModal.tsx`)
- Rich text editor para creación de tesis
- Pre-fill desde Arena si viene de ahí
- Validación: Min 300 palabras, ticker opcional
- "Publish" button → crea post + opción de abrir Arena

### Integration
- Enhance `CommunityIdeas` widget existente
- Conectar con ThesisArena para flow "Entrar Arena"
- Usar sistema de upvote/comment existente

---

## Phase 4: Portfolio Ideal - Sistema de Validación y Reentrenamiento (Week 4)

### Problema Identificado
- Portfolio ideal actual puede no estar en sintonía con el modelo
- Necesitamos exponer el modelo a elegir portfolios reales y validar performance

### Backend Changes

**4.1 Portfolio Selection Service** (`backend/api/services/portfolio_selection_service.py`)
- Nuevo servicio que selecciona portfolios usando el modelo
- Criterios:
  - **Outliers**: Seleccionar holdings que son outliers según factores del modelo
  - **10-20 Holdings**: Limitar tamaño de portfolio
  - **Diversificación**: Asegurar diversificación por sector/geografía
  - **Regime Alignment**: Alinear con régimen actual detectado
- Método: `select_portfolio(strategy: "outliers"|"balanced"|"regime_aligned", num_holdings: int) -> List[Holding]`

**4.2 Portfolio Tracking** (`backend/api/routes/portfolio_tracking.py`)
- POST `/api/portfolio/model/select` - Modelo selecciona portfolio
- GET `/api/portfolio/model/track/{portfolio_id}` - Tracking de performance
- Compara performance vs benchmarks (SPY, QQQ, etc.)
- Calcula métricas: Sharpe ratio, max drawdown, alpha, beta

**4.3 Performance Database** (`backend/api/models/portfolio_performance.py`)
- Tabla `model_portfolios`:
  - `id`, `user_id`, `selected_at`, `holdings: JSON`, `strategy`, `regime_at_selection`
- Tabla `portfolio_performance`:
  - `portfolio_id`, `date`, `return_pct`, `benchmark_return_pct`, `metrics: JSON`

**4.4 Reentrenamiento Trigger** (`backend/api/services/model_retraining.py`)
- Analiza performance de portfolios seleccionados por modelo
- Si performance < benchmark por X períodos → trigger reentrenamiento
- Endpoint: POST `/api/model/retrain` (admin only)
- Usa datos de performance para ajustar pesos del modelo

### Frontend Changes

**4.5 ModelPortfolioWidget Component** (`frontend/caria-app/components/widgets/ModelPortfolioWidget.tsx`)
- Reemplaza o enhance `IdealPortfolio`
- Muestra portfolio seleccionado por modelo
- Opciones:
  - "Ver Portfolio Seleccionado" (10-20 holdings)
  - "Estrategia: Outliers" o "Estrategia: Balanceada"
  - "Performance Tracking" (si portfolio tiene tracking)
- Botón: "Aceptar este Portfolio" → guarda como portfolio del usuario

**4.6 PortfolioPerformance Component** (`frontend/caria-app/components/widgets/PortfolioPerformance.tsx`)
- Muestra performance de portfolios seleccionados por modelo
- Gráfico: Performance vs Benchmark (SPY, QQQ)
- Métricas: Sharpe, Alpha, Beta, Max Drawdown
- Timeline: "Portfolio seleccionado hace X días, performance Y% vs benchmark Z%"

**4.7 ModelValidationDashboard Component** (`frontend/caria-app/components/widgets/ModelValidationDashboard.tsx`)
- Dashboard para validar modelo (admin o usuarios avanzados)
- Muestra estadísticas agregadas:
  - "Modelo ha seleccionado X portfolios"
  - "Performance promedio: Y% vs benchmark Z%"
  - "Portfolios que superaron benchmark: W%"
- Botón: "Trigger Reentrenamiento" (si performance < threshold)

### Integration
- Conectar con `RegimeService` para régimen actual
- Usar `FactorService` para identificar outliers
- Integrar con sistema de holdings existente

---

## Database Schema Updates

```sql
-- Thesis Arena Threads (sin cambios)
CREATE TABLE thesis_arena_threads (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    thesis_text TEXT,
    initial_conviction FLOAT,
    current_conviction FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Arena Rounds (actualizado para comunidades)
CREATE TABLE arena_rounds (
    id UUID PRIMARY KEY,
    thread_id UUID REFERENCES thesis_arena_threads(id),
    community_name VARCHAR(50), -- 'value', 'crypto_bro', 'growth', 'contrarian'
    community_response TEXT,
    user_response TEXT,
    conviction_impact FLOAT,
    round_number INT,
    created_at TIMESTAMP
);

-- Enhanced community_posts (actualizado)
ALTER TABLE community_posts ADD COLUMN arena_conviction_pre FLOAT;
ALTER TABLE community_posts ADD COLUMN arena_conviction_post FLOAT;
ALTER TABLE community_posts ADD COLUMN arena_communities_challenged TEXT[]; -- Array de comunidades
ALTER TABLE community_posts ADD COLUMN arena_survived BOOLEAN DEFAULT FALSE;
ALTER TABLE community_posts ADD COLUMN arena_wins INT DEFAULT 0;
ALTER TABLE community_posts ADD COLUMN arena_losses INT DEFAULT 0;
ALTER TABLE community_posts ADD COLUMN arena_thread_id UUID REFERENCES thesis_arena_threads(id);

-- Model Portfolio Tracking (nuevo)
CREATE TABLE model_portfolios (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    selected_at TIMESTAMP,
    holdings JSONB, -- [{ticker, allocation, sector, ...}]
    strategy VARCHAR(50), -- 'outliers', 'balanced', 'regime_aligned'
    regime_at_selection VARCHAR(50),
    num_holdings INT,
    created_at TIMESTAMP
);

-- Portfolio Performance Tracking (nuevo)
CREATE TABLE portfolio_performance (
    id UUID PRIMARY KEY,
    portfolio_id UUID REFERENCES model_portfolios(id),
    date DATE,
    return_pct FLOAT,
    benchmark_return_pct FLOAT, -- SPY o QQQ
    metrics JSONB, -- {sharpe, alpha, beta, max_drawdown, ...}
    created_at TIMESTAMP
);

-- Index para queries de performance
CREATE INDEX idx_portfolio_performance_portfolio_date ON portfolio_performance(portfolio_id, date DESC);
```

---

## Implementation Order (Revisado)

1. **Week 1: Regime Testing Tool**
   - Backend: Regime testing endpoint, Monte Carlo integration
   - Frontend: RegimeTestWidget, ProtectionVisualization
   - Integration: Conectar con Portfolio y MonteCarloSimulation

2. **Week 2: Thesis Arena con Comunidades**
   - Backend: Community prompts, challenge endpoint, conviction calculator
   - Frontend: ThesisArena con comunidades, CommunityCard con tooltips
   - Integration: Reemplazar StartAnalysisCTA

3. **Week 3: Enhanced Feed**
   - Backend: Enhanced posts schema, rankings, filtering
   - Frontend: Enhanced CommunityFeed, RankingsWidget, ThesisEditor
   - Integration: Conectar Arena → Feed flow

4. **Week 4: Portfolio Ideal - Validación**
   - Backend: Portfolio selection service, tracking, performance DB
   - Frontend: ModelPortfolioWidget, PortfolioPerformance, ValidationDashboard
   - Integration: Conectar con RegimeService y FactorService

---

## Success Metrics

- **Regime Testing**: % usuarios que prueban escenarios, reducción promedio de exposición
- **Thesis Arena**: Challenges por usuario, cambio promedio de conviction, retención
- **Community Feed**: Posts por día, conversión Arena → Feed, engagement
- **Model Validation**: Performance de portfolios seleccionados vs benchmarks, % que superan benchmark

---

## Files to Create/Modify

### Backend (New)
- `backend/api/routes/regime_testing.py`
- `backend/api/services/asset_regime_service.py`
- `backend/api/routes/thesis_arena.py` (actualizado para comunidades)
- `backend/api/services/conviction_service.py`
- `backend/api/services/portfolio_selection_service.py`
- `backend/api/routes/portfolio_tracking.py`
- `backend/api/services/model_retraining.py`
- `backend/prompts/communities/value.txt`
- `backend/prompts/communities/crypto_bro.txt`
- `backend/prompts/communities/growth.txt`
- `backend/prompts/communities/contrarian.txt`
- `backend/api/migrations/add_arena_communities.sql`
- `backend/api/migrations/add_model_portfolio_tracking.sql`

### Backend (Modify)
- `backend/api/routes/monte_carlo.py` (extender para regime testing)
- `backend/api/routes/community.py` (enhance posts, add rankings)
- `backend/api/routes/tactical_allocation.py` (integrar con portfolio selection)

### Frontend (New)
- `frontend/caria-app/components/widgets/RegimeTestWidget.tsx`
- `frontend/caria-app/components/widgets/ProtectionVisualization.tsx`
- `frontend/caria-app/components/widgets/RegimeTestResults.tsx`
- `frontend/caria-app/components/widgets/ThesisArena.tsx` (actualizado para comunidades)
- `frontend/caria-app/components/widgets/CommunityCard.tsx`
- `frontend/caria-app/components/widgets/CommunityTooltip.tsx`
- `frontend/caria-app/components/modals/ArenaThreadModal.tsx`
- `frontend/caria-app/components/modals/ThesisEditorModal.tsx`
- `frontend/caria-app/components/widgets/RankingsWidget.tsx`
- `frontend/caria-app/components/widgets/ModelPortfolioWidget.tsx`
- `frontend/caria-app/components/widgets/PortfolioPerformance.tsx`
- `frontend/caria-app/components/widgets/ModelValidationDashboard.tsx`

### Frontend (Modify)
- `frontend/caria-app/components/widgets/CommunityIdeas.tsx` → `CommunityFeed.tsx` (enhance)
- `frontend/caria-app/components/Dashboard.tsx` (add new widgets)
- `frontend/caria-app/components/widgets/IdealPortfolio.tsx` (reemplazar con ModelPortfolioWidget)
- `frontend/caria-app/components/widgets/MonteCarloSimulation.tsx` (integrar con regime testing)

### Dependencies
- Frontend: No nuevas dependencias (usar tooltips nativos de React)
- Backend: No nuevas dependencias (usar Gemini/LLM existente)

---

## Notas de Implementación

### Iconos de Comunidades
- Usar emojis o SVG custom (no imágenes reales)
- Value: 💎 o SVG de diamante/gem
- Crypto Bro: 🚀 o SVG de cohete
- Growth: 📈 o SVG de gráfico ascendente
- Contrarian: 🔄 o SVG de flechas opuestas

### Tooltips
- Usar `title` attribute nativo o librería ligera como `react-tooltip`
- Aparecer en hover (no click)
- Mostrar descripción de características del oponente

### Regime Testing
- Reutilizar Monte Carlo existente
- Agregar visualización de protección (gauge/barra)
- Mostrar recomendaciones basadas en exposición

### Model Portfolio Selection
- Usar factores existentes para identificar outliers
- Limitar a 10-20 holdings para practicidad
- Tracking automático de performance vs benchmarks
- Sistema de alertas si performance < threshold

