# Frontend Completion & Verification Prompt

## Contexto del Proyecto

Caria es una aplicación de inversión que combina análisis cuantitativo, modelos de machine learning, y herramientas de comunidad para ayudar a los usuarios a tomar decisiones de inversión informadas.

**Stack Tecnológico:**
- Frontend: React + TypeScript + Vite
- Backend: FastAPI (Python) en Google Cloud Run
- Base de datos: PostgreSQL (Cloud SQL)
- Deploy: Frontend en Vercel, Backend en Cloud Run

## Estado Actual del Frontend

### Componentes Implementados

#### Widgets Principales
1. **Portfolio** - Muestra holdings del usuario con precios en tiempo real
2. **ModelOutlook** - Gauge del régimen macroeconómico actual (Expansion, Slowdown, Recession, Stress)
3. **FearGreedIndex** - Índice CNN Fear & Greed en tiempo real con gauge semicircular
4. **HoldingsManager** - Gestión de holdings (agregar/editar/eliminar)
5. **TopMovers** - Movimientos del mercado
6. **GlobalMarketBar** - Barra de indicadores globales

#### Sección de Investigación
7. **ResearchSection** - Agrupa:
   - **ValuationTool** - Valuación DCF y simulaciones Monte Carlo
   - **RedditSentiment** - Análisis de sentimiento de Reddit
   - **Resources** - Artículos y recursos recomendados

#### Nuevas Funcionalidades Implementadas
8. **RegimeTestWidget** - Prueba de cartera contra diferentes regímenes macroeconómicos
   - Dropdown para seleccionar régimen (Expansion, Slowdown, Recession, Stress)
   - Visualización de nivel de protección (gauge)
   - Resultados con drawdown estimado y recomendaciones
   - Integración con Monte Carlo para visualización

9. **RankingsWidget** - Rankings de la comunidad
   - Tabs: Top Communities, Hot Theses, Survivors
   - Muestra comunidades más activas, tesis trending, y tesis con alta convicción mantenida

10. **CommunityFeed** - Feed mejorado de la comunidad (reemplaza CommunityIdeas)
    - Búsqueda por título, preview, ticker, usuario
    - Badges de Arena para posts originados en Thesis Arena
    - Link a threads de Arena
    - Ordenamiento: upvotes, fecha, análisis

11. **ThesisArena** - Modal para desafiar tesis de inversión
    - Input de tesis y ticker
    - Slider de convicción inicial
    - Grid 2x2 de comunidades (Value Investor, Crypto Bro, Growth Investor, Contrarian)
    - Muestra impacto en convicción
    - Botones: "Continuar Conversación" y "Publicar en Feed"

12. **ThesisEditorModal** - Editor de posts para la comunidad
    - Campos: título, ticker, preview, full thesis
    - Validación LLM de calidad
    - Pre-fill desde Arena
    - Badge de Arena cuando aplica

13. **ArenaThreadModal** - Conversaciones multi-ronda con comunidades
    - Historial de rounds
    - Continuar conversación

14. **ModelPortfolioWidget** - Portfolios seleccionados por el modelo
    - Selección de tipo: balanced, outlier, random
    - Slider de número de holdings (10-20)
    - Lista de portfolios activos
    - Detalles de holdings con allocations
    - Integración con PortfolioPerformance

15. **PortfolioPerformance** - Métricas de performance vs benchmarks
    - Return, Alpha, Beta, Sharpe Ratio
    - Max Drawdown, Volatility
    - Comparación con S&P 500, QQQ, VTI

16. **ModelValidationDashboard** - Dashboard admin
    - Análisis de performance de portfolios modelo
    - Trigger de retraining
    - Historial de retraining
    - Estadísticas agregadas

#### Componentes de Soporte
- **WidgetCard** - Wrapper común para todos los widgets
- **OnboardingTour** - Tour de onboarding
- **StartAnalysisCTA** - CTA principal con botón "Enter Arena"
- **CommunityCard** - Tarjeta de respuesta de comunidad
- **CommunityTooltip** - Tooltip con descripción de comunidad
- **ProtectionVisualization** - Gauge de nivel de protección
- **RegimeTestResults** - Resultados de prueba de régimen

### Estilo y Diseño

**Variables CSS (definidas en el tema):**
- `--color-bg-primary` - Fondo principal (dark)
- `--color-bg-secondary` - Fondo secundario
- `--color-bg-tertiary` - Fondo terciario/borders
- `--color-cream` - Texto principal (crema/blanco)
- `--color-text-primary` - Texto primario
- `--color-text-secondary` - Texto secundario
- `--color-text-muted` - Texto muted
- `--color-primary` - Color primario (azul/acento)
- `--color-secondary` - Color secundario
- `--font-display` - Fuente para títulos
- `--font-body` - Fuente para cuerpo

**Patrones de Diseño:**
- Todos los widgets usan `WidgetCard` como wrapper
- Espaciado consistente: `space-y-6` o `space-y-7` entre widgets
- Animaciones: `fade-in` con delays escalonados
- Hover effects en botones e interactivos
- Gauge visualizations para métricas (ModelOutlook, FearGreedIndex, ProtectionVisualization)

## Tareas de Finalización

### 1. Verificación de Funcionalidades

#### Autenticación
- [ ] Registro de usuario funciona correctamente
- [ ] Login funciona correctamente
- [ ] Logout funciona correctamente
- [ ] Tokens se refrescan automáticamente
- [ ] Manejo de errores de autenticación

#### Portfolio Management
- [ ] Agregar holdings funciona
- [ ] Editar holdings funciona
- [ ] Eliminar holdings funciona
- [ ] Precios se actualizan en tiempo real
- [ ] Cálculos de gain/loss son correctos

#### Model Outlook & Regime
- [ ] Régimen se carga correctamente
- [ ] Gauge muestra el régimen correcto
- [ ] RegimeTestWidget funciona con holdings del usuario
- [ ] Monte Carlo se ejecuta correctamente
- [ ] Visualizaciones se renderizan correctamente

#### Fear & Greed Index
- [ ] Índice se carga desde la API
- [ ] Gauge se renderiza correctamente
- [ ] Clasificación es correcta
- [ ] Cambio desde día anterior se muestra
- [ ] Auto-refresh cada 5 minutos funciona

#### Thesis Arena
- [ ] Modal se abre correctamente
- [ ] Input de tesis funciona
- [ ] Slider de convicción funciona
- [ ] Challenge con comunidades funciona
- [ ] Respuestas de comunidades se muestran
- [ ] Impacto en convicción se calcula
- [ ] Botón "Continuar Conversación" abre ArenaThreadModal
- [ ] Botón "Publicar en Feed" abre ThesisEditorModal con pre-fill

#### Community Features
- [ ] CommunityFeed muestra posts
- [ ] Búsqueda funciona
- [ ] Ordenamiento funciona
- [ ] Badges de Arena se muestran
- [ ] Links a threads de Arena funcionan
- [ ] Voting funciona
- [ ] RankingsWidget muestra datos correctos
- [ ] Tabs funcionan (Top Communities, Hot Theses, Survivors)

#### Model Portfolio
- [ ] Selección de portfolio funciona
- [ ] Lista de portfolios se carga
- [ ] Detalles de holdings se muestran
- [ ] PortfolioPerformance muestra métricas
- [ ] ModelValidationDashboard muestra análisis

#### Research Section
- [ ] ValuationTool funciona
- [ ] RedditSentiment muestra datos
- [ ] Resources muestra contenido

### 2. Mejoras de UI/UX

#### Responsive Design
- [ ] Verificar en mobile (320px, 375px, 414px)
- [ ] Verificar en tablet (768px, 1024px)
- [ ] Verificar en desktop (1280px, 1920px)
- [ ] Grid se adapta correctamente
- [ ] Widgets no se rompen en pantallas pequeñas

#### Accesibilidad
- [ ] Contraste de colores cumple WCAG AA
- [ ] Navegación por teclado funciona
- [ ] Screen readers pueden leer el contenido
- [ ] Focus states son visibles
- [ ] Labels están asociados correctamente

#### Performance
- [ ] Carga inicial < 3 segundos
- [ ] Lazy loading de componentes pesados
- [ ] Imágenes optimizadas
- [ ] No hay memory leaks
- [ ] Re-renders innecesarios minimizados

#### Consistencia Visual
- [ ] Todos los widgets usan el mismo estilo
- [ ] Espaciado consistente
- [ ] Tipografía consistente
- [ ] Colores siguen el tema
- [ ] Iconos son consistentes

### 3. Testing

#### Funcional Testing
- [ ] Todos los endpoints del backend responden
- [ ] Manejo de errores de red
- [ ] Loading states se muestran
- [ ] Empty states se muestran
- [ ] Validación de formularios funciona

#### Integration Testing
- [ ] Flujo completo de registro → login → uso
- [ ] Flujo de Thesis Arena completo
- [ ] Flujo de Community Feed completo
- [ ] Flujo de Model Portfolio completo

#### Visual Testing
- [ ] Screenshots de todos los widgets
- [ ] Verificar en diferentes navegadores (Chrome, Firefox, Safari, Edge)
- [ ] Verificar en modo claro/oscuro (si aplica)

### 4. Bugs Conocidos a Verificar

1. **CORS Issues** - Verificar que todas las requests funcionan desde producción
2. **API Connection** - Verificar que FMP, Gemini, Reddit APIs funcionan
3. **Database Connections** - Verificar que Cloud SQL funciona correctamente
4. **WebSocket Chat** - Verificar que funciona si está implementado

### 5. Documentación

- [ ] README actualizado con instrucciones de desarrollo
- [ ] Documentación de componentes principales
- [ ] Guía de estilos
- [ ] Guía de testing

## Endpoints del Backend Disponibles

### Autenticación
- `POST /api/auth/register` - Registro
- `POST /api/auth/login` - Login
- `POST /api/auth/refresh` - Refresh token
- `GET /api/auth/me` - Usuario actual

### Portfolio
- `GET /api/holdings` - Lista de holdings
- `POST /api/holdings` - Crear holding
- `PUT /api/holdings/{id}` - Actualizar holding
- `DELETE /api/holdings/{id}` - Eliminar holding
- `GET /api/holdings/with-prices` - Holdings con precios

### Market Data
- `GET /api/market/fear-greed` - Fear & Greed Index
- `POST /api/prices/realtime` - Precios en tiempo real
- `GET /api/regime/current` - Régimen actual

### Thesis Arena
- `POST /api/thesis/arena/challenge` - Desafiar tesis
- `POST /api/thesis/arena/respond` - Responder en thread
- `GET /api/thesis/arena/thread/{id}` - Obtener thread

### Community
- `GET /api/community/posts` - Lista de posts (con search)
- `POST /api/community/posts` - Crear post
- `POST /api/community/posts/validate` - Validar post
- `POST /api/community/posts/{id}/vote` - Votar
- `GET /api/community/rankings` - Rankings

### Regime Testing
- `POST /api/portfolio/regime-test` - Probar cartera contra régimen

### Model Portfolio
- `POST /api/portfolio/model/select` - Seleccionar portfolio
- `GET /api/portfolio/model/track` - Tracking de performance
- `GET /api/portfolio/model/list` - Lista de portfolios
- `GET /api/portfolio/model/analyze` - Análisis de performance

### Monte Carlo
- `POST /api/portfolio/simulate` - Simulación Monte Carlo

## Variables de Entorno Necesarias

**Frontend (Vercel):**
- `VITE_API_URL` - URL del backend (ej: `https://caria-api-xxx.run.app`)

**Backend (Cloud Run):**
- `DATABASE_URL` - Connection string de PostgreSQL
- `GEMINI_API_KEY` - API key de Gemini
- `FMP_API_KEY` - API key de Financial Modeling Prep
- `REDDIT_CLIENT_ID` - Reddit client ID
- `REDDIT_CLIENT_SECRET` - Reddit client secret
- `REDDIT_USER_AGENT` - User agent para Reddit
- `JWT_SECRET_KEY` - Secret para JWT tokens
- `POSTGRES_PASSWORD` - Password de PostgreSQL
- `CORS_ORIGINS` - Orígenes permitidos (separados por `;`)

## Checklist de Verificación Final

### Funcionalidades Core
- [ ] Usuario puede registrarse y loguearse
- [ ] Usuario puede agregar/editar/eliminar holdings
- [ ] Precios se actualizan en tiempo real
- [ ] Model Outlook muestra régimen correcto
- [ ] Fear & Greed Index se carga y muestra correctamente
- [ ] Thesis Arena funciona end-to-end
- [ ] Community Feed funciona con búsqueda y filtros
- [ ] Rankings se muestran correctamente
- [ ] Regime Testing funciona
- [ ] Model Portfolio selection funciona
- [ ] Performance tracking funciona

### UI/UX
- [ ] Diseño es consistente en toda la app
- [ ] Responsive en mobile, tablet, desktop
- [ ] Loading states apropiados
- [ ] Error states apropiados
- [ ] Empty states apropiados
- [ ] Animaciones suaves
- [ ] Transiciones fluidas

### Performance
- [ ] Carga inicial rápida
- [ ] No hay lag en interacciones
- [ ] Imágenes optimizadas
- [ ] Bundle size razonable

### Testing
- [ ] Funciona en Chrome
- [ ] Funciona en Firefox
- [ ] Funciona en Safari
- [ ] Funciona en Edge
- [ ] Funciona en mobile browsers

## Notas Importantes

1. **Estilo Visual**: Mantener el diseño dark theme con colores cream/azul. Los gauges deben ser reconocibles pero adaptados al estilo de la app.

2. **Fear & Greed Index**: Debe mantener el formato reconocible de CNN (gauge semicircular, zonas de color, aguja) pero adaptado al estilo de la app.

3. **Thesis Arena**: Es una funcionalidad clave - asegurar que el flujo completo funciona: challenge → responses → continue conversation → publish to feed.

4. **Community Feed**: Debe ser intuitivo y fácil de usar, con búsqueda rápida y filtros claros.

5. **Model Portfolio**: Es una funcionalidad avanzada - asegurar que los datos se muestran correctamente y las métricas son comprensibles.

## Prioridades

1. **Alta**: Verificar que todas las funcionalidades core funcionan
2. **Alta**: Asegurar responsive design
3. **Media**: Mejorar UI/UX donde sea necesario
4. **Media**: Optimizar performance
5. **Baja**: Documentación adicional

## Recursos

- **Backend API Docs**: Disponible en `/docs` cuando el backend está corriendo
- **Componentes**: Ubicados en `frontend/caria-app/components/widgets/`
- **Servicios**: `frontend/caria-app/services/apiService.ts`
- **Estilos**: Variables CSS en el tema principal

---

**Objetivo Final**: Tener una aplicación frontend completamente funcional, bien diseñada, responsive, y lista para producción que integre todas las funcionalidades implementadas en el backend.

