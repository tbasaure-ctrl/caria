# Estructura de Componentes de CARIA

Este documento describe la estructura y organización de los componentes React en CARIA.

## Estructura de Directorios

```
caria-app/
├── components/
│   ├── Dashboard.tsx           # Dashboard principal
│   ├── LoginModal.tsx          # Modal de autenticación
│   ├── RegisterModal.tsx       # Modal de registro
│   ├── AnalysisTool.tsx        # Herramienta de análisis/chat
│   ├── Icons.tsx               # Iconos SVG reutilizables
│   ├── Message.tsx             # Componente de mensaje
│   └── widgets/                # Widgets del dashboard
│       ├── Portfolio.tsx       # Portfolio snapshot
│       ├── MarketIndices.tsx    # Índices de mercado
│       ├── GlobalMarketBar.tsx # Barra de mercado global
│       ├── IdealPortfolio.tsx  # Portfolio ideal según régimen
│       ├── ModelOutlook.tsx    # Outlook del modelo
│       ├── ValuationTool.tsx   # Herramienta de valuación rápida
│       └── HoldingsManager.tsx # Gestor de holdings
├── services/
│   └── apiService.ts           # Servicio de API
└── styles/
    └── (estilos globales)
```

## Componentes Principales

### Dashboard.tsx

**Propósito**: Layout principal de la aplicación

**Características**:
- Grid layout responsivo
- Integra todos los widgets
- Maneja estado global (usuario, régimen, etc.)
- Navegación y autenticación

**Uso**:
```tsx
<Dashboard />
```

### WidgetCard

**Propósito**: Wrapper estándar para todos los widgets

**Props**:
- `title`: Título del widget
- `id`: ID único (opcional)

**Uso**:
```tsx
<WidgetCard title="MI WIDGET" id="widget-id">
    {/* Contenido */}
</WidgetCard>
```

## Widgets

### Portfolio.tsx

**Propósito**: Muestra snapshot del portfolio del usuario

**Características**:
- Muestra holdings con precios en tiempo real
- Gráfico de performance
- Gráfico de allocation (pie chart)
- Resumen de ganancias/pérdidas

**Datos que consume**:
- `/api/holdings/with-prices`

### MarketIndices.tsx

**Propósito**: Muestra índices de mercado principales

**Características**:
- Precios en tiempo real
- Polling cada 30 segundos
- Indicadores de cambio porcentual

**Datos que consume**:
- `/api/prices/realtime`

### IdealPortfolio.tsx

**Propósito**: Muestra portfolio ideal según régimen macro

**Características**:
- Basado en factor screening
- Ajustado según régimen actual
- Lista de empresas recomendadas

**Datos que consume**:
- `/api/regime/current`
- `/api/factors/screen`

### ModelOutlook.tsx

**Propósito**: Muestra outlook del modelo de régimen

**Características**:
- Probabilidades de régimen
- Confianza del modelo
- Features usadas

**Datos que consume**:
- `/api/regime/current`

### ValuationTool.tsx

**Propósito**: Herramienta rápida de valuación

**Características**:
- Input de ticker
- Múltiples métodos de valuación
- Resultados en tiempo real

**Datos que consume**:
- `/api/valuation/{ticker}`

### HoldingsManager.tsx

**Propósito**: Gestión de holdings del usuario

**Características**:
- Agregar nuevas posiciones
- Eliminar posiciones existentes
- Lista de holdings actuales

**Datos que consume**:
- `/api/holdings` (GET, POST, DELETE)

## Servicios

### apiService.ts

**Propósito**: Centraliza todas las llamadas a la API

**Funciones principales**:
- `fetchWithAuth()`: Fetch con autenticación
- `fetchPrices()`: Obtener precios en tiempo real
- `fetchHoldings()`: Obtener holdings
- `fetchHoldingsWithPrices()`: Holdings con precios
- `createHolding()`: Crear nuevo holding
- `deleteHolding()`: Eliminar holding
- `fetchValuation()`: Obtener valuación
- `fetchRegime()`: Obtener régimen actual
- `fetchFactorScreen()`: Factor screening
- `fetchAnalysis()`: Análisis de tesis

**Uso**:
```tsx
import { fetchPrices } from '../services/apiService';

const prices = await fetchPrices(['AAPL', 'MSFT']);
```

## Patrones de Diseño

### Polling Pattern

Los widgets que necesitan datos en tiempo real usan polling:

```tsx
useEffect(() => {
    const updateData = async () => {
        const data = await fetchData();
        setData(data);
    };
    
    updateData();
    const interval = setInterval(updateData, 30000); // 30 segundos
    
    return () => clearInterval(interval);
}, []);
```

### Error Handling Pattern

Manejo consistente de errores:

```tsx
try {
    const data = await fetchData();
    setData(data);
    setError(null);
} catch (err) {
    setError(err instanceof Error ? err.message : 'Error desconocido');
    setData(null);
}
```

### Loading States Pattern

Estados de carga consistentes:

```tsx
const [loading, setLoading] = useState(true);

useEffect(() => {
    const loadData = async () => {
        setLoading(true);
        try {
            const data = await fetchData();
            setData(data);
        } finally {
            setLoading(false);
        }
    };
    loadData();
}, []);
```

## Convenciones

### Nombres de Componentes

- Componentes principales: PascalCase (`Dashboard.tsx`)
- Widgets: PascalCase (`Portfolio.tsx`)
- Utilidades: camelCase (`apiService.ts`)

### Props

- Usar TypeScript interfaces para props
- Props opcionales con `?`
- Props con valores por defecto cuando sea apropiado

### Estilos

- Usar Tailwind CSS para todos los estilos
- Mantener consistencia de colores
- Usar clases de utilidad en lugar de CSS personalizado

## Agregar Nuevos Componentes

1. **Crear archivo** en la ubicación apropiada
2. **Definir interface** de props con TypeScript
3. **Implementar componente** siguiendo patrones existentes
4. **Agregar al Dashboard** o donde corresponda
5. **Probar** en diferentes tamaños de pantalla

## Recursos

- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)

