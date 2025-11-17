# Guía de Edición y Personalización de UI

Esta guía te ayudará a entender cómo editar y modificar la interfaz de usuario de CARIA.

## Estructura del Proyecto

```
caria_data/caria-app/
├── components/          # Componentes React principales
│   ├── Dashboard.tsx    # Dashboard principal
│   ├── LoginModal.tsx   # Modal de login
│   └── widgets/         # Widgets del dashboard
│       ├── Portfolio.tsx
│       ├── MarketIndices.tsx
│       ├── IdealPortfolio.tsx
│       └── ...
├── services/
│   └── apiService.ts    # Servicio para llamadas a la API
├── styles/              # Estilos globales
└── package.json         # Dependencias
```

## Estilos: Tailwind CSS

CARIA usa **Tailwind CSS** para estilos. Tailwind es un framework utility-first que te permite diseñar rápidamente sin escribir CSS personalizado.

### Conceptos Básicos

**Clases de utilidad**:
```tsx
// Espaciado
<div className="p-4">        // padding: 1rem
<div className="m-2">         // margin: 0.5rem
<div className="space-y-4">   // espacio vertical entre hijos

// Colores
<div className="bg-slate-900">     // fondo oscuro
<div className="text-blue-400">    // texto azul
<div className="border-slate-800"> // borde gris oscuro

// Tamaños de texto
<h1 className="text-2xl">    // texto grande
<p className="text-sm">      // texto pequeño

// Flexbox
<div className="flex justify-between items-center">
```

### Tema Dark Actual

CARIA usa un tema dark con estos colores principales:

```tsx
// Colores principales
bg-slate-900      // Fondo principal (muy oscuro)
bg-slate-800      // Fondo secundario
bg-slate-700      // Fondo hover/activo
text-slate-100    // Texto principal (claro)
text-slate-400    // Texto secundario (gris)
text-slate-500    // Texto terciario (más gris)

// Colores de acento
text-blue-400     // Azul (éxito/info)
text-green-400    // Verde (positivo)
text-red-400      // Rojo (error/negativo)
text-purple-400   // Púrpura (destacado)
```

## Cómo Modificar Componentes

### Ejemplo 1: Cambiar Colores de un Widget

**Archivo**: `components/widgets/Portfolio.tsx`

```tsx
// Antes
<div className="bg-slate-900 border border-slate-800">

// Después (más moderno con gradiente)
<div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border border-slate-700">
```

### Ejemplo 2: Agregar Animaciones

```tsx
// Agregar hover effect
<button className="transition-all duration-200 hover:bg-slate-700 hover:scale-105">
  Click me
</button>
```

### Ejemplo 3: Mejorar Tipografía

```tsx
// Antes
<h2 className="text-lg">Título</h2>

// Después (con gradiente de texto)
<h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
  Título
</h2>
```

## Cómo Agregar Nuevos Widgets

1. **Crear nuevo componente** en `components/widgets/`:

```tsx
// components/widgets/NewWidget.tsx
import { WidgetCard } from './WidgetCard'; // Reutilizar card base

export const NewWidget: React.FC = () => {
    return (
        <WidgetCard title="MI NUEVO WIDGET">
            <div className="space-y-4">
                {/* Tu contenido aquí */}
            </div>
        </WidgetCard>
    );
};
```

2. **Agregar al Dashboard**:

```tsx
// components/Dashboard.tsx
import { NewWidget } from './widgets/NewWidget';

// En el JSX del Dashboard
<NewWidget />
```

## Cómo Modificar el Layout

**Archivo**: `components/Dashboard.tsx`

El dashboard usa CSS Grid. Para cambiar el layout:

```tsx
// Grid actual (3 columnas)
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

// Cambiar a 2 columnas
<div className="grid grid-cols-1 md:grid-cols-2 gap-6">

// Cambiar a 4 columnas
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
```

## Cómo Agregar Nuevas Páginas

1. **Crear componente de página**:

```tsx
// components/NewPage.tsx
export const NewPage: React.FC = () => {
    return (
        <div className="min-h-screen bg-slate-900 p-8">
            <h1 className="text-3xl font-bold text-slate-100">Nueva Página</h1>
            {/* Contenido */}
        </div>
    );
};
```

2. **Agregar ruta** (si usas React Router):

```tsx
import { NewPage } from './components/NewPage';

<Route path="/nueva-pagina" element={<NewPage />} />
```

## Cómo Personalizar Colores

### Opción 1: Modificar clases directamente

Busca y reemplaza clases de color en los componentes:

```bash
# Buscar todas las instancias de un color
grep -r "bg-slate-900" caria-app/components/
```

### Opción 2: Crear tema personalizado

Crea un archivo de configuración de tema:

```tsx
// styles/theme.ts
export const theme = {
    colors: {
        primary: 'slate-900',
        secondary: 'slate-800',
        accent: 'blue-400',
        // ... más colores
    }
};
```

Luego úsalo en componentes:

```tsx
import { theme } from '../styles/theme';

<div className={`bg-${theme.colors.primary}`}>
```

## Componentes Reutilizables

### WidgetCard

Wrapper estándar para todos los widgets:

```tsx
<WidgetCard title="TÍTULO" id="widget-id">
    {/* Contenido del widget */}
</WidgetCard>
```

### Botones

```tsx
// Botón primario
<button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded">
    Acción
</button>

// Botón secundario
<button className="bg-slate-700 hover:bg-slate-600 text-slate-100 px-4 py-2 rounded">
    Cancelar
</button>
```

## Mejores Prácticas

1. **Mantener consistencia**: Usa los mismos colores y espaciados en todo el proyecto
2. **Reutilizar componentes**: No duplicar código, crear componentes reutilizables
3. **Responsive design**: Siempre prueba en móvil, tablet y desktop
4. **Accesibilidad**: Usa semántica HTML correcta y aria-labels cuando sea necesario
5. **Performance**: Evita re-renders innecesarios usando `React.memo` cuando sea apropiado

## Herramientas Útiles

1. **Tailwind CSS IntelliSense** (VS Code extension): Autocompletado de clases
2. **React Developer Tools**: Inspeccionar componentes en el navegador
3. **Tailwind Play**: Probar clases rápidamente (https://play.tailwindcss.com/)

## Ejemplos Completos

### Widget Moderno con Gradientes

```tsx
export const ModernWidget: React.FC = () => {
    return (
        <WidgetCard title="WIDGET MODERNO">
            <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 rounded-lg p-6 border border-blue-500/20">
                <h3 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-4">
                    Título con Gradiente
                </h3>
                <p className="text-slate-300">
                    Contenido del widget con diseño moderno.
                </p>
            </div>
        </WidgetCard>
    );
};
```

### Widget con Animación

```tsx
export const AnimatedWidget: React.FC = () => {
    return (
        <WidgetCard title="WIDGET ANIMADO">
            <div className="space-y-4">
                <div className="transform transition-all duration-300 hover:scale-105 hover:shadow-lg">
                    <div className="bg-slate-800 rounded-lg p-4">
                        Hover sobre mí
                    </div>
                </div>
            </div>
        </WidgetCard>
    );
};
```

## Próximos Pasos

1. **Explora los componentes existentes** para entender la estructura
2. **Haz cambios pequeños** primero para familiarizarte
3. **Prueba en el navegador** después de cada cambio
4. **Consulta la documentación de Tailwind CSS** para más opciones: https://tailwindcss.com/docs

## Recursos

- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

