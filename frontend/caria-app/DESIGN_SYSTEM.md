# Caria Design System

## Overview
Caria's design follows a **"Financial Editorial"** aesthetic that merges the sophistication of premium financial publications with modern data visualization. The design emphasizes elegance, clarity, and intentional details.

## Design Philosophy

### Core Principles
1. **Sophisticated Minimalism**: Every element serves a purpose
2. **Editorial Quality**: Typography and layout inspired by premium financial publications
3. **Data-First**: Information hierarchy that prioritizes insights
4. **Purposeful Motion**: Animations that guide attention and create delight
5. **Refined Details**: Subtle textures, shadows, and accents that elevate the experience

## Color Palette

### Primary Colors
```css
--color-primary: #8B1E3F        /* Deep Burgundy - Primary actions, accents */
--color-primary-light: #A8385D  /* Lighter burgundy for hover states */
--color-primary-dark: #6B1530   /* Darker burgundy for active states */
```

### Secondary Colors
```css
--color-secondary: #D4AF37      /* Muted Gold - Data highlights, important metrics */
--color-accent: #2D5016         /* Forest Green - Positive indicators, growth */
```

### Background Colors
```css
--color-bg-primary: #0A0D12     /* Deep Navy - Main background */
--color-bg-secondary: #12161D   /* Secondary surfaces, cards */
--color-bg-tertiary: #1A1F28    /* Tertiary surfaces, nested elements */
```

### Text Colors
```css
--color-cream: #F4F1DE          /* Warm Cream - Primary text, headlines */
--color-cream-dark: #E8E3CA     /* Darker cream for subtle emphasis */
--color-text-primary: #F4F1DE   /* Primary readable text */
--color-text-secondary: #C4BFA8 /* Secondary text, descriptions */
--color-text-muted: #8B8570     /* Muted text, labels, captions */
```

### Usage Guidelines
- **Burgundy (`primary`)**: Use for primary CTAs, active states, important actions
- **Gold (`secondary`)**: Reserve for key metrics, data highlights, success states
- **Forest Green (`accent`)**: Positive indicators, growth metrics, profit
- **Cream shades**: All text content, maintaining readability hierarchy
- **Navy backgrounds**: Create depth through layered surfaces

## Typography

### Font Families
```css
--font-display: 'Cormorant Garamond', serif  /* Headlines, large text */
--font-body: 'Manrope', sans-serif           /* Body text, UI elements */
--font-mono: 'JetBrains Mono', monospace     /* Data, code, labels */
```

### Type Scale
- **Display**: 4xl-8xl (Headlines, hero text) - Cormorant Garamond
- **Heading**: 2xl-3xl (Section headers) - Cormorant Garamond
- **Body**: base-lg (Content, descriptions) - Manrope
- **Caption**: sm-xs (Labels, metadata) - Manrope or JetBrains Mono

### Typography Best Practices
1. Use **Cormorant Garamond** for emotional impact and brand voice
2. Use **Manrope** for readability and UI clarity
3. Use **JetBrains Mono** for data that needs precision (tickers, prices, percentages)
4. Maintain clear hierarchy: display → heading → body → caption
5. Apply `-0.02em` letter spacing on large display text

## Layout & Spacing

### Grid System
- Dashboard uses a 3-column grid on large screens
- 2:1 ratio (main content : sidebar)
- Consistent gap of `1.75rem` (7 spacing units)

### Spacing Scale
```css
--spacing-section: 6rem  /* Between major sections */
```
- Internal padding: 1.25rem - 2rem depending on component
- Card padding: 1.25rem
- Element gaps: 0.75rem - 1rem

### Border Radius
```css
--border-radius: 0.75rem  /* Standard for all cards and buttons */
```

## Components

### Cards (WidgetCard)
- Background: `--color-bg-secondary`
- Border: `1px solid var(--color-bg-tertiary)`
- Shadow: Subtle `0 2px 10px rgba(0,0,0,0.1)`
- Title: Uppercase, mono font, muted color
- Decorative accent line in header

### Buttons
**Primary Button:**
- Background: `--color-primary`
- Text: `--color-cream`
- Hover: Lift effect + lighter background
- Shadow: Colored glow matching button

**Secondary Button:**
- Border: `1px solid var(--color-primary)`
- Background: transparent
- Hover: Fill with primary color

### Data Visualization
- Use semantic colors: green for positive, burgundy for negative
- Monospace font for all numerical data
- Smooth transitions (0.5s - 1s) for data updates
- Animated gauges and charts with easing

## Animation & Motion

### Timing Functions
- **Ease-out**: Default for most transitions (0.2s - 0.3s)
- **Ease-in-out**: For complex multi-step animations
- **Linear**: For continuous animations (like shimmer effects)

### Key Animations
```css
fadeIn: opacity + translateY (1s ease-out)
fadeInScale: opacity + scale (0.6s ease-out)
slideInLeft/Right: opacity + translateX (0.8s ease-out)
```

### Animation Delays
Use staggered delays to create orchestrated reveals:
- `delay-100` to `delay-800` (0.1s increments)
- Apply to sibling elements for cascading effect

### Micro-interactions
- Hover states: Subtle lift (2px), color shift, border glow
- Button hover: Scale 1.05 or translateY(-2px)
- Card hover: Border color change, subtle lift
- Data updates: Smooth color transitions (0.5s)

## Visual Details

### Textures
- Subtle noise overlay on body (`opacity: 0.03`)
- Creates depth without distraction

### Shadows
- Light shadows for cards: `0 2px 10px rgba(0,0,0,0.1)`
- Elevated shadows for CTAs: `0 10px 30px -10px var(--color-primary)`
- Glow effects for active elements: `0 0 8px var(--color-cream)`

### Gradients
- Subtle radial gradients for decorative backgrounds
- Linear gradients for text effects (cream → gold)
- Border gradients: `linear-gradient(135deg, primary, secondary)`

### Scrollbar
- Track: `--color-bg-secondary`
- Thumb: `--color-primary`
- Hover: `--color-primary-light`

### Selection
- Background: `--color-primary`
- Text: `--color-cream`

## Accessibility

### Color Contrast
All text combinations meet WCAG AA standards:
- Cream on navy backgrounds: 9.5:1
- Secondary text on backgrounds: 5.2:1
- Muted text: Use sparingly, only for non-essential content

### Focus States
- Visible focus rings using `--color-primary`
- Keyboard navigation fully supported
- ARIA labels on all interactive elements

### Motion
- All animations respect `prefers-reduced-motion`
- Essential information never conveyed through motion alone

## Best Practices

### Do's
✓ Use the design system variables consistently
✓ Maintain clear visual hierarchy
✓ Apply animations purposefully
✓ Test on dark backgrounds
✓ Use appropriate font families for content type
✓ Layer surfaces for depth (primary → secondary → tertiary)

### Don'ts
✗ Don't use arbitrary colors outside the palette
✗ Don't mix font families within a single text block
✗ Don't over-animate - less is more
✗ Don't use pure white or pure black
✗ Don't ignore the spacing scale
✗ Don't create visual clutter

## Implementation Notes

### CSS Variables
All design tokens are defined in `index.html` as CSS custom properties. Use them consistently:
```javascript
style={{backgroundColor: 'var(--color-bg-secondary)'}}
```

### Inline Styles vs Classes
- Use CSS variables with inline styles for dynamic theming
- Tailwind classes for layout and spacing
- Custom animations defined globally in `<style>` tag

### Component Patterns
1. **WidgetCard**: Base component for all dashboard widgets
2. **Hero**: Large display text with decorative elements
3. **Buttons**: Consistent hover states and transitions
4. **Data displays**: Monospace fonts, semantic colors

---

*This design system represents Caria's commitment to sophisticated, user-centered financial interfaces that respect both data and design.*
