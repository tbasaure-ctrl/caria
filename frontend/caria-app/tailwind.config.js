/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
        "./components/**/*.{js,ts,jsx,tsx}",
        "./App.tsx",
        "./frontend/caria-app/**/*.{js,ts,jsx,tsx}"
    ],
    theme: {
        extend: {
            colors: {
                // Core Backgrounds - Deep Navy/Black Theme
                bg: {
                    primary: '#020408',    // Deepest black/navy
                    secondary: '#0B101B',  // Card background
                    tertiary: '#151E32',   // Elevated surface
                    elevated: '#1E293B',   // Hover states
                    surface: '#243045',    // Input fields etc
                },
                // Text Colors
                text: {
                    primary: '#F1F5F9',    // Bright white-ish
                    secondary: '#94A3B8',  // Muted blue-grey
                    muted: '#64748B',      // Darker grey
                    subtle: '#475569',     // Very subtle
                },
                // Accent Colors
                accent: {
                    DEFAULT: '#38BDF8',    // Sky Blue
                    primary: '#38BDF8',
                    secondary: '#D4AF37',  // Gold (for financial elegance)
                    tertiary: '#0EA5E9',   // Darker Blue
                    cyan: '#22D3EE',       // Bright Cyan (from images)
                    gold: '#C5A059',       // Muted Gold
                },
                // Semantic
                positive: {
                    DEFAULT: '#10B981',
                    muted: 'rgba(16, 185, 129, 0.1)',
                },
                negative: {
                    DEFAULT: '#EF4444',
                    muted: 'rgba(239, 68, 68, 0.1)',
                },
                warning: {
                    DEFAULT: '#F59E0B',
                    muted: 'rgba(245, 158, 11, 0.1)',
                },
                // Border
                border: {
                    subtle: 'rgba(148, 163, 184, 0.1)',
                    DEFAULT: 'rgba(148, 163, 184, 0.2)',
                    emphasis: 'rgba(56, 189, 248, 0.4)',
                    gold: 'rgba(212, 175, 55, 0.3)',
                },
            },
            fontFamily: {
                display: ['Instrument Serif', 'Playfair Display', 'serif'], // Serif for titles
                body: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
                mono: ['JetBrains Mono', 'SF Mono', 'Consolas', 'monospace'],
                editorial: ['Spectral', 'Georgia', 'serif'],
            },
            backgroundImage: {
                'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
                'hero-glow': 'radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.15) 0%, rgba(2, 4, 8, 0) 70%)',
                'card-gradient': 'linear-gradient(180deg, rgba(11, 16, 27, 0.8) 0%, rgba(11, 16, 27, 1) 100%)',
            },
            boxShadow: {
                'glow-sm': '0 0 10px rgba(56, 189, 248, 0.1)',
                'glow-md': '0 0 20px rgba(56, 189, 248, 0.15)',
                'glow-gold': '0 0 15px rgba(212, 175, 55, 0.15)',
            },
            animation: {
                'fade-in': 'fadeIn 0.5s ease-out forwards',
                'slide-up': 'slideUp 0.6s ease-out forwards',
            },
            keyframes: {
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
                slideUp: {
                    '0%': { opacity: '0', transform: 'translateY(20px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
            }
        },
    },
    plugins: [],
}
