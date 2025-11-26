/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
        "./components/**/*.{js,ts,jsx,tsx}",
        "./App.tsx"
    ],
    theme: {
        extend: {
            colors: {
                // Core Backgrounds
                bg: {
                    primary: '#0A0E14',
                    secondary: '#0F1419',
                    tertiary: '#151C24',
                    elevated: '#1A222D',
                    surface: '#1E2733',
                },
                // Text Colors
                text: {
                    primary: '#F2F4F7',
                    secondary: '#B4BCC8',
                    muted: '#6B7A8F',
                    subtle: '#4A5568',
                },
                // Accent Colors
                accent: {
                    DEFAULT: '#2E7CF6',
                    primary: '#2E7CF6',
                    secondary: '#1E88E5',
                    tertiary: '#0D47A1',
                },
                // Semantic
                positive: {
                    DEFAULT: '#00C853',
                    muted: '#1B4332',
                },
                negative: {
                    DEFAULT: '#F44336',
                    muted: '#4A1C1C',
                },
                warning: {
                    DEFAULT: '#FF9800',
                    muted: '#4A3200',
                },
                // Border
                border: {
                    subtle: 'rgba(107, 122, 143, 0.15)',
                    DEFAULT: 'rgba(107, 122, 143, 0.25)',
                    emphasis: 'rgba(46, 124, 246, 0.4)',
                },
            },
            fontFamily: {
                display: ['Suisse Intl', 'Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
                body: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
                mono: ['JetBrains Mono', 'SF Mono', 'Consolas', 'monospace'],
                editorial: ['Spectral', 'Georgia', 'Times New Roman', 'serif'],
            },
            fontSize: {
                // Display
                'display-xl': ['56px', { lineHeight: '1.05', letterSpacing: '-0.03em', fontWeight: '700' }],
                'display-lg': ['48px', { lineHeight: '1.1', letterSpacing: '-0.025em', fontWeight: '700' }],
                'display-md': ['36px', { lineHeight: '1.15', letterSpacing: '-0.02em', fontWeight: '600' }],
                // Headlines
                'headline-lg': ['28px', { lineHeight: '1.2', letterSpacing: '-0.015em', fontWeight: '600' }],
                'headline-md': ['24px', { lineHeight: '1.25', letterSpacing: '-0.01em', fontWeight: '600' }],
                'headline-sm': ['20px', { lineHeight: '1.3', letterSpacing: '-0.005em', fontWeight: '600' }],
                // Body
                'body-lg': ['18px', { lineHeight: '1.7', fontWeight: '400' }],
                'body-md': ['16px', { lineHeight: '1.6', fontWeight: '400' }],
                'body-sm': ['14px', { lineHeight: '1.5', fontWeight: '400' }],
                // Labels
                'label-lg': ['13px', { lineHeight: '1.4', letterSpacing: '0.02em', fontWeight: '600' }],
                'label-md': ['11px', { lineHeight: '1.4', letterSpacing: '0.04em', fontWeight: '600' }],
                'label-sm': ['10px', { lineHeight: '1.3', letterSpacing: '0.06em', fontWeight: '600' }],
                // Data
                'data-lg': ['24px', { lineHeight: '1.2', letterSpacing: '-0.02em', fontWeight: '500' }],
                'data-md': ['16px', { lineHeight: '1.3', letterSpacing: '-0.01em', fontWeight: '500' }],
                'data-sm': ['13px', { lineHeight: '1.4', fontWeight: '500' }],
            },
            spacing: {
                '18': '72px',
                '22': '88px',
                '26': '104px',
                '30': '120px',
            },
            borderRadius: {
                'sm': '4px',
                'md': '6px',
                'lg': '8px',
                'xl': '12px',
            },
            boxShadow: {
                'sm': '0 1px 2px rgba(0, 0, 0, 0.4)',
                'md': '0 4px 12px rgba(0, 0, 0, 0.5)',
                'lg': '0 8px 24px rgba(0, 0, 0, 0.6)',
                'xl': '0 16px 48px rgba(0, 0, 0, 0.7)',
                'glow-accent': '0 0 24px rgba(46, 124, 246, 0.15)',
            },
            transitionTimingFunction: {
                'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
            },
            transitionDuration: {
                'fast': '150ms',
                'base': '250ms',
                'slow': '400ms',
            },
            animation: {
                'fade-in': 'fadeIn 0.4s ease-out forwards',
                'fade-in-up': 'fadeInUp 0.5s ease-out forwards',
                'fade-in-scale': 'fadeInScale 0.4s ease-out forwards',
                'slide-in-left': 'slideInLeft 0.5s ease-out forwards',
                'pulse-subtle': 'pulse-subtle 2s ease-in-out infinite',
                'ticker': 'ticker-scroll 30s linear infinite',
            },
            keyframes: {
                fadeIn: {
                    'from': { opacity: '0' },
                    'to': { opacity: '1' },
                },
                fadeInUp: {
                    'from': { opacity: '0', transform: 'translateY(12px)' },
                    'to': { opacity: '1', transform: 'translateY(0)' },
                },
                fadeInScale: {
                    'from': { opacity: '0', transform: 'scale(0.98)' },
                    'to': { opacity: '1', transform: 'scale(1)' },
                },
                slideInLeft: {
                    'from': { opacity: '0', transform: 'translateX(-20px)' },
                    'to': { opacity: '1', transform: 'translateX(0)' },
                },
                'pulse-subtle': {
                    '0%, 100%': { opacity: '1' },
                    '50%': { opacity: '0.7' },
                },
                'ticker-scroll': {
                    '0%': { transform: 'translateX(0)' },
                    '100%': { transform: 'translateX(-50%)' },
                },
            },
        },
    },
    plugins: [],
}
