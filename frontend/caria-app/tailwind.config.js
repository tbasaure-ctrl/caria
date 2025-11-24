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
                primary: {
                    DEFAULT: '#5A2A27',
                    light: '#6B3634',
                    dark: '#3D1C1A',
                },
                secondary: '#8B7355',
                accent: '#3A5A40',
                blue: {
                    DEFAULT: '#4A6FA5',
                    light: '#5B7FB5',
                    dark: '#3A5A85',
                },
                cream: {
                    DEFAULT: '#D4D4D4',
                    dark: '#B8B8B8',
                },
            },
            fontFamily: {
                display: ['"Cormorant Garamond"', 'serif'],
                body: ['"Manrope"', 'sans-serif'],
                mono: ['"JetBrains Mono"', 'monospace'],
            },
        },
    },
    plugins: [],
}
