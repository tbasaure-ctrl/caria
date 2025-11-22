/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./**/*.{js,ts,jsx,tsx}",
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
        bg: {
          primary: '#0B0E11',
          secondary: '#13171C',
          tertiary: '#1C2127',
        },
        cream: {
          DEFAULT: '#D4D4D4',
          dark: '#B8B8B8',
        },
        text: {
          primary: '#D4D4D4',
          secondary: '#9CA3AF',
          muted: '#6B7280',
        },
      },
      fontFamily: {
        display: ['Cormorant Garamond', 'serif'],
        body: ['Manrope', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      spacing: {
        section: '6rem',
      },
      borderRadius: {
        DEFAULT: '0.75rem',
      },
    },
  },
  plugins: [],
}
