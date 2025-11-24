# Design Recovery Guide

If you're experiencing styling issues after recent updates, follow these steps:

## Quick Fixes

### 1. Clear Browser Cache & Hard Reload
- **Chrome/Edge:** Ctrl + Shift + R (Windows) or Cmd + Shift + R (Mac)
- **Firefox:** Ctrl + F5
- Or open DevTools (F12) → Right-click refresh button → "Empty Cache and Hard Reload"

### 2. Clear Vite Cache
```bash
cd frontend/caria-app

# Stop the dev server (Ctrl+C)

# Remove Vite cache
rm -rf node_modules/.vite
# Windows:
rmdir /s /q node_modules\.vite

# Restart dev server
npm run dev
```

### 3. Reinstall Dependencies
```bash
cd frontend/caria-app

# Remove node_modules and package-lock
rm -rf node_modules package-lock.json
# Windows:
rmdir /s /q node_modules
del package-lock.json

# Fresh install
npm install

# Start dev server
npm run dev
```

## If Design is Still Broken

### Check These Files:

**1. tailwind.config.js** should have:
```javascript
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
                // ... other colors
            },
            fontFamily: {
                display: ['"Cormorant Garamond"', 'serif'],
                body: ['"Manrope"', 'sans-serif'],
            },
        },
    },
}
```

**2. postcss.config.js** should be:
```javascript
export default {
    plugins: {
        '@tailwindcss/postcss': {},
        autoprefixer: {},
    },
}
```

**3. src/index.css** should have:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  padding: 0;
  font-family: var(--font-body);
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
}
```

### Merge with Main (Recommended)

If you're on the cursor branch and want the working design:

```bash
# Stash any uncommitted changes
git stash

# Switch to main
git checkout main

# Pull latest
git pull origin main

# If you want to keep cursor branch changes:
git checkout cursor/fix-broken-deployment-after-feature-updates-gemini-3-pro-preview-4174
git merge main

# Resolve any conflicts, then:
git stash pop  # if you stashed changes
```

## Nuclear Option: Complete Reset

If nothing else works:

```bash
# Save any important uncommitted work first!
git stash

# Hard reset to main
git checkout main
git reset --hard origin/main

# Clean everything
cd frontend/caria-app
rm -rf node_modules package-lock.json node_modules/.vite

# Fresh install
npm install
npm run dev
```

## Verify It's Working

Open http://localhost:3000 and check:
- [ ] Renaissance fonts (Cormorant Garamond headers)
- [ ] Dark elegant background
- [ ] Proper colors (burgundy, cream, blue accents)
- [ ] "Discover Caria" button visible
- [ ] All sections properly styled

## Still Having Issues?

1. Check browser console for errors (F12 → Console tab)
2. Check terminal for Vite errors
3. Verify all CSS files are being loaded
4. Try incognito/private window to rule out extensions
