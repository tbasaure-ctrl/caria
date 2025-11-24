## Vercel Deployment

The Caria app frontend is deployed on Vercel. 

### Latest Deployment Status

The application has been updated with:
- Tailwind CSS v4 PostCSS plugin (`@tailwindcss/postcss`)
- Enhanced UI components (Monte Carlo, Watchlist, Chat Debug Panel)
- RAG embeddings integration (backend)

### Auto-Deployment

Vercel automatically deploys from the `main` branch on push. The latest commit should include all fixes.

### Manual Deployment

If automatic deployment fails or doesn't pick up the latest commit:

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Select the Caria project
3. Click "Deployments"
4. Click "Redeploy" on the latest commit (dd8b423 or newer)

### Known Issues & Fixes

**Tailwind PostCSS Error:**
- ✅ Fixed in commit dd8b423
- Made sure `@tailwindcss/postcss` is in `devDependencies`
- Updated `postcss.config.js` to use `'@tailwindcss/postcss'`

If deployment still fails, verify:
- Latest commit is being built
- `package-lock.json` is committed
- Build command is `vite build`
- Node version is 18+ in Vercel settings

### Environment Variables

Make sure these are set in Vercel:
- `VITE_API_URL` - Backend API URL
- `VITE_GEMINI_API_KEY` - Gemini API key (if using)

### Build Configuration

- **Framework:** Vite
- **Build Command:** `vite build`
- **Output Directory:** `dist`
- **Install Command:** `npm install`
- **Node Version:** 18.x or higher
