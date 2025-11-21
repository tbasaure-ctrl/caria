# 🔧 Fix Render Deployment Issue

## The Problem
Render is looking for `Dockerfile` in the repository root, but can't find it.

## Solution: Configure Render Build Settings

### Option 1: Set Root Directory in Render (Recommended)

1. Go to Render Dashboard → Your Service → Settings
2. Scroll to **Build & Deploy**
3. Set **Root Directory** to: `notebooks`
4. Save and redeploy

### Option 2: Move Dockerfile to Repository Root

If your repository root is NOT `notebooks/`, you need to:

1. Check where Render thinks the root is
2. Either:
   - Move `Dockerfile` to that location, OR
   - Update Dockerfile paths to match the actual root

### Option 3: Use render.yaml Configuration

Create `render.yaml` in your repository root:

```yaml
services:
  - type: web
    name: caria-api
    env: docker
    dockerfilePath: ./notebooks/Dockerfile
    dockerContext: ./notebooks
    envVars:
      - key: POSTGRES_HOST
        sync: false
      - key: POSTGRES_PORT
        value: 5432
      - key: CORS_ORIGINS
        value: https://tu-proyecto.vercel.app
      - key: GEMINI_API_KEY
        sync: false
      - key: FMP_API_KEY
        value: your-fmp-api-key-here
      - key: FRED_API_KEY
        value: your-fred-api-key
```

## Why Railway is Better

If you're willing to pay, **Railway is much easier**:
- ✅ Automatic Dockerfile detection
- ✅ Better error messages
- ✅ Easier PostgreSQL setup
- ✅ More reliable deployments
- ✅ Better pricing

See `RAILWAY_SETUP.md` for Railway setup.


