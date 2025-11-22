# 🔗 Fix: Connecting GitHub Repo to Railway

## Current Setup ✅
- ✅ Railway CLI installed and logged in (`tbasaure@icloud.com`)
- ✅ `railway.json` configured correctly (points to `backend/Dockerfile`)
- ✅ Git repo: `tbasaure-ctrl/caria`
- ✅ Dockerfile exists at `backend/Dockerfile`

## The Problem
Railway project exists but GitHub repo isn't connected for automatic deployments.

## Solution: Connect GitHub Repo to Railway

### Option 1: Via Railway Dashboard (Recommended - Easiest)

1. **Go to Railway Dashboard**
   - Open https://railway.app
   - Log in if needed

2. **Create New Project from GitHub** (or connect existing)
   - Click **"New Project"** (or open your existing project)
   - Select **"Deploy from GitHub repo"**
   - Authorize Railway to access your GitHub if prompted
   - Select repository: `tbasaure-ctrl/caria`
   - Select branch: `main` (or your deployment branch)

3. **Configure Root Directory**
   - After Railway creates the service, click on your **service name**
   - Go to **Settings** tab
   - Scroll to **"Build & Deploy"** section
   - Find **"Root Directory"** field
   - **Leave it EMPTY** (or set to `.`) - Your repo root IS the project root
   - Click **"Save"**

4. **Verify railway.json is detected**
   - Railway should automatically detect `railway.json` in the root
   - It will use `backend/Dockerfile` as specified in `railway.json`

5. **Set Environment Variables**
   - Go to **Variables** tab
   - Add all required environment variables (see below)

6. **Deploy**
   - Railway will automatically start building
   - Or click **"Deploy"** button
   - Watch the **Logs** tab for progress

### Option 2: Via Railway CLI (If you have project ID)

If you already have a Railway project and know its ID:

```bash
# Link to existing project
railway link <project-id>

# Or create new project and link
railway init
railway link
```

Then connect GitHub via dashboard (Railway CLI doesn't support GitHub connection directly).

### Option 3: Manual GitHub Integration

If Railway dashboard doesn't show your repo:

1. **Check GitHub Permissions**
   - Railway Dashboard → Settings → GitHub
   - Ensure Railway has access to `tbasaure-ctrl/caria`
   - Re-authorize if needed

2. **Manual Deploy via CLI**
   ```bash
   railway up
   ```
   This will deploy current directory to Railway (but won't auto-deploy on git push)

## Required Environment Variables

Add these in Railway → Your Service → Variables:

```bash
# Database (if using Railway PostgreSQL)
POSTGRES_HOST=<railway-auto-provided>
POSTGRES_PORT=5432
POSTGRES_USER=<railway-auto-provided>
POSTGRES_PASSWORD=<railway-auto-provided>
POSTGRES_DB=railway

# OR if using Neon PostgreSQL
DATABASE_URL=postgresql://user:password@host.neon.tech/dbname?sslmode=require

# CORS - Add your Vercel frontend URL(s)
CORS_ORIGINS=https://your-frontend.vercel.app,https://*.vercel.app

# API Keys
FMP_API_KEY=your-fmp-api-key
LLAMA_API_KEY=your-groq-api-key
LLAMA_API_URL=https://api.groq.com/openai/v1/chat/completions
LLAMA_MODEL=llama-3.1-8b-instruct

# Authentication
JWT_SECRET_KEY=<generate-with: python -c "import secrets; print(secrets.token_urlsafe(32))">

# Server
PORT=8080
PYTHONUNBUFFERED=1
```

## Verify Connection

After connecting:

1. **Check Railway Dashboard**
   - Go to your service → **Settings** → **Source**
   - Should show: `tbasaure-ctrl/caria` connected
   - Branch should be set (usually `main`)

2. **Test Auto-Deploy**
   - Make a small change to your repo
   - Push to GitHub: `git push origin main`
   - Railway should automatically detect and deploy

3. **Check Deployment Logs**
   - Railway Dashboard → Your Service → **Deployments**
   - Should show new deployment triggered by GitHub push

## Troubleshooting

### Issue: "Repository not found" in Railway
- **Fix**: Re-authorize Railway's GitHub access
- Railway Dashboard → Settings → GitHub → Reconnect

### Issue: Railway doesn't detect Dockerfile
- **Fix**: Verify Root Directory is EMPTY (not `backend` or `notebooks`)
- Railway uses `railway.json` which points to `backend/Dockerfile`

### Issue: Build fails
- **Fix**: Check Railway logs for specific error
- Common issues:
  - Missing environment variables
  - Dockerfile path incorrect
  - Build timeout (increase in Railway settings)

### Issue: Auto-deploy not working
- **Fix**: 
  1. Verify GitHub webhook is set up (Railway Dashboard → Settings → Source)
  2. Check GitHub repo → Settings → Webhooks (should show Railway webhook)
  3. Test webhook delivery in GitHub

## Quick Commands Reference

```bash
# Check Railway status
railway status

# View logs
railway logs

# Deploy manually
railway up

# Open Railway dashboard
railway open

# Check who you're logged in as
railway whoami
```

## Next Steps

After connecting:
1. ✅ GitHub repo connected to Railway
2. ✅ Environment variables set
3. ✅ First deployment successful
4. ✅ Auto-deploy working (test with a git push)
5. ✅ Get Railway URL and update Vercel `VITE_API_URL`

## Need Help?

If still having issues:
1. Check Railway logs: `railway logs` or Railway Dashboard → Logs
2. Verify git remote: `git remote -v` (should show GitHub repo)
3. Check Railway project settings match this guide
4. Share specific error message from Railway logs
