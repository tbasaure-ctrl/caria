# Railway Deployment Guide

## Auto-Deployment Setup

Railway automatically deploys when you push to your connected branch. However, if auto-deployment isn't working:

### 1. Check Railway Project Settings

1. Go to your Railway project dashboard
2. Navigate to **Settings** → **Service**
3. Verify:
   - **Source**: Should be connected to your GitHub repo
   - **Branch**: Should match your working branch (e.g., `cursor/configure-railway-deployment-and-api-client-default-e0a6`)
   - **Root Directory**: Should be empty (Railway uses `railway.json` to determine build context)

### 2. Manual Deployment Trigger

If auto-deployment isn't working, you can manually trigger a deployment:

**Option A: Via Railway Dashboard**
1. Go to your Railway project
2. Click on your service
3. Click **"Deploy"** or **"Redeploy"** button
4. Select the commit you want to deploy

**Option B: Via Railway CLI**
```bash
railway up
```

### 3. Verify railway.json Configuration

Your `railway.json` is correctly configured:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "backend/Dockerfile",
    "buildContext": "."
  },
  "deploy": {
    "startCommand": "/app/backend/start.sh",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10,
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100
  }
}
```

This tells Railway to:
- Build from repository root (`.`)
- Use `backend/Dockerfile` for the build
- Start with `/app/backend/start.sh`
- Health check at `/health` endpoint

### 4. Sync Deployments (Frontend + Backend)

To keep all three services (Frontend Vercel, Backend Railway, Database) in sync:

1. **Backend (Railway)**: 
   - Push to your branch → Railway auto-deploys
   - Or manually trigger via dashboard

2. **Frontend (Vercel)**:
   - Push to your branch → Vercel auto-deploys
   - Make sure `VITE_API_URL` env var points to Railway backend URL

3. **Database (Railway/Neon)**:
   - Migrations run automatically on backend startup
   - Or manually via Railway CLI/console

### 5. Environment Variables

Ensure these are set in Railway:

**Required:**
- `CORS_ORIGINS`: `https://caria-way.com,http://localhost:3000,http://localhost:5173`
- `DATABASE_URL`: Your PostgreSQL connection string
- Other backend env vars (see `backend/.env.example`)

### 6. Troubleshooting

**Issue: Railway not auto-deploying**
- Check GitHub integration in Railway settings
- Verify branch name matches
- Check Railway logs for errors

**Issue: Build fails**
- Check Railway build logs
- Verify `backend/Dockerfile` exists
- Verify `backend/start.sh` exists and is executable

**Issue: Service won't start**
- Check Railway logs
- Verify `PORT` environment variable (Railway sets this automatically)
- Check health endpoint: `https://your-service.up.railway.app/health`

### 7. Current Deployment Status

After pushing changes:
- ✅ **Vercel**: Will auto-deploy (fixed TypeScript error)
- ⚠️ **Railway**: Check if auto-deployment is enabled, or manually trigger

### 8. Manual Railway Deployment

If you need to manually deploy the latest commit:

1. **Via Dashboard:**
   - Railway Dashboard → Your Service → Deployments
   - Click "Deploy" → Select latest commit

2. **Via CLI:**
   ```bash
   railway login
   railway link  # Link to your project
   railway up    # Deploy latest commit
   ```

3. **Via GitHub:**
   - Push to main/master branch (if Railway is watching that branch)
   - Or create a new deployment via Railway dashboard

### 9. Verify Deployment

After deployment, verify:

1. **Backend Health:**
   ```bash
   curl https://your-backend.up.railway.app/health
   ```

2. **CORS Configuration:**
   - Test from frontend
   - Check Railway logs for CORS errors

3. **Environment Variables:**
   - Railway Dashboard → Variables
   - Verify all required vars are set
