# 🚀 Clean Deployment Guide - From Scratch

This guide will help you deploy the entire project correctly from scratch.

## 📋 Pre-Deployment Checklist

### 1. Fix .gitignore (IMPORTANT!)
Make sure large data files are excluded:
- ✅ `*.parquet` 
- ✅ `*.jsonl`
- ✅ `*.csv`
- ✅ `data/` directory

### 2. Clean Up Repository
Remove tracked data files that shouldn't be in Git:
```bash
git rm --cached **/*.parquet
git rm --cached **/*.jsonl
git rm --cached **/*.csv
git rm -r --cached data/
git rm -r --cached caria_data/data/
```

### 3. Verify Project Structure
Your repository should have this structure:
```
notebooks/                    # Root directory (for Railway/Render)
├── Dockerfile                # Backend Dockerfile
├── .gitignore               # Should exclude data files
├── services/                # Backend API
│   ├── api/
│   │   ├── app.py
│   │   └── ...
│   └── requirements.txt
├── caria_data/              # Caria package
│   ├── src/
│   │   └── caria/          # Main package
│   ├── configs/
│   └── requirements.txt
└── caria_data/caria-app/   # Frontend (Vercel)
    ├── package.json
    ├── vercel.json
    └── ...
```

### 4. Backend Deployment (Railway - Recommended)

#### Step 1: Prepare Backend
1. Ensure `Dockerfile` is in `notebooks/` directory
2. Verify `PYTHONPATH` is set correctly in Dockerfile
3. Check all environment variables are documented

#### Step 2: Deploy to Railway
1. Go to https://railway.app
2. New Project → Deploy from GitHub
3. Select repository: `tbasaure-ctrl/caria`
4. **Root Directory Setting**:
   - **IMPORTANT**: Since your Git repository root IS `notebooks/`, you should:
     - **Leave Root Directory EMPTY** (blank) OR set it to `.` (dot)
     - Do NOT set it to `notebooks` - that would look for `notebooks/notebooks/` which doesn't exist!
   - Railway will automatically detect the Dockerfile in the root
5. Add PostgreSQL database
6. Configure environment variables (see below)

#### Step 3: Environment Variables for Railway
```bash
# Database (Railway auto-provides these, verify they're set)
POSTGRES_HOST=<railway-provided>
POSTGRES_PORT=5432
POSTGRES_USER=<railway-provided>
POSTGRES_PASSWORD=<railway-provided>
POSTGRES_DB=railway

# CORS - Your Vercel frontend URL
CORS_ORIGINS=https://tu-proyecto.vercel.app

# API Keys
GEMINI_API_KEY=tu-gemini-key
FMP_API_KEY=your-fmp-api-key-here
FRED_API_KEY=your-fred-api-key

# Optional
JWT_SECRET_KEY=tu-secret-key-min-32-chars
CARIA_SETTINGS_PATH=/app/caria_data/configs/base.yaml
PYTHONUNBUFFERED=1
```

#### Step 4: Enable pgvector
After deployment, connect to Railway PostgreSQL and run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 5. Frontend Deployment (Vercel)

#### Step 1: Prepare Frontend
1. Ensure `caria_data/caria-app/` has all frontend files
2. Verify `vercel.json` is configured
3. Check `package.json` has correct build scripts

#### Step 2: Deploy to Vercel
1. Go to https://vercel.com
2. Import Git Repository
3. **Root Directory**: `caria_data/caria-app`
4. Framework Preset: Vite
5. Build Command: `npm run build`
6. Output Directory: `dist`

#### Step 3: Environment Variables for Vercel
```bash
# Backend API URL (from Railway)
VITE_API_URL=https://tu-proyecto.up.railway.app

# Firebase (if using)
VITE_FIREBASE_API_KEY=...
VITE_FIREBASE_AUTH_DOMAIN=...
# ... other Firebase vars
```

### 6. Connect Frontend to Backend

1. **Update Vercel Environment Variable**:
   - Go to Vercel → Project → Settings → Environment Variables
   - Set `VITE_API_URL` to your Railway backend URL

2. **Update Railway CORS**:
   - Go to Railway → Service → Variables
   - Set `CORS_ORIGINS` to your Vercel URL: `https://tu-proyecto.vercel.app`

3. **Redeploy Both**:
   - Railway will auto-redeploy when you update env vars
   - Vercel: Go to Deployments → Redeploy

## 🔍 Verification Steps

### Backend Health Check
```bash
curl https://tu-proyecto.up.railway.app/health/live
# Should return: {"status":"ok"}
```

### Frontend Connection
1. Open your Vercel app
2. Open browser DevTools → Network tab
3. Try to login or make an API call
4. Check that requests go to Railway backend URL
5. Verify no CORS errors

### Database Connection
1. Check Railway logs for database connection errors
2. Verify PostgreSQL is running in Railway dashboard
3. Test pgvector extension is installed

## 🐛 Common Issues & Solutions

### Issue: ModuleNotFoundError: No module named 'caria'
**Solution**: 
- Verify `PYTHONPATH=/app/caria_data/src:/app/services` in Dockerfile
- Check Root Directory is set to `notebooks` in Railway

### Issue: CORS Errors
**Solution**:
- Add Vercel URL to `CORS_ORIGINS` in Railway
- Include both `https://` and `http://` if needed
- Check backend logs for CORS errors

### Issue: Database Connection Failed
**Solution**:
- Verify PostgreSQL is running in Railway
- Check environment variables are set correctly
- Verify connection string format

### Issue: Frontend Can't Connect to Backend
**Solution**:
- Verify `VITE_API_URL` is set in Vercel
- Check backend is accessible (health check)
- Verify CORS is configured
- Check browser console for errors

## 📝 Deployment Order

1. ✅ Fix .gitignore and commit
2. ✅ Remove tracked data files
3. ✅ Commit all changes
4. ✅ Push to GitHub
5. ✅ Deploy Backend (Railway)
6. ✅ Get backend URL
7. ✅ Deploy Frontend (Vercel)
8. ✅ Connect frontend to backend (env vars)
9. ✅ Test everything
10. ✅ Celebrate! 🎉

## 💰 Cost Estimate

- **Railway**: ~$5-10/month (Hobby plan)
- **Vercel**: Free tier (sufficient for most projects)
- **Total**: ~$5-10/month

## 🎯 Next Steps After Deployment

1. Set up monitoring (Railway has built-in logs)
2. Configure custom domains (optional)
3. Set up CI/CD (automatic deployments)
4. Add error tracking (Sentry, etc.)
5. Set up backups for database

---

**Ready to deploy? Follow the steps above in order!**


