# 🔍 Railway Deployment Crashes - Troubleshooting Guide

## The Problem
Railway shows "Deployment Successful" but then the service crashes after a few seconds.

## Common Causes & Solutions

### 1. Missing Environment Variables ⚠️ (Most Common)

**Symptoms:**
- Build succeeds
- Service starts but crashes immediately
- Logs show `KeyError` or `NoneType` errors

**Solution:**
Go to Railway → Your Service → **Variables** tab and add:

```bash
# Database (Railway auto-provides these - verify they exist!)
POSTGRES_HOST=<should be auto-set by Railway>
POSTGRES_PORT=5432
POSTGRES_USER=<should be auto-set by Railway>
POSTGRES_PASSWORD=<should be auto-set by Railway>
POSTGRES_DB=railway

# CORS - Your Vercel frontend URL
CORS_ORIGINS=https://tu-proyecto.vercel.app

# API Keys (REQUIRED)
GEMINI_API_KEY=tu-gemini-key-here
FMP_API_KEY=your-fmp-api-key-here
FRED_API_KEY=your-fred-api-key

# Optional but recommended
JWT_SECRET_KEY=tu-secret-key-min-32-chars-long
CARIA_SETTINGS_PATH=/app/caria_data/configs/base.yaml
PYTHONUNBUFFERED=1
```

**How to check:**
1. Railway Dashboard → Your Service → **Variables**
2. Make sure all required variables are set
3. Check if database variables are auto-linked (they should have a database icon)

### 2. Database Connection Failed

**Symptoms:**
- Logs show `psycopg2.OperationalError` or `connection refused`
- Service crashes when trying to connect to PostgreSQL

**Solution:**
1. **Verify PostgreSQL is running:**
   - Railway Dashboard → Your Project
   - Check if PostgreSQL service exists and is running
   - If not, add it: "+ New" → "Database" → "Add PostgreSQL"

2. **Link Database to Service:**
   - Click on your web service
   - Go to **Variables** tab
   - Railway should auto-link database variables
   - If not, manually add them (Railway provides them in database settings)

3. **Check Connection String:**
   - Database variables should be automatically set
   - Format: `POSTGRES_HOST=containers-us-west-xxx.railway.app`

### 3. Import Errors (ModuleNotFoundError)

**Symptoms:**
- Logs show `ModuleNotFoundError: No module named 'caria'`
- Build succeeds but runtime fails

**Solution:**
The Dockerfile should have:
```dockerfile
ENV PYTHONPATH=/app/caria_data/src:/app/services:$PYTHONPATH
```

**Verify:**
1. Check Railway build logs - look for the verification step
2. Should see: `PYTHONPATH: /app/caria_data/src:/app/services`
3. Should see directory listings for `caria/models/` and `api/`

### 4. Port Binding Issues

**Symptoms:**
- Service crashes immediately
- Logs show "Address already in use" or port errors

**Solution:**
Railway automatically sets `PORT` environment variable. Make sure your app uses it:

The Dockerfile CMD should be:
```dockerfile
CMD ["uvicorn", "api.app:socketio_app", "--host", "0.0.0.0", "--port", "8000"]
```

**For Railway, you might need to use the PORT env var:**
```dockerfile
CMD ["sh", "-c", "uvicorn api.app:socketio_app --host 0.0.0.0 --port ${PORT:-8000}"]
```

### 5. Health Check Failing

**Symptoms:**
- Service starts but Railway marks it as "Crashed"
- Health check endpoint not responding

**Solution:**
1. Verify health endpoint exists: `/health/live`
2. Check health check in Dockerfile:
   ```dockerfile
   HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
       CMD curl -f http://localhost:${PORT:-8000}/health/live || exit 1
   ```

## 🔍 How to Debug

### Step 1: Check Railway Logs
1. Railway Dashboard → Your Service
2. Click **"Deployments"** tab
3. Click on the failed deployment
4. Click **"View Logs"** or **"Build Logs"** / **"Deploy Logs"**
5. Look for error messages at the end

### Step 2: Check Build Logs
- Look for: `PYTHONPATH: ...`
- Look for: Directory verification messages
- Look for: Any warnings about missing files

### Step 3: Check Deploy Logs
- Look for: Import errors
- Look for: Database connection errors
- Look for: Missing environment variable errors
- Look for: Port binding errors

### Step 4: Test Locally
```bash
# Build the Docker image
docker build -t caria-test .

# Run with environment variables
docker run -p 8000:8000 \
  -e POSTGRES_HOST=localhost \
  -e POSTGRES_PORT=5432 \
  -e POSTGRES_USER=test \
  -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=test \
  -e CORS_ORIGINS=http://localhost:3000 \
  caria-test
```

## 🚀 Quick Fix Checklist

- [ ] All environment variables are set in Railway
- [ ] PostgreSQL database is running and linked
- [ ] Database variables are auto-linked (have database icon)
- [ ] CORS_ORIGINS includes your frontend URL
- [ ] GEMINI_API_KEY is set (if using Gemini)
- [ ] Build logs show PYTHONPATH is set correctly
- [ ] Build logs show directories exist
- [ ] Deploy logs don't show import errors
- [ ] Health check endpoint is accessible

## 📝 Most Likely Issue

**90% of the time, it's missing environment variables.**

Go to Railway → Your Service → **Variables** and make sure:
1. Database variables are linked (auto-provided by Railway)
2. `CORS_ORIGINS` is set to your Vercel URL
3. All API keys are set

After adding variables, Railway will automatically redeploy.

