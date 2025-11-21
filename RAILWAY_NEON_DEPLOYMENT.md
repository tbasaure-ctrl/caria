# Caria Deployment Guide: Railway + Neon + Vercel

Complete deployment guide for Caria backend on Railway with Neon PostgreSQL and Vercel frontend.

## Prerequisites

- GitHub repository: `tbasaure-ctrl/caria`
- Railway account (https://railway.app)
- Neon account (https://neon.tech)
- Vercel account (https://vercel.com)
- Groq API key (https://console.groq.com)

## 1. Database Setup (Neon)

### Step 1.1: Create Neon Database

1. Go to https://console.neon.tech
2. Click "Create Project"
3. Name: `caria-production`
4. Region: Choose closest to your Railway region (e.g., `us-west-2`)
5. Click "Create Project"

### Step 1.2: Enable pgvector Extension

1. In Neon dashboard, go to your project
2. Click "SQL Editor"
3. Run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
4. Verify:
   ```sql
   SELECT * FROM pg_extension WHERE extname = 'vector';
   ```

### Step 1.3: Get Connection String

1. In Neon dashboard → Project Settings → Connection Details
2. Copy the **Connection String** (pooler format recommended)
3. Format: `postgresql://user:password@host.neon.tech/dbname?sslmode=require`

## 2. Backend Deployment (Railway)

### Step 2.1: Create Railway Service

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Connect repository: `tbasaure-ctrl/caria`
5. Select branch: `main` (or your deployment branch)

### Step 2.2: Configure Service Settings

1. In Railway project, click on the service
2. Go to **Settings** tab
3. Set:
   - **Root Directory**: `notebooks`
   - **Build Command**: (leave empty, Dockerfile handles it)
   - **Start Command**: (leave empty, Dockerfile handles it)

### Step 2.3: Configure Environment Variables

Go to **Variables** tab and add:

```bash
# Database (Neon)
DATABASE_URL=postgresql://user:password@host.neon.tech/dbname?sslmode=require

# Authentication
JWT_SECRET_KEY=<generate-with-python-secrets-token-urlsafe-32>

# LLM (Groq)
LLAMA_API_KEY=your-groq-api-key-here
LLAMA_API_URL=https://api.groq.com/openai/v1/chat/completions
LLAMA_MODEL=llama-3.1-8b-instruct

# RAG / Embeddings
RETRIEVAL_PROVIDER=local
RETRIEVAL_EMBEDDING_MODEL=nomic-embed-text-v1
RETRIEVAL_EMBEDDING_DIM=768

# Market Data
FMP_API_KEY=your-fmp-api-key-here
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-client-secret

# CORS (semicolon or comma separated)
CORS_ORIGINS=https://caria-way.com;https://caria-git-main-tomas-projects-70a0592d.vercel.app

# Server
PORT=8080

# Python
PYTHONUNBUFFERED=1
PYTHON_VERSION=3.11.14
```

**Important**: Replace `DATABASE_URL` with your actual Neon connection string from Step 1.3.

### Step 2.4: Generate JWT Secret

Run locally:
```python
import secrets
print(secrets.token_urlsafe(32))
```

Copy the output to `JWT_SECRET_KEY` in Railway.

### Step 2.5: Deploy

1. Railway will automatically detect the Dockerfile in `notebooks/backend/Dockerfile`
2. Click "Deploy" or push to GitHub to trigger deployment
3. Wait for build to complete (5-10 minutes)
4. Check **Logs** tab for any errors

### Step 2.6: Get Railway URL

1. After deployment succeeds, go to **Settings** → **Networking**
2. Click "Generate Domain" or use custom domain
3. Copy the URL (e.g., `https://caria-production.up.railway.app`)

## 3. Database Migrations

### Step 3.1: Run Bootstrap Script

The backend automatically runs migrations on startup via `db_bootstrap.py`. Check logs to verify:

```
✓ Vector extension created successfully
✓ Base schema initialized successfully
✓ Default user TBL created successfully
```

### Step 3.2: Verify Database Schema

Connect to Neon database and verify tables exist:

```sql
\dt  -- List tables
SELECT * FROM users LIMIT 1;  -- Verify users table
SELECT * FROM pg_extension WHERE extname = 'vector';  -- Verify pgvector
```

## 4. Frontend Deployment (Vercel)

### Step 4.1: Connect Repository

1. Go to https://vercel.com
2. Click "Add New Project"
3. Import repository: `tbasaure-ctrl/caria`
4. Root Directory: `notebooks/frontend/caria-app`

### Step 4.2: Configure Build Settings

- **Framework Preset**: Vite
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Install Command**: `npm install`

### Step 4.3: Set Environment Variables

Go to **Settings** → **Environment Variables**:

```bash
VITE_API_URL=https://your-railway-url.up.railway.app
```

Replace with your Railway backend URL from Step 2.6.

### Step 4.4: Deploy

1. Click "Deploy"
2. Wait for build to complete
3. Vercel will provide a preview URL

## 5. Verification

### Step 5.1: Backend Health Check

```bash
curl https://your-railway-url.up.railway.app/health
```

Expected response:
```json
{
  "status": "ok",
  "database": "available",
  "auth": "available",
  "rag": "available",
  "regime": "available",
  "factors": "available",
  "valuation": "available"
}
```

### Step 5.2: Test Authentication

```bash
# Register user
curl -X POST https://your-railway-url.up.railway.app/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "TestPassword123!",
    "full_name": "Test User"
  }'

# Login
curl -X POST https://your-railway-url.up.railway.app/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "TestPassword123!"
  }'
```

### Step 5.3: Test Frontend

1. Open your Vercel deployment URL
2. Try logging in with test credentials
3. Verify portfolio, analysis, and community features work

## 6. Troubleshooting

### Database Connection Issues

**Problem**: `connection to server at "localhost" failed`

**Solution**: 
- Verify `DATABASE_URL` in Railway is set correctly
- Ensure Neon connection string includes `?sslmode=require`
- Check Railway logs for connection errors

### RAG Not Available

**Problem**: `"rag": "disabled"` in health check

**Solution**:
- Verify `RETRIEVAL_PROVIDER=local` is set
- Check that pgvector extension is enabled in Neon
- Review logs for embedding model loading errors

### CORS Errors

**Problem**: Frontend can't call backend API

**Solution**:
- Add Vercel URL to `CORS_ORIGINS` in Railway
- Format: `https://your-app.vercel.app` (comma or semicolon separated)
- Redeploy backend after updating CORS_ORIGINS

### LLM Not Working

**Problem**: Analysis endpoints return errors

**Solution**:
- Verify `LLAMA_API_KEY` is set correctly in Railway
- Check Groq API key is valid at https://console.groq.com
- Review logs for LLM API errors

## 7. Monitoring

### Railway Logs

1. Go to Railway dashboard → Your service → **Logs**
2. Monitor for errors, warnings, and startup messages
3. Use filters to search for specific errors

### Neon Monitoring

1. Go to Neon dashboard → Your project → **Metrics**
2. Monitor connection count, query performance
3. Check **Logs** for database errors

### Vercel Analytics

1. Go to Vercel dashboard → Your project → **Analytics**
2. Monitor page views, API calls, errors
3. Check **Functions** tab for serverless function logs

## 8. Updates & Redeployment

### Backend Updates

1. Push changes to GitHub
2. Railway automatically redeploys
3. Monitor logs for deployment status

### Frontend Updates

1. Push changes to GitHub
2. Vercel automatically redeploys
3. Check deployment status in Vercel dashboard

### Database Migrations

Migrations run automatically on backend startup. To run manually:

1. Connect to Neon database
2. Run SQL files from `caria_data/migrations/`
3. Or restart Railway service to trigger bootstrap

## 9. Security Checklist

- [ ] `JWT_SECRET_KEY` is strong and unique
- [ ] `DATABASE_URL` contains strong password
- [ ] API keys are stored in Railway environment variables (not in code)
- [ ] CORS_ORIGINS only includes trusted domains
- [ ] Frontend `VITE_API_URL` points to Railway backend
- [ ] Database has pgvector extension enabled
- [ ] Default admin user password changed (if using)

## 10. Cost Optimization

- **Railway**: Free tier includes $5/month credit
- **Neon**: Free tier includes 0.5GB storage, 1 project
- **Vercel**: Free tier includes unlimited deployments
- **Groq**: Free tier includes generous rate limits

Monitor usage in each platform's dashboard.

## Support

- Railway Docs: https://docs.railway.app
- Neon Docs: https://neon.tech/docs
- Vercel Docs: https://vercel.com/docs
- Groq Docs: https://console.groq.com/docs

