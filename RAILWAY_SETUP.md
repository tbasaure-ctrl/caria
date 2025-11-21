# 🚂 Railway Deployment Guide for Caria Backend

## Why Railway?
- ✅ **Easiest deployment** - Just connect GitHub and deploy
- ✅ **PostgreSQL included** - Managed database with pgvector support
- ✅ **Automatic HTTPS** - No SSL configuration needed
- ✅ **WebSocket support** - Perfect for your Socket.IO chat
- ✅ **Great pricing** - $5-20/month for most apps
- ✅ **Better than Render** - More reliable, easier setup

## Quick Setup (5 minutes)

### Step 1: Create Railway Account
1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"

### Step 2: Connect Repository
1. Select your repository: `tbasaure-ctrl/caria`
2. Railway will detect the Dockerfile automatically and start deploying
3. **Important**: Root Directory Setting:
   - **If your Git repo root IS `notebooks/`** (which it is in your case):
     - **Leave Root Directory EMPTY** or set it to `.` (current directory)
     - Railway will use the repository root automatically
   - **If your Git repo root is the parent of `notebooks/`**:
     - Set Root Directory to `notebooks`
   
   **To check/change Root Directory:**
   - After Railway creates the service, click on your **service name** (e.g., "caria" or "web")
   - Go to the **Settings** tab (top navigation)
   - Scroll down to **"Build & Deploy"** section
   - Find **"Root Directory"** field
   - **Leave it EMPTY** (or set to `.`) since your repo root is already `notebooks/`
   - Click **"Save"** or **"Update"**
   - Railway will automatically redeploy

### Step 3: Add PostgreSQL Database
1. In your Railway project, click "+ New"
2. Select "Database" → "Add PostgreSQL"
3. Railway will create a PostgreSQL instance with connection string

### Step 4: Configure Environment Variables

Add these in Railway → Your Service → Variables:

```bash
# Database (Railway provides these automatically, but verify)
POSTGRES_HOST=<railway-provided>
POSTGRES_PORT=5432
POSTGRES_USER=<railway-provided>
POSTGRES_PASSWORD=<railway-provided>
POSTGRES_DB=railway

# CORS - Add your Vercel frontend URL
CORS_ORIGINS=https://tu-proyecto.vercel.app,https://tu-proyecto.vercel.app

# API Keys
GEMINI_API_KEY=tu-gemini-key
FMP_API_KEY=your-fmp-api-key-here
FRED_API_KEY=your-fred-api-key

# Optional
JWT_SECRET_KEY=tu-secret-key-min-32-chars
CARIA_SETTINGS_PATH=/app/caria_data/configs/base.yaml
```

### Step 5: Deploy
1. Railway will automatically build and deploy
2. Wait for deployment to complete (~3-5 minutes)
3. Get your backend URL: `https://tu-proyecto.up.railway.app`

### Step 6: Update Vercel Frontend
1. Go to Vercel → Your Project → Settings → Environment Variables
2. Add/Update:
   ```
   VITE_API_URL=https://tu-proyecto.up.railway.app
   ```
3. Redeploy frontend

## Enable pgvector Extension

After deployment, connect to your Railway PostgreSQL and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

You can do this via Railway's database dashboard or using psql.

## Cost Estimate
- **Railway Hobby Plan**: $5/month (includes $5 credit)
- **PostgreSQL**: Included in Hobby plan
- **Total**: ~$5-10/month for most use cases

## Troubleshooting

### Build Fails
- Check Root Directory is set to `notebooks/`
- Verify Dockerfile exists in `notebooks/Dockerfile`

### Database Connection Issues
- Verify environment variables are set correctly
- Check Railway database is running

### CORS Errors
- Add your Vercel URL to `CORS_ORIGINS`
- Include both `https://` and `http://` versions if needed

## Next Steps
1. Deploy backend on Railway
2. Update Vercel `VITE_API_URL`
3. Test the connection
4. Enjoy! 🎉

