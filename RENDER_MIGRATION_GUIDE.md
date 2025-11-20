# Render Migration Guide

This guide will help you migrate the Caria backend from Google Cloud Run to Render.

## Prerequisites

1. A Render account (sign up at https://render.com)
2. GitHub repository access
3. API keys ready:
   - Llama API Key (Groq): `gsk_****************************************************`
   - FMP API Key
   - Reddit Client ID and Secret
   - JWT Secret Key (generate a new secure one)

## Step 1: Create PostgreSQL Database on Render

1. Log in to Render dashboard: https://dashboard.render.com
2. Click **"New +"** → **"PostgreSQL"**
3. Configure:
   - **Name**: `caria-db`
   - **Database**: `caria`
   - **User**: `caria_user`
   - **Region**: Oregon (or closest to your users)
   - **Plan**: Starter (free tier available)
4. Click **"Create Database"**
5. Wait for database to be provisioned (2-3 minutes)
6. **Important**: Copy the **Internal Database URL** - you'll need it later

## Step 2: Deploy Web Service

1. In Render dashboard, click **"New +"** → **"Web Service"**
2. Connect your GitHub repository:
   - Click **"Connect account"** if not already connected
   - Select your repository: `tbasaure-ctrl/caria` (or your repo name)
   - Click **"Connect"**
3. Configure the service:
   - **Name**: `caria-api`
   - **Region**: Oregon (same as database)
   - **Branch**: `main`
   - **Root Directory**: Leave empty (or `notebooks` if your repo structure requires it)
   - **Environment**: **Docker**
   - **Dockerfile Path**: `backend/Dockerfile`
   - **Docker Context**: `.` (or `notebooks` if needed)
4. Click **"Advanced"** and configure:
   - **Build Command**: Leave empty (Dockerfile handles this)
   - **Start Command**: `/app/backend/start.sh`
   - **Health Check Path**: `/health`

## Step 3: Configure Environment Variables

In the Render service dashboard, go to **"Environment"** tab and add:

### Database Variables (Auto-provided by Render)
- `DATABASE_URL` - **Auto-provided** by Render when you link the database
- `POSTGRES_PASSWORD` - **Auto-provided** by Render

### API Keys
- `LLAMA_API_KEY` = `gsk_****************************************************`
- `LLAMA_API_URL` = `https://api.groq.com/openai/v1/chat/completions` (optional, has default)
- `LLAMA_MODEL` = `llama-3.1-8b-instruct` (optional, has default)
- `FMP_API_KEY` = `<your-fmp-api-key>`
- `REDDIT_CLIENT_ID` = `<your-reddit-client-id>`
- `REDDIT_CLIENT_SECRET` = `<your-reddit-client-secret>`
- `REDDIT_USER_AGENT` = `Caria-Investment-App-v1.0`

### Security
- `JWT_SECRET_KEY` = `<generate-new-secure-key>` (use: `python -c "import secrets; print(secrets.token_urlsafe(32))"`)

### Application Config
- `CORS_ORIGINS` = `https://caria-way.com;https://caria-git-main-tomas-projects-70a0592d.vercel.app`
- `RETRIEVAL_PROVIDER` = `llama`
- `RETRIEVAL_EMBEDDING_DIM` = `768`

## Step 4: Link Database to Web Service

1. In your web service dashboard, go to **"Environment"** tab
2. Scroll to **"Linked Resources"** section
3. Click **"Link Resource"**
4. Select your `caria-db` PostgreSQL database
5. Render will automatically add `DATABASE_URL` environment variable

## Step 5: Deploy

1. Click **"Create Web Service"** or **"Save Changes"**
2. Render will:
   - Build the Docker image
   - Deploy the container
   - Start the service
3. Monitor the build logs in the **"Logs"** tab
4. Wait for deployment to complete (5-10 minutes)

## Step 6: Run Database Migrations

After the service is deployed:

1. Get your service URL (e.g., `https://caria-api.onrender.com`)
2. Test health endpoint: `curl https://caria-api.onrender.com/health`
3. Run database migrations:
   - Option A: Use Render Shell
     - Go to service dashboard → **"Shell"** tab
     - Run: `python backend/api/db_bootstrap.py`
   - Option B: Connect locally
     ```bash
     # Get connection string from Render dashboard
     psql <DATABASE_URL>
     # Then run migrations manually
     ```

## Step 6.5: RAG (Optional - Not Required)

**RAG (Retrieval Augmented Generation) is OPTIONAL**. The API works perfectly without it.

### Current Status
- RAG requires `pgvector` extension in PostgreSQL
- Render's PostgreSQL **does NOT include pgvector by default**
- **You can ignore RAG** - all main features work without it:
  - ✅ User authentication
  - ✅ Portfolio management
  - ✅ Real-time prices
  - ✅ Valuations (DCF, Multiples, Monte Carlo)
  - ✅ Regime detection
  - ✅ Factor analysis
  - ✅ Thesis Arena (uses Llama directly, no RAG needed)

### If You Want to Enable RAG Later

**Option 1: Use Supabase or Neon (PostgreSQL with pgvector)**
- These services include pgvector pre-installed
- Update `DATABASE_URL` to point to Supabase/Neon instead

**Option 2: Install pgvector in Render PostgreSQL (Advanced)**
- Requires custom PostgreSQL setup (not supported on Render's managed PostgreSQL)
- Not recommended - use Option 1 instead

**Option 3: Disable RAG Completely**
- Just ignore any RAG-related warnings
- The API will show `"rag": "disabled"` in `/health` endpoint
- This is **completely normal and fine**

### Verify RAG Status

Check the health endpoint:
```bash
curl https://caria-api.onrender.com/health
```

You'll see:
```json
{
  "status": "ok",
  "rag": "disabled"  // ← This is normal and OK
}
```

**Bottom line**: RAG never worked in Google Cloud and you don't need it. The app works fine without it.

## Step 7: Update Frontend API URL

1. Go to Vercel dashboard (or your frontend hosting)
2. Update environment variable:
   - `VITE_API_URL` = `https://caria-api.onrender.com`
3. Redeploy frontend

## Step 8: Verify Deployment

Test the following endpoints:

```bash
# Health check
curl https://caria-api.onrender.com/health

# Secrets status (should show llama_api_key: true)
curl https://caria-api.onrender.com/api/debug/secrets-status

# Test authentication
curl -X POST https://caria-api.onrender.com/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123","username":"testuser"}'
```

## Troubleshooting

### Build Fails
- Check Dockerfile path is correct
- Verify all required files are in repository
- Check build logs for specific errors

### Database Connection Issues
- Verify `DATABASE_URL` is set correctly
- Check database is in same region as web service
- Ensure database is "Available" (not paused)

### Service Won't Start
- Check start command: `/app/backend/start.sh`
- Verify PORT environment variable (Render sets this automatically)
- Check logs for Python import errors

### API Keys Not Working
- Verify all environment variables are set
- Check for typos in variable names
- Ensure no extra spaces in values

### CORS Errors
- Update `CORS_ORIGINS` with your frontend URL
- Restart service after changing CORS_ORIGINS

## Cost Considerations

- **Free Tier**: 
  - Web service spins down after 15 minutes of inactivity
  - Database pauses after 90 days of inactivity
- **Starter Plan**: $7/month per service
  - Always-on web service
  - Always-on database

## Next Steps

1. Set up monitoring/alerts in Render dashboard
2. Configure custom domain (optional)
3. Set up automatic backups for database
4. Monitor usage and upgrade plan if needed

## Rollback Plan

If something goes wrong:
1. Keep Google Cloud Run service running until migration is verified
2. Update frontend `VITE_API_URL` back to Cloud Run URL if needed
3. Fix issues and redeploy to Render

## Support

- Render Docs: https://render.com/docs
- Render Support: support@render.com
- Check service logs in Render dashboard for detailed error messages

