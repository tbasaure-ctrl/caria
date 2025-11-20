# Neon Migration Guide

This guide will help you migrate the Caria backend to Neon (PostgreSQL with pgvector) and deploy the backend to Render or Railway.

## Why Neon?

- ✅ **pgvector included** - Perfect for RAG functionality
- ✅ **Serverless** - Auto-scaling, pay only for what you use
- ✅ **Fast** - Low latency, global edge network
- ✅ **Free tier** - 3 GB storage, generous limits
- ✅ **Database branching** - Great for development/testing

## Prerequisites

1. A Neon account (sign up at https://neon.tech)
2. A Render or Railway account for backend hosting
3. API keys ready:
   - Llama API Key (Groq): `gsk_****************************************************`
   - FMP API Key
   - Reddit Client ID and Secret
   - JWT Secret Key (generate a new secure one)

## Step 1: Create Neon Database

1. **Sign up/Login**: Go to https://console.neon.tech
2. **Create Project**:
   - Click **"Create Project"**
   - **Project name**: `caria-backend`
   - **Region**: Choose closest to your users (e.g., `us-east-2`)
   - **PostgreSQL version**: `15` (recommended)
   - Click **"Create Project"**
3. **Get Connection String**:
   - After project creation, you'll see the connection string
   - Format: `postgresql://user:password@host.neon.tech/dbname?sslmode=require`
   - **Copy this** - you'll need it for environment variables
4. **Enable pgvector** (if not already enabled):
   - Go to **"SQL Editor"** in Neon dashboard
   - Run: `CREATE EXTENSION IF NOT EXISTS vector;`
   - Verify: `SELECT * FROM pg_extension WHERE extname = 'vector';`

## Step 2: Choose Backend Hosting

You have two options:

### Option A: Render (Recommended for simplicity)

1. **Create Web Service**:
   - Go to https://dashboard.render.com
   - Click **"New +"** → **"Web Service"**
   - Connect GitHub repository
   - Configure:
     - **Name**: `caria-api`
     - **Environment**: **Docker**
     - **Dockerfile Path**: `backend/Dockerfile`
     - **Docker Context**: `.` (or `notebooks` if needed)
     - **Start Command**: `/app/backend/start.sh`
     - **Health Check Path**: `/health`

2. **Set Environment Variables** in Render:
   ```
   DATABASE_URL=<your-neon-connection-string>
   LLAMA_API_KEY=gsk_****************************************************
   LLAMA_API_URL=https://api.groq.com/openai/v1/chat/completions
   LLAMA_MODEL=llama-3.1-8b-instruct
   REDDIT_CLIENT_ID=<your-reddit-client-id>
   REDDIT_CLIENT_SECRET=<your-reddit-client-secret>
   REDDIT_USER_AGENT=Caria-Investment-App-v1.0
   FMP_API_KEY=<your-fmp-api-key>
   JWT_SECRET_KEY=<generate-new-secure-key>
   CORS_ORIGINS=https://caria-way.com;https://caria-git-main-tomas-projects-70a0592d.vercel.app
   RETRIEVAL_PROVIDER=llama
   RETRIEVAL_EMBEDDING_DIM=768
   ```

3. **Deploy**: Click **"Create Web Service"**

### Option B: Railway (Alternative)

1. **Create Project**:
   - Go to https://railway.app
   - Click **"New Project"**
   - Select **"Deploy from GitHub repo"**
   - Choose your repository

2. **Configure Service**:
   - Railway auto-detects Dockerfile
   - Add environment variables (same as Render above)
   - Railway will auto-deploy

## Step 3: Run Database Migrations

After backend is deployed:

### Option A: Using Neon SQL Editor

1. Go to Neon dashboard → **"SQL Editor"**
2. Copy contents of migration files:
   - `caria_data/migrations/init.sql`
   - `caria_data/migrations/012_model_portfolios.sql`
   - `caria_data/migrations/013_fix_missing_columns.sql` (if exists)
3. Run each migration in SQL Editor

### Option B: Using Backend Shell

1. Go to Render/Railway dashboard → **"Shell"** tab
2. Run:
   ```bash
   python backend/api/db_bootstrap.py
   ```

### Option C: Using psql Locally

```bash
# Install psql if needed
# macOS: brew install postgresql
# Linux: sudo apt-get install postgresql-client

# Connect to Neon
psql "<your-neon-connection-string>"

# Run migrations
\i caria_data/migrations/init.sql
\i caria_data/migrations/012_model_portfolios.sql
```

## Step 4: Verify pgvector Installation

In Neon SQL Editor, run:

```sql
-- Check if pgvector is installed
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Should return:
-- extname | extversion
-- --------+------------
-- vector  | 0.5.0

-- Test vector type
SELECT '[1,2,3]'::vector;

-- Should return without error
```

## Step 5: Verify RAG is Working

After deployment, test:

```bash
# Health check (should show rag: available)
curl https://your-backend-url.onrender.com/health

# Expected response:
{
  "status": "ok",
  "database": "available",
  "rag": "available",  // ← Should be "available" now!
  "regime": "available",
  "factors": "available"
}
```

## Step 6: Update Frontend API URL

1. Go to Vercel dashboard
2. Update environment variable:
   - `VITE_API_URL` = `https://your-backend-url.onrender.com`
3. Redeploy frontend

## Step 7: Load Initial Data (Optional)

If you have wisdom chunks/data for RAG:

1. Use Neon SQL Editor or connect via psql
2. Insert data into `wisdom_chunks` table (if you have embeddings)
3. Or use the RAG ingestion endpoints if available

## Neon-Specific Tips

### Connection Pooling

Neon supports connection pooling. Use the pooled connection string:
- Format: `postgresql://user:password@host-pooler.neon.tech/dbname?sslmode=require`
- Better for serverless/serverless-like environments
- Reduces connection overhead

### Database Branching

Neon allows you to create branches (like git branches):
- Create a branch for testing migrations
- Test changes without affecting production
- Merge branches when ready

### Monitoring

- Check **"Metrics"** tab in Neon dashboard
- Monitor query performance
- Check connection usage

## Troubleshooting

### RAG Still Shows "disabled"

1. **Check pgvector is installed**:
   ```sql
   SELECT * FROM pg_extension WHERE extname = 'vector';
   ```

2. **Check backend logs** for RAG initialization errors:
   - Look for "Stack RAG inicializado correctamente" message
   - Check for any exceptions during VectorStore initialization

3. **Verify DATABASE_URL** is correct:
   - Should include `?sslmode=require` for Neon
   - Check connection string format

### Connection Timeouts

- Use **pooled connection string** (ends with `-pooler.neon.tech`)
- Check Neon dashboard for connection limits
- Verify network connectivity

### Migration Errors

- Run migrations one at a time
- Check for existing tables before creating
- Verify user has CREATE/ALTER permissions

## Cost Considerations

### Neon Free Tier
- 3 GB storage
- 10 compute hours/month
- 1 project
- Perfect for development/testing

### Neon Paid Plans
- Starts at $19/month
- More storage and compute
- Better for production

### Render Free Tier
- Spins down after 15 min inactivity
- Good for testing

### Render Paid Plans
- Starts at $7/month
- Always-on service
- Better for production

## Next Steps

1. ✅ Set up Neon database
2. ✅ Deploy backend to Render/Railway
3. ✅ Configure environment variables
4. ✅ Run database migrations
5. ✅ Verify RAG is working
6. ✅ Update frontend API URL
7. ✅ Test all endpoints
8. ✅ Monitor Neon metrics

## Support

- **Neon Docs**: https://neon.tech/docs
- **Neon Discord**: https://discord.gg/neondatabase
- **Render Docs**: https://render.com/docs
- **Railway Docs**: https://docs.railway.app


