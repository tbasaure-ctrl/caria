# 🔑 Llama/Groq API Keys Setup Guide

## ⚠️ Important Security Note

**DO NOT expose API keys in the frontend!** API keys should **ONLY** be set in Railway (backend). The frontend communicates with your backend, which securely handles all LLM API calls.

## Architecture Overview

```
Frontend (Vercel) → Backend (Railway) → Groq API
                    ↑
              API Keys stored here
```

- **Frontend**: Only needs `VITE_API_URL` (points to Railway backend)
- **Backend**: Stores all API keys securely in Railway environment variables
- **Backend**: Makes LLM calls to Groq using the stored keys

---

## Step 1: Get Your Groq API Key

1. **Sign up for Groq**
   - Go to https://console.groq.com
   - Sign up or log in with your account

2. **Create API Key**
   - Go to **API Keys** section
   - Click **"Create API Key"**
   - Copy the key (starts with `gsk_...` or `sk_...`)
   - ⚠️ **Save it securely** - you won't be able to see it again!

3. **Check Rate Limits**
   - Free tier: Usually 30 requests/minute
   - Check your limits in Groq dashboard

---

## Step 2: Set Keys in Railway (Backend)

### Via Railway Dashboard (Recommended)

1. **Go to Railway Dashboard**
   - Open https://railway.app
   - Select your project → Your service

2. **Add Environment Variables**
   - Click **"Variables"** tab
   - Click **"+ New Variable"** for each:

   ```bash
   # Required - Your Groq API Key
   LLAMA_API_KEY=gsk_your_actual_groq_api_key_here
   
   # Required - Groq API Endpoint
   LLAMA_API_URL=https://api.groq.com/openai/v1/chat/completions
   
   # Required - Model to use
   LLAMA_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
   
   # Optional - Fallback model if primary fails
   LLAMA_FALLBACK_MODEL=llama-3.1-70b-instruct
   ```

3. **Save and Redeploy**
   - Railway will automatically redeploy after adding variables
   - Or manually trigger: **"Deploy"** → **"Redeploy"**

### Via Railway CLI

If you have Railway CLI set up:

```bash
# Set the API key
railway variables set LLAMA_API_KEY=gsk_your_actual_groq_api_key_here

# Set the API URL
railway variables set LLAMA_API_URL=https://api.groq.com/openai/v1/chat/completions

# Set the model
railway variables set LLAMA_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# Optional fallback
railway variables set LLAMA_FALLBACK_MODEL=llama-3.1-70b-instruct
```

---

## Step 3: Configure Frontend (Vercel)

The frontend **does NOT** need the API keys. It only needs to know where your backend is:

### Set VITE_API_URL in Vercel

1. **Go to Vercel Dashboard**
   - Open https://vercel.com/dashboard
   - Select your frontend project

2. **Add Environment Variable**
   - Go to **Settings** → **Environment Variables**
   - Add:
     ```
     Key: VITE_API_URL
     Value: https://caria-production.up.railway.app
     Environments: Production, Preview, Development
     ```

3. **Redeploy Frontend**
   - Vercel will automatically redeploy
   - Or manually: **Deployments** → **Redeploy**

---

## Step 4: Verify Setup

### Test Backend Connection

```bash
# Check if backend is running
curl https://caria-production.up.railway.app/health

# Test LLM endpoint (if you have auth)
curl -X POST https://caria-production.up.railway.app/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"message": "Hello"}'
```

### Test Groq API Directly (Optional)

```bash
# Replace YOUR_KEY with your actual Groq API key
curl https://api.groq.com/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GROQ_API_KEY" \
  -d '{
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "messages": [
      {"role": "system", "content": "You are Caria, an investment mentor."},
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.5
  }'
```

### Check Railway Logs

1. Railway Dashboard → Your Service → **Logs**
2. Look for:
   - ✅ `Groq response received` - Success!
   - ❌ `LLAMA_API_KEY not configured` - Key missing
   - ❌ `Groq call failed` - Check API key or rate limits

---

## Available Groq Models

You can use these models (update `LLAMA_MODEL` in Railway):

```bash
# Latest (recommended)
meta-llama/llama-4-scout-17b-16e-instruct

# Alternatives
llama-3.1-70b-instruct
llama-3.1-8b-instruct
llama-3.3-70b-instruct
mixtral-8x7b-32768
```

---

## Troubleshooting

### Issue: "LLAMA_API_KEY not configured"
- **Fix**: Add `LLAMA_API_KEY` in Railway Variables
- **Verify**: Railway Dashboard → Variables → Check key exists

### Issue: "Groq call failed"
- **Check**: API key is correct (starts with `gsk_` or `sk_`)
- **Check**: Rate limits not exceeded (Groq dashboard)
- **Check**: Model name is correct
- **Check**: Railway logs for specific error

### Issue: Frontend can't connect to backend
- **Fix**: Set `VITE_API_URL` in Vercel
- **Verify**: Value points to Railway backend URL
- **Check**: CORS_ORIGINS in Railway includes Vercel URL

### Issue: API key exposed in frontend code
- **Fix**: Remove any API keys from frontend code
- **Fix**: Only use `VITE_API_URL` in frontend
- **Security**: API keys should ONLY be in Railway (backend)

---

## Environment Variables Summary

### Railway (Backend) - Required
```bash
LLAMA_API_KEY=gsk_your_groq_api_key
LLAMA_API_URL=https://api.groq.com/openai/v1/chat/completions
LLAMA_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
LLAMA_FALLBACK_MODEL=llama-3.1-70b-instruct  # Optional
```

### Vercel (Frontend) - Required
```bash
VITE_API_URL=https://caria-production.up.railway.app
```

### Railway (Backend) - Other Required Variables
```bash
# Database
DATABASE_URL=postgresql://...  # or POSTGRES_* variables

# Authentication
JWT_SECRET_KEY=your_jwt_secret_key

# CORS
CORS_ORIGINS=https://your-frontend.vercel.app

# Market Data (if needed)
FMP_API_KEY=your_fmp_key
```

---

## Security Best Practices

1. ✅ **Never commit API keys to git**
2. ✅ **Never expose API keys in frontend code**
3. ✅ **Use environment variables in Railway**
4. ✅ **Rotate keys periodically**
5. ✅ **Use different keys for dev/prod if possible**
6. ✅ **Monitor API usage in Groq dashboard**

---

## Cost Estimate

- **Groq**: Free tier available (30 req/min)
- **Railway**: $5-20/month (backend hosting)
- **Vercel**: Free tier available (frontend hosting)
- **Total**: ~$5-20/month for most use cases

---

## Next Steps

1. ✅ Get Groq API key
2. ✅ Set keys in Railway Variables
3. ✅ Set `VITE_API_URL` in Vercel
4. ✅ Test backend connection
5. ✅ Test chat/LLM functionality
6. ✅ Monitor Railway logs

---

## Need Help?

- **Groq Docs**: https://console.groq.com/docs
- **Railway Docs**: https://docs.railway.app
- **Check Railway Logs**: Railway Dashboard → Logs
- **Check Groq Usage**: Groq Dashboard → Usage
