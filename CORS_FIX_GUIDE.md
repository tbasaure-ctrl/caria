# CORS Fix Guide

## Current Issue

The frontend at `https://caria-way.com` cannot connect to the Railway backend at `https://caria-production.up.railway.app` due to CORS errors.

## Root Cause

The preflight OPTIONS request is failing because:
1. Railway backend may not have `CORS_ORIGINS` environment variable set
2. The regex pattern needs to explicitly match `caria-way.com`

## Solution

### Step 1: Set Railway Environment Variable

In Railway Dashboard:
1. Go to your backend service
2. Navigate to **Variables** tab
3. Add/Update `CORS_ORIGINS`:
   ```
   https://caria-way.com,http://localhost:3000,http://localhost:5173
   ```
4. Save and redeploy

### Step 2: Verify Backend CORS Configuration

The backend code has been updated to:
- Include `caria-way.com` in the regex pattern
- Properly match origins (case-insensitive)

### Step 3: Test CORS

After redeploying Railway backend, test:

```bash
# Test preflight request
curl -X OPTIONS https://caria-production.up.railway.app/api/auth/login \
  -H "Origin: https://caria-way.com" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -v

# Should return:
# Access-Control-Allow-Origin: https://caria-way.com
# Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD
```

### Step 4: Check Railway Logs

After setting the environment variable and redeploying:
1. Check Railway logs for CORS configuration messages
2. Look for: `CORS configured with origins: ['https://caria-way.com', ...]`
3. Test a login request from the frontend

## Additional Issues to Fix

### 1. Firebase API Key Suspended

The Firebase API key has been suspended. You need to:
1. Go to Firebase Console
2. Check API restrictions
3. Create a new API key or remove restrictions
4. Update `frontend/caria-app/src/firebase/config.ts` with new key

### 2. CSS MIME Type Error

The `index.css` file is returning HTML instead of CSS. This suggests:
- Vercel routing issue
- Build output missing CSS file

**Fix:**
1. Check Vercel build logs
2. Verify `vite.config.ts` build output
3. Ensure CSS is being generated in build

### 3. Tailwind CDN Warning

Remove Tailwind CDN from `index.html` (already done in code) and ensure Tailwind is installed as a PostCSS plugin.

## Quick Fix Commands

```bash
# Railway CLI - Set CORS_ORIGINS
railway variables set CORS_ORIGINS="https://caria-way.com,http://localhost:3000,http://localhost:5173"

# Redeploy Railway
railway up
```

## Verification

After fixes:
1. ✅ CORS preflight requests succeed
2. ✅ Login requests work from `https://caria-way.com`
3. ✅ No CORS errors in browser console
4. ✅ CSS files load correctly
5. ✅ Firebase API key is valid
