# 🔗 How to Connect Vercel Frontend to Railway Backend

## Step 1: Find Your Vercel URL(s)

### Option A: From Vercel Dashboard (Easiest)

1. Go to https://vercel.com and log in
2. Click on your project: **"caria"**
3. You'll see your deployment - look for **"Domains"** section
4. You'll see URLs like:
   - `caria-git-main-tomas-projects-70a0592d.vercel.app`
   - `caria-foaxuhi8z-tomas-projects-70a0592d.vercel.app`
   - Or a custom domain if you have one

### Option B: From Deployment Details

1. In Vercel Dashboard → Your Project
2. Click on any deployment
3. Look at the right panel under **"Domains"**
4. Copy the URL(s) shown there

### Option C: From Your Browser

1. Open your deployed Vercel app in browser
2. Look at the address bar - that's your Vercel URL!
3. Example: `https://caria-git-main-tomas-projects-70a0592d.vercel.app`

## Step 2: Set CORS_ORIGINS in Railway

### Method 1: Via Railway Dashboard (Recommended)

1. **Go to Railway Dashboard**: https://railway.app
2. **Click on your service** (the one that's crashing - probably named "caria" or "web")
3. **Click "Variables" tab** (top navigation, next to "Deployments")
4. **Click "+ New Variable"** button
5. **Add the variable:**
   - **Key**: `CORS_ORIGINS`
   - **Value**: Your Vercel URL(s), separated by commas
   
   **Example values:**
   ```
   https://caria-git-main-tomas-projects-70a0592d.vercel.app,https://caria-foaxuhi8z-tomas-projects-70a0592d.vercel.app
   ```
   
   **Or if you have a custom domain:**
   ```
   https://caria.vercel.app,https://www.caria.com
   ```
   
   **Or for development (include localhost too):**
   ```
   https://caria-git-main-tomas-projects-70a0592d.vercel.app,http://localhost:3000,http://localhost:5173
   ```

6. **Click "Add"** or **"Save"**
7. Railway will **automatically redeploy** with the new variable

### Method 2: Via Railway CLI (Advanced)

If you have Railway CLI installed:
```bash
railway variables set CORS_ORIGINS="https://caria-git-main-tomas-projects-70a0592d.vercel.app"
```

## Step 3: Set VITE_API_URL in Vercel

Your frontend also needs to know where your backend is!

1. **Go to Vercel Dashboard** → Your Project
2. **Click "Settings"** (top navigation)
3. **Click "Environment Variables"** (left sidebar)
4. **Click "Add New"**
5. **Add the variable:**
   - **Key**: `VITE_API_URL`
   - **Value**: Your Railway backend URL
   
   **How to find Railway backend URL:**
   - Go to Railway Dashboard → Your Service
   - Look for the URL under the service name
   - It will be like: `https://caria-production.up.railway.app`
   - Or click "Settings" → "Networking" to see the public URL
   
6. **Select environments**: Check "Production", "Preview", "Development"
7. **Click "Save"**
8. **Redeploy your Vercel app**:
   - Go to "Deployments" tab
   - Click the three dots (⋯) on latest deployment
   - Click "Redeploy"

## Step 4: Verify It Works

### Check Railway Logs
1. Railway Dashboard → Your Service → "Deployments"
2. Click on the latest deployment
3. Check "Deploy Logs"
4. Should see: `CORS configured with origins: ['https://caria-git-main-...']`

### Test the Connection
1. Open your Vercel app in browser
2. Open browser DevTools (F12) → "Network" tab
3. Try to login or make an API call
4. Check if requests go to Railway backend
5. Check for CORS errors in console

## Common Issues

### Issue: "CORS policy: No 'Access-Control-Allow-Origin' header"
**Solution**: 
- Make sure `CORS_ORIGINS` in Railway includes your exact Vercel URL
- Include `https://` (not `http://`)
- No trailing slashes
- Case-sensitive - must match exactly

### Issue: "Network Error" or "Failed to fetch"
**Solution**:
- Check `VITE_API_URL` is set in Vercel
- Verify Railway backend is running (check Railway dashboard)
- Check Railway backend URL is correct

### Issue: Railway service still crashes
**Solution**:
- Check Railway "Deploy Logs" for errors
- Make sure all required environment variables are set
- See `RAILWAY_CRASH_TROUBLESHOOTING.md` for more help

## Quick Checklist

- [ ] Found Vercel URL(s) from Vercel dashboard
- [ ] Set `CORS_ORIGINS` in Railway with Vercel URL(s)
- [ ] Found Railway backend URL
- [ ] Set `VITE_API_URL` in Vercel with Railway URL
- [ ] Redeployed Vercel app (to pick up new env var)
- [ ] Railway service is running (not crashed)
- [ ] Tested connection - no CORS errors

## Example Configuration

**Railway Variables:**
```
CORS_ORIGINS=https://caria-git-main-tomas-projects-70a0592d.vercel.app
POSTGRES_HOST=<railway-auto-provided>
POSTGRES_PORT=5432
POSTGRES_USER=<railway-auto-provided>
POSTGRES_PASSWORD=<railway-auto-provided>
POSTGRES_DB=railway
GEMINI_API_KEY=your-key-here
FMP_API_KEY=79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq
FRED_API_KEY=4b90ca15ff28cfec137179c22fd8246d
```

**Vercel Environment Variables:**
```
VITE_API_URL=https://caria-production.up.railway.app
```

---

**Need more help?** Check the Railway "Deploy Logs" and share the error message!

