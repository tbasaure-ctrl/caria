# Fix Vercel Deployment - Changes Not Appearing

## Problem
Changes are pushed to GitHub and deployed on Railway, but Vercel is not showing the updates.

## Quick Fix Steps

### Step 1: Check Vercel Project Settings

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your project
3. Go to **Settings** → **General**
4. Check **Root Directory** - it should be:
   ```
   frontend/caria-app
   ```
   NOT `caria_data/caria-app` (old path)

### Step 2: Verify Git Integration

1. In Vercel Dashboard → **Settings** → **Git**
2. Verify:
   - ✅ Repository is connected
   - ✅ Production Branch is `main`
   - ✅ Auto-deploy is enabled

### Step 3: Manual Redeploy

**Option A: From Vercel Dashboard**
1. Go to **Deployments** tab
2. Find the latest deployment
3. Click the **⋯** (three dots) menu
4. Select **Redeploy**
5. Wait for build to complete

**Option B: Trigger via Git Push**
```bash
# Make a small change to trigger deployment
echo "# Trigger deployment" >> frontend/caria-app/README.md
git add frontend/caria-app/README.md
git commit -m "chore: trigger Vercel deployment"
git push
```

### Step 4: Check Build Logs

1. Go to **Deployments** tab
2. Click on the latest deployment
3. Check **Build Logs** for errors
4. Common issues:
   - ❌ Build command failing
   - ❌ Missing environment variables
   - ❌ Wrong root directory

### Step 5: Verify Environment Variables

Go to **Settings** → **Environment Variables** and verify:
- `VITE_API_URL` is set (should be your Railway backend URL)
- Variables are enabled for **Production**, **Preview**, and **Development**

## If Root Directory is Wrong

1. Go to **Settings** → **General**
2. Click **Edit** next to **Root Directory**
3. Change to: `frontend/caria-app`
4. Click **Save**
5. Vercel will automatically redeploy

## If Auto-Deploy is Disabled

1. Go to **Settings** → **Git**
2. Enable **Automatic deployments from Git**
3. Save changes
4. Make a new commit and push to trigger deployment

## Verify Deployment

After redeploy, check:
1. ✅ Build completed successfully (green checkmark)
2. ✅ No errors in build logs
3. ✅ Visit your Vercel URL - app should load
4. ✅ Open Analysis Tool - Risk-Reward panel should be visible
5. ✅ Check browser console - no MIME type errors

## Troubleshooting

### Issue: "Build failed"
- Check build logs for specific error
- Verify `package.json` has correct scripts
- Ensure all dependencies are in `package.json`

### Issue: "404 Not Found"
- Verify Root Directory is `frontend/caria-app`
- Check that `vercel.json` exists in `frontend/caria-app/`
- Verify `dist` folder is being created during build

### Issue: "Module script MIME type error"
- This should be fixed with the updated `vercel.json`
- Clear browser cache and hard refresh
- Verify the rewrite rules exclude `.js` and `.mjs` files

## Current Configuration

Your `vercel.json` should have:
- ✅ Build Command: `npm run build`
- ✅ Output Directory: `dist`
- ✅ Framework: `vite`
- ✅ Rewrites exclude static assets (JS, CSS, etc.)

## Quick Test

After fixing, test by:
1. Opening your Vercel URL
2. Opening Analysis Tool (Chat with Caria)
3. Looking for "Show Risk-Reward" button in header
4. Panel should appear on the right (desktop) or below (mobile)

