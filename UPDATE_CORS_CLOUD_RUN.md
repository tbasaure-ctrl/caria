# Update CORS in Cloud Run to Allow Vercel Frontend

## Problem
Backend is blocking requests from Vercel frontend due to CORS configuration.

## Solution

### Option 1: Via Google Cloud Console (Easiest)

1. Go to: https://console.cloud.google.com/run
2. Click on your service: `caria-api`
3. Click **EDIT & DEPLOY NEW REVISION**
4. Scroll down to **Container, Volumes, Networking, Security**
5. Click on **VARIABLES & SECRETS** tab
6. Find or add `CORS_ORIGINS` environment variable
7. Set value to:
   ```
   http://localhost:3000,http://localhost:5173,https://caria-r4pmdu1eb-tomas-projects-70a0592d.vercel.app,https://*.vercel.app
   ```

   **Note:** Replace with your actual Vercel production URL if different. The `https://*.vercel.app` pattern will match all Vercel preview deployments.

8. Click **DEPLOY**
9. Wait for deployment to complete (2-3 minutes)

### Option 2: Via gcloud CLI

```bash
gcloud run services update caria-api \
  --region=us-central1 \
  --update-env-vars="CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://caria-r4pmdu1eb-tomas-projects-70a0592d.vercel.app,https://*.vercel.app"
```

## Get Your Vercel Production URL

1. Go to: https://vercel.com/dashboard
2. Select your project
3. The production URL will be shown (usually `https://your-project.vercel.app`)
4. Use this URL in the CORS_ORIGINS variable

## Preview vs Production URLs

- **Preview URL** (per deployment): `https://caria-r4pmdu1eb-tomas-projects-70a0592d.vercel.app`
- **Production URL** (permanent): Usually `https://caria-app.vercel.app` or similar

**Recommendation:** Use the wildcard `https://*.vercel.app` to allow all Vercel deployments, or list both production and preview URLs explicitly.

## Testing After Update

1. Wait for Cloud Run deployment to complete
2. Visit your Vercel frontend
3. Try to login - the error should be gone
4. Check browser console (F12) for any errors

## Alternative: Allow All Origins (Not Recommended for Production)

If you want to temporarily allow all origins for testing:

```bash
gcloud run services update caria-api \
  --region=us-central1 \
  --update-env-vars="CORS_ORIGINS=*"
```

**Warning:** This is less secure and should only be used for testing.
