# Frontend Diagnostic Guide

## If Only Extension Errors Show, Your App Might Not Be Loading

### Step 1: Check if JavaScript Files Are Loading

1. **Open DevTools → Network Tab**
2. **Reload the page**
3. **Look for these files:**
   - `index.tsx` or `index.js`
   - `index-CH_5SE3M.js` (or similar hash)
   - `index-K4t4p628.css` (CSS file)

**What to check:**
- ✅ **Green (200)**: Files loaded successfully
- ❌ **Red (404)**: Files not found - build/deployment issue
- ❌ **Red (CORS)**: CORS error
- ⏳ **Pending**: Files still loading

### Step 2: Check Page Source

1. **Right-click page → "View Page Source"**
2. **Look for:**
   ```html
   <div id="root"></div>
   <script type="module" src="/index.tsx"></script>
   ```
3. **Check if script tag points to correct file**

### Step 3: Check Console for Silent Errors

Even if you only see extension errors, check:

1. **Open Console**
2. **Look for:**
   - Red errors (even if they're extension errors)
   - Warnings
   - Any messages mentioning your app

3. **Try this in console:**
   ```javascript
   // Check if root element exists
   console.log('Root element:', document.getElementById('root'));
   
   // Check if React loaded
   console.log('React:', typeof React);
   
   // Check if scripts loaded
   console.log('Scripts:', document.querySelectorAll('script'));
   ```

### Step 4: Check Vercel Deployment

1. **Go to Vercel Dashboard**
2. **Check latest deployment:**
   - Status: Should be "Ready"
   - Build: Should be "Success"
   - Check build logs for errors

3. **Check deployment URL:**
   - Make sure you're accessing the correct URL
   - Try both production and preview URLs

### Step 5: Common Issues

#### Issue: Blank Page, No Errors
**Possible causes:**
- JavaScript bundle not loading
- React app crashing silently
- Build failed but deployment succeeded

**Fix:**
- Check Network tab for failed JS file loads
- Check Vercel build logs
- Try hard refresh (Ctrl+Shift+R)

#### Issue: 404 on JS Files
**Possible causes:**
- Build output path incorrect
- Vercel configuration issue

**Fix:**
- Check `vite.config.ts` build settings
- Verify `package.json` build script
- Check Vercel project settings

#### Issue: CORS Errors
**Possible causes:**
- Backend CORS not configured
- Wrong API URL

**Fix:**
- Check Railway `CORS_ORIGINS` variable
- Verify `VITE_API_URL` in Vercel

### Step 6: Quick Test

Run these in browser console:

```javascript
// 1. Check if page loaded
console.log('Document ready:', document.readyState);
console.log('Root exists:', !!document.getElementById('root'));

// 2. Check if scripts loaded
const scripts = Array.from(document.querySelectorAll('script'));
console.log('Scripts found:', scripts.length);
scripts.forEach(s => console.log('Script:', s.src || s.textContent.substring(0, 50)));

// 3. Check if React is trying to load
console.log('Window.React:', window.React);
console.log('Window.ReactDOM:', window.ReactDOM);

// 4. Check for errors
window.addEventListener('error', (e) => {
  console.error('Page error:', e);
});

// 5. Check network requests
console.log('Check Network tab for failed requests');
```

### Step 7: What to Share

If still not working, share:

1. **Network Tab Screenshot:**
   - Show all requests
   - Highlight any failed (red) requests

2. **Console Output:**
   - After running diagnostic commands above
   - Any errors (even if they look unrelated)

3. **Page Source:**
   - Right-click → View Source
   - Check if `<div id="root">` exists
   - Check if script tags are correct

4. **Vercel Build Logs:**
   - Latest deployment build output
   - Any errors or warnings

5. **What You See:**
   - Blank white page?
   - Blank black page?
   - Any text/content at all?
