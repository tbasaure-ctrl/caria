# Frontend Testing & Debugging Guide

## 🔍 Testing Prices Feature

### Issue: Prices Not Loading

**Root Cause**: Prices endpoint requires authentication (`get_current_user`)

**Solution Options**:
1. **Make prices optional auth** (recommended for public market data)
2. **Ensure user is logged in** before Dashboard already implemented

**Current Behavior**:
- GlobalMarketBar calls `fetchPrices()` which uses `fetchWithAuth()`
- If user not logged in → 401 error → Shows error message
- If user logged in → Should work

**To Test**:
1. Log in first
2. Check browser console for errors
3. Verify token exists: `localStorage.getItem('caria-auth-token')`
4. Check Network tab for `/api/prices/realtime` request
5. Verify response contains `prices` object

**If Still Failing**:
- Check Railway logs for OpenBB/FMP API errors
- Verify `FMP_API_KEY` is set in Railway
- Check if OpenBB client is initialized correctly

## 🔍 Testing Chat/WebSocket Feature

### Issue: Chat Not Connecting

**Potential Causes**:
1. **WebSocket URL incorrect** - Should be Railway URL without `/api`
2. **CORS blocking WebSocket** - Check Railway CORS_ORIGINS
3. **Socket.IO path incorrect** - Should be `/socket.io/`
4. **Authentication failing** - Token invalid or expired
5. **Railway WebSocket support** - Verify Railway supports WebSockets

**Current Implementation**:
- ChatWindow uses Socket.IO client
- Connects to `API_BASE_URL` without `/api` suffix
- Sends token in `auth` object
- Path: `/socket.io/`

**To Test**:
1. Log in first
2. Open chat window from sidebar
3. Check connection status (should show green "Connected")
4. Check browser console for WebSocket errors
5. Check Railway logs for WebSocket connection attempts

**Debug Steps**:
```javascript
// In browser console, check WebSocket connection:
const token = localStorage.getItem('caria-auth-token');
console.log('Token exists:', !!token);
console.log('API URL:', import.meta.env.VITE_API_URL);
```

**If Still Failing**:
- Verify Railway is running `uvicorn api.app:socketio_app` (not just `app`)
- Check Railway logs for Socket.IO initialization
- Verify CORS allows WebSocket upgrade requests
- Check if Railway supports WebSocket connections (some platforms don't)

## 🚀 Deployment Verification

### Railway Configuration:
- [ ] App runs: `uvicorn api.app:socketio_app --host 0.0.0.0 --port $PORT`
- [ ] `CORS_ORIGINS` includes Vercel URL
- [ ] WebSocket connections are allowed
- [ ] `FMP_API_KEY` is set (for prices)
- [ ] `LLAMA_API_KEY` is set (for chat)
- [ ] Database connection works

### Vercel Configuration:
- [ ] `VITE_API_URL` = Railway backend URL
- [ ] Build completes successfully
- [ ] No TypeScript errors
- [ ] All environment variables set

## 📋 Quick Test Checklist

### After Deployment:
1. **Login**: ✅ Works
2. **Dashboard Loads**: ✅ Works  
3. **Prices Display**: ⚠️ Check (requires auth)
4. **Chat Connects**: ⚠️ Check (WebSocket)
5. **Weekly Media**: ✅ Should work (static)
6. **Gold Indicator**: ✅ Should work (uses prices)

### Browser Console Checks:
- No red errors
- WebSocket connection successful
- API requests return 200 (not 401/403)
- CORS errors absent

## 🐛 Common Issues & Fixes

### Prices Not Loading:
- **401 Error**: User not logged in → Login first
- **500 Error**: Backend issue → Check Railway logs
- **Network Error**: CORS or URL issue → Check `VITE_API_URL`

### Chat Not Working:
- **Connection Failed**: Check WebSocket URL, CORS, Railway WebSocket support
- **Auth Failed**: Token expired → Log out and back in
- **No Response**: LLM service not initialized → Check Railway logs

### Quick Fixes:
1. **Clear browser cache** and localStorage
2. **Log out and back in** to refresh token
3. **Check Railway logs** for backend errors
4. **Verify environment variables** in both Railway and Vercel
