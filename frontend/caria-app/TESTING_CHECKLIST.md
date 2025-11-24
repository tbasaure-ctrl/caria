# Frontend Testing Checklist

## ✅ Pre-Deployment Verification

### 1. Authentication & Login
- [ ] Login with valid credentials works
- [ ] Login with invalid credentials shows error
- [ ] Token is saved to localStorage after login
- [ ] Token persists on page refresh
- [ ] Logout clears token and redirects

### 2. Prices & Market Data
**Endpoint**: `POST /api/prices/realtime`
- [ ] GlobalMarketBar displays SPY, STOXX 600, EEM, and Gold (GLD)
- [ ] Prices update every 30 seconds
- [ ] Shows correct price, change, and percentage
- [ ] Handles errors gracefully (shows error message)
- [ ] Works when logged in (requires authentication)

**Potential Issues**:
- Prices endpoint requires authentication - ensure user is logged in
- Check browser console for 401/403 errors
- Verify `VITE_API_URL` is set correctly in Vercel

### 3. Chat/WebSocket
**Endpoint**: WebSocket connection + `/api/chat/history`
- [ ] Chat window opens from sidebar
- [ ] WebSocket connects successfully (green status indicator)
- [ ] Can send messages
- [ ] Receives AI responses
- [ ] Chat history loads on connection
- [ ] Reconnects automatically if connection drops

**Potential Issues**:
- WebSocket URL might need to be `wss://` in production (HTTPS)
- CORS might block WebSocket connections
- Check Railway logs for WebSocket connection errors
- Verify Socket.IO path is correct (`/socket.io/`)

### 4. Dashboard Widgets
- [ ] ModelOutlook displays regime data
- [ ] FearGreedIndex loads and displays
- [ ] Portfolio widget loads holdings
- [ ] CommunityFeed displays posts
- [ ] RankingsWidget displays rankings
- [ ] WeeklyMedia displays podcast/youtube links
- [ ] All widgets handle errors gracefully

### 5. Weekly Media Component
- [ ] Displays both podcast and YouTube video
- [ ] Links open in new tab
- [ ] Thumbnails load correctly
- [ ] Positioned below Research section

### 6. Gold Indicator
- [ ] GLD (Gold) appears in GlobalMarketBar
- [ ] Shows price and change percentage
- [ ] Updates with other indicators

## 🔍 Debugging Steps

### If Prices Don't Work:
1. Open browser console (F12)
2. Check Network tab for `/api/prices/realtime` request
3. Verify:
   - Request includes `Authorization: Bearer <token>` header
   - Response status (should be 200)
   - Response contains `prices` object
4. Check Railway backend logs for errors

### If Chat Doesn't Work:
1. Open browser console (F12)
2. Check for WebSocket connection errors
3. Verify:
   - Token exists in localStorage (`caria-auth-token`)
   - WebSocket URL is correct (should be Railway URL without `/api`)
   - Connection status shows "connected" (green dot)
4. Check Railway backend logs for WebSocket errors
5. Verify CORS allows WebSocket connections

### Common Issues:
- **CORS Errors**: Check `CORS_ORIGINS` in Railway includes Vercel URL
- **401/403 Errors**: Token expired or invalid - try logging out and back in
- **WebSocket Connection Failed**: Check Railway URL, verify Socket.IO is mounted correctly
- **Prices Not Loading**: Verify authentication, check backend logs for OpenBB/FMP API errors

## 🚀 Deployment Checklist

### Vercel Environment Variables:
- [ ] `VITE_API_URL` = Railway backend URL (e.g., `https://caria-production.up.railway.app`)

### Railway Environment Variables:
- [ ] `CORS_ORIGINS` includes Vercel frontend URL
- [ ] `LLAMA_API_KEY` (Groq API key)
- [ ] `ANTHROPIC_API_KEY` (optional, for Claude fallback)
- [ ] `DATABASE_URL` (PostgreSQL connection string)
- [ ] All other required backend variables

## 📝 Testing Commands

### Test Prices Endpoint (from browser console):
```javascript
const token = localStorage.getItem('caria-auth-token');
fetch('https://caria-production.up.railway.app/api/prices/realtime', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ tickers: ['SPY', 'GLD'] })
})
.then(r => r.json())
.then(console.log)
.catch(console.error);
```

### Test Chat History (from browser console):
```javascript
const token = localStorage.getItem('caria-auth-token');
fetch('https://caria-production.up.railway.app/api/chat/history', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
})
.then(r => r.json())
.then(console.log)
.catch(console.error);
```

## ⚠️ Known Issues to Check

1. **Prices**: Requires authentication - ensure user is logged in
2. **Chat**: WebSocket might need `wss://` protocol in production
3. **CORS**: Verify Railway CORS_ORIGINS includes Vercel URL
4. **Token Expiry**: Tokens might expire - check refresh token logic
