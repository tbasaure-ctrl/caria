# Railway & Frontend API Configuration Summary

## Overview
This document summarizes the changes made to ensure proper Railway backend deployment and unified API configuration across the frontend.

## Changes Made

### 1. Railway Configuration (`railway.json`)
**File:** `/workspace/railway.json`

**Changes:**
- Added `buildContext: "."` to ensure Railway builds from the repository root
- Backend Dockerfile path: `backend/Dockerfile` (already correct)
- Start command: `/app/backend/start.sh` (already correct)

**Status:** ✅ Configured to use `backend` folder as root directory

### 2. Unified API Configuration (`apiConfig.ts`)
**File:** `/workspace/frontend/caria-app/services/apiConfig.ts`

**New File Created:**
- Single source of truth for API URL configuration
- Supports both `VITE_API_URL` (Vite) and `NEXT_PUBLIC_API_URL` (Next.js compatibility)
- Development fallback to `http://localhost:8000` (only in development)
- Production requires environment variable to be set

**Key Exports:**
- `API_BASE_URL`: Base URL for all REST API calls
- `WS_BASE_URL`: Base URL for WebSocket/Socket.IO connections (automatically removes `/api` suffix if present)

**Usage:**
```typescript
import { API_BASE_URL, WS_BASE_URL } from './services/apiConfig';
```

### 3. Frontend Service Updates

**Files Updated:**
- `services/apiService.ts`: Now imports `API_BASE_URL` from `apiConfig.ts`
- `services/websocketService.ts`: Uses `WS_BASE_URL` from `apiConfig.ts`
- `services/communityService.ts`: Uses `API_BASE_URL` from `apiConfig.ts`
- `services/uxTrackingService.ts`: Uses `API_BASE_URL` from `apiConfig.ts`

**All Components Updated:**
- All widget components now import `API_BASE_URL` from `apiConfig.ts`
- Login/Register modals updated
- ChatWindow updated to use `WS_BASE_URL`
- Dashboard and other main components updated

**Total Files Updated:** 20+ component and service files

### 4. Removed Hardcoded URLs

**Removed:**
- ❌ `https://caria-production.up.railway.app` (hardcoded fallback in apiService.ts)
- ❌ `http://localhost:8000` (hardcoded fallback in ChatWindow.tsx)
- ❌ Hardcoded localhost URLs in vite.config.ts

**Replaced With:**
- ✅ Environment variable-based configuration
- ✅ Development fallback only when running on localhost
- ✅ Production requires explicit env var configuration

### 5. Backend CORS Configuration

**File:** `/workspace/backend/api/app.py`

**Current Configuration:**
```python
cors_origins_env = os.getenv("CORS_ORIGINS", "https://caria-way.com,http://localhost:3000,http://localhost:5173")
```

**Allowed Origins:**
- ✅ `https://caria-way.com` (production)
- ✅ `http://localhost:3000` (local development - Vite default port)
- ✅ `http://localhost:5173` (local development - Vite alternative port)
- ✅ All `*.vercel.app` domains (via regex pattern)

**Status:** ✅ Already correctly configured

### 6. Environment Variables

**Frontend (.env.example created):**
```bash
VITE_API_URL=http://localhost:8000
# NEXT_PUBLIC_API_URL=http://localhost:8000  # Alternative for Next.js
```

**Backend (CORS_ORIGINS):**
```bash
CORS_ORIGINS=https://caria-way.com,http://localhost:3000,http://localhost:5173
```

## Deployment Instructions

### Railway Backend Setup

1. **Link Repository:**
   - Railway will automatically detect `railway.json`
   - Build context: Repository root
   - Dockerfile: `backend/Dockerfile`

2. **Set Environment Variables:**
   ```bash
   CORS_ORIGINS=https://caria-way.com,http://localhost:3000,http://localhost:5173
   # ... other backend env vars
   ```

3. **Get Railway Backend URL:**
   - After deployment, Railway provides a public URL
   - Example: `https://your-backend.up.railway.app`

### Frontend Setup (Vercel/Other)

1. **Set Environment Variable:**
   ```bash
   VITE_API_URL=https://your-backend.up.railway.app
   # OR
   NEXT_PUBLIC_API_URL=https://your-backend.up.railway.app
   ```

2. **Build & Deploy:**
   - Frontend will use the env var for all API calls
   - No hardcoded URLs remain

## Testing Locally

### 1. Start Backend
```bash
cd backend
# Set CORS_ORIGINS if needed (default includes localhost:3000)
python -m uvicorn api.app:socketio_app --host 0.0.0.0 --port 8000
```

### 2. Start Frontend
```bash
cd frontend/caria-app
# Create .env file with:
# VITE_API_URL=http://localhost:8000
npm install
npm run dev
```

### 3. Verify
- Frontend should connect to backend without CORS errors
- Check browser console for API calls
- Verify WebSocket connections work

## File Structure

```
/workspace/
├── railway.json                    # Railway config (backend root)
├── backend/
│   ├── Dockerfile                  # Backend Docker build
│   ├── start.sh                    # Backend startup script
│   └── api/
│       └── app.py                  # CORS configuration
└── frontend/
    └── caria-app/
        ├── .env.example            # Frontend env template
        ├── services/
        │   ├── apiConfig.ts        # ✨ NEW: Unified API config
        │   ├── apiService.ts       # Updated to use apiConfig
        │   └── websocketService.ts # Updated to use apiConfig
        └── components/             # All updated to use apiConfig
```

## Key Benefits

1. **Single Source of Truth:** All API URLs come from `apiConfig.ts`
2. **Environment-Aware:** Automatically uses correct URL based on environment
3. **No Hardcoded URLs:** All URLs are configurable via environment variables
4. **CORS Configured:** Backend allows both production and development origins
5. **Railway Ready:** Backend configured for Railway deployment

## Troubleshooting

### CORS Errors
- Verify `CORS_ORIGINS` includes your frontend URL
- Check backend logs for CORS origin matching
- Ensure frontend URL matches exactly (including protocol)

### API Connection Errors
- Verify `VITE_API_URL` or `NEXT_PUBLIC_API_URL` is set correctly
- Check that Railway backend is running and accessible
- Verify backend URL doesn't have trailing slash

### WebSocket Connection Errors
- Verify `WS_BASE_URL` is correct (should be base URL without `/api`)
- Check that Socket.IO path is correct (`/socket.io/`)
- Ensure JWT token is being sent in handshake
