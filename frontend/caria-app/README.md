<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Caria Frontend Application

React + TypeScript + Vite frontend for the Caria investment intelligence platform.

## Prerequisites

- Node.js 20+ 
- npm or yarn

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env.local
   ```
   
   Edit `.env.local` and set:
   - `VITE_API_URL=http://localhost:8000` (or your API URL)
   - `VITE_GEMINI_API_KEY=your-key` (optional, for chat features)

3. **Run development server:**
   ```bash
   npm run dev
   ```
   
   The app will be available at `http://localhost:3000`

## Build for Production

```bash
npm run build
```

The built files will be in `dist/` directory.

## Docker Deployment

See `Dockerfile` and `nginx.conf` for production deployment configuration.

```bash
docker build -t caria-frontend .
docker run -p 80:80 caria-frontend
```

## Project Structure

- `components/` - React components
  - `widgets/` - Dashboard widgets (ModelOutlook, IdealPortfolio, etc.)
- `services/` - API service layer
  - `apiService.ts` - API client with auth and refresh token handling
- `data/` - Mock data and tour steps
- `types.ts` - TypeScript type definitions

## Connecting to Backend

The frontend connects to the Caria API backend. Ensure:

1. Backend API is running (default: `http://localhost:8000`)
2. CORS is configured in backend to allow frontend origin
3. Environment variable `VITE_API_URL` matches your backend URL

## Features

- User authentication (login/register)
- Dashboard with investment analysis widgets
- Challenge Your Thesis tool (RAG-powered analysis)
- Real-time regime detection display
- Factor-based stock screening
- Company valuation tool

## Development

- Uses Vite for fast HMR (Hot Module Replacement)
- TypeScript for type safety
- Tailwind CSS for styling
- React 19 for UI components
