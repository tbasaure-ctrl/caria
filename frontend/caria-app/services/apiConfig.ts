/**
 * Unified API Configuration
 * 
 * This is the single source of truth for API URL configuration.
 * Supports both VITE_API_URL (for Vite) and NEXT_PUBLIC_API_URL (for Next.js compatibility).
 * 
 * Usage:
 *   import { API_BASE_URL } from './apiConfig';
 *   const response = await fetch(`${API_BASE_URL}/api/endpoint`);
 */

/**
 * Get the API base URL from environment variables.
 * 
 * Priority:
 * 1. NEXT_PUBLIC_API_URL (Next.js convention, from process.env)
 * 2. VITE_API_URL (Vite convention, from import.meta.env)
 * 3. Development fallback (localhost:8000) - only in development mode
 * 
 * In production, one of the env vars MUST be set.
 */
export const getApiBaseUrl = (): string => {
  // Check environment variables
  // Vite replaces import.meta.env.VITE_* at build time
  // Next.js uses process.env.NEXT_PUBLIC_*
  const viteApiUrl = typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL
    ? String(import.meta.env.VITE_API_URL).trim()
    : '';
  
  const nextPublicApiUrl = typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_API_URL
    ? String(process.env.NEXT_PUBLIC_API_URL).trim()
    : '';

  const envApiUrl = nextPublicApiUrl || viteApiUrl;

  if (envApiUrl) {
    // Ensure it's a valid absolute URL
    if (envApiUrl.startsWith('http://') || envApiUrl.startsWith('https://')) {
      return envApiUrl;
    }
    // If it doesn't start with http/https, assume https in production
    if (typeof window !== 'undefined' && window.location.protocol === 'https:') {
      return `https://${envApiUrl}`;
    }
    return `http://${envApiUrl}`;
  }

  // Development fallback - only use in development
  if (typeof window !== 'undefined') {
    const isLocalhost = window.location.hostname === 'localhost' || 
                       window.location.hostname === '127.0.0.1';
    if (isLocalhost) {
      console.warn(
        '⚠️ API URL not configured. Using development fallback: http://localhost:8000\n' +
        'Please set VITE_API_URL or NEXT_PUBLIC_API_URL environment variable.'
      );
      return 'http://localhost:8000';
    }
  }

  // Production fallback - should not happen if env vars are set correctly
  throw new Error(
    'API URL not configured. Please set VITE_API_URL or NEXT_PUBLIC_API_URL environment variable.'
  );
};

/**
 * The base URL for all API requests.
 * This should be used consistently throughout the application.
 */
export const API_BASE_URL = getApiBaseUrl();

/**
 * Get the WebSocket base URL (API_BASE_URL without /api suffix if present).
 */
export const getWebSocketBaseUrl = (): string => {
  let wsUrl = API_BASE_URL;
  // Remove /api suffix if present (WebSocket connects to base URL)
  if (wsUrl.endsWith('/api')) {
    wsUrl = wsUrl.slice(0, -4);
  }
  return wsUrl;
};

/**
 * WebSocket base URL for Socket.IO connections.
 */
export const WS_BASE_URL = getWebSocketBaseUrl();
