const TOKEN_KEY = 'caria-auth-token';
const REFRESH_TOKEN_KEY = 'caria-refresh-token';

// Get API URL from environment or default - MUST be absolute URL per audit document
// INCORRECT: "/api/login" or "localhost:8000/api/login"
// CORRECT: "http://localhost:8000/api/login"
export const API_BASE_URL =
    (import.meta.env.VITE_API_URL || 'https://caria-production.up.railway.app').replace(/\/$/, '');


// Use API_BASE_URL consistently everywhere (per audit document 1.1)
const API_URL = API_BASE_URL;

export const saveToken = (token: string, refreshToken?: string): void => {
    localStorage.setItem(TOKEN_KEY, token);
    if (refreshToken) {
        localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
    }
};

export const getToken = (): string | null => {
    return localStorage.getItem(TOKEN_KEY);
};

// Alias for consistency
export const getAuthToken = getToken;

export const getRefreshToken = (): string | null => {
    return localStorage.getItem(REFRESH_TOKEN_KEY);
};

export const removeToken = (): void => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
};

export const logout = (): void => {
    removeToken();
    localStorage.removeItem('cariaChatHistory');
    window.location.href = '/';
};

const refreshAccessToken = async (): Promise<string | null> => {
    const refreshToken = getRefreshToken();
    if (!refreshToken) {
        return null;
    }

    try {
        const response = await fetch(`${API_URL}/api/auth/refresh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: refreshToken }),
        });

        if (!response.ok) {
            return null;
        }

        const data = await response.json();
        if (data.access_token && data.refresh_token) {
            saveToken(data.access_token, data.refresh_token);
            return data.access_token;
        }
        return null;
    } catch {
        return null;
    }
};

/**
 * Fetch with authentication and improved error handling per audit document.
 * Captures 4xx/5xx errors correctly (not just network errors).
 */
export const fetchWithAuth = async (url: string, options: RequestInit = {}, retry = true): Promise<Response> => {
    let token = getToken();

    const headers = new Headers(options.headers || {});
    headers.set('Content-Type', 'application/json');

    if (token) {
        headers.set('Authorization', `Bearer ${token}`);
    }

    let response: Response;

    try {
        response = await fetch(url, {
            ...options,
            headers,
        });
    } catch (error) {
        // This captures network errors (CORS, server down, DNS)
        console.error(`Network error connecting to service: ${url}`, error);
        throw new Error(`Failed to connect to server. Please check that the API is running at ${API_BASE_URL}`);
    }

    // Si recibimos 401 y tenemos refresh token, intentar refrescar
    if (response.status === 401 && retry) {
        const newToken = await refreshAccessToken();
        if (newToken) {
            // Reintentar request con nuevo token
            headers.set('Authorization', `Bearer ${newToken}`);
            try {
                response = await fetch(url, {
                    ...options,
                    headers,
                });
            } catch (error) {
                console.error(`Network error on retry: ${url}`, error);
                throw new Error(`Failed to connect to server. Please check that the API is running at ${API_BASE_URL}`);
            }
        } else {
            // Refresh falló, logout
            logout();
            throw new Error('Session expired. Please log in again.');
        }
    }

    if (response.status === 401 && !retry) {
        // Refresh ya falló, logout
        logout();
        throw new Error('Session expired. Please log in again.');
    }

    // Per audit document: Capture server errors (4xx, 5xx) that are NOT network errors
    // This is crucial for debugging
    if (!response.ok) {
        let errorData: any = {};
        try {
            errorData = await response.json();
        } catch {
            // If response is not JSON, use status text
            errorData = { detail: response.statusText };
        }
        console.error(`API error: ${response.status}`, errorData);
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return response;
};

// ============================================================================
// PRICES API
// ============================================================================

export interface RealtimePrice {
    symbol: string;
    price: number;
    change: number;
    changesPercentage: number;
    previousClose?: number;
    [key: string]: any;
}

export const fetchPrices = async (tickers: string[]): Promise<Record<string, RealtimePrice>> => {
    const response = await fetchWithAuth(`${API_URL}/api/prices/realtime`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tickers }),
    });

    if (!response.ok) {
        throw new Error(`Error fetching prices: ${response.statusText}`);
    }

    const data = await response.json();
    return data.prices || {};
};

// ============================================================================
// HOLDINGS API
// ============================================================================

export interface Holding {
    id: string;
    ticker: string;
    quantity: number;
    average_cost: number;
    notes?: string;
    created_at: string;
    updated_at: string;
}

export interface HoldingWithPrice extends Holding {
    current_price: number;
    cost_basis: number;
    current_value: number;
    gain_loss: number;
    gain_loss_pct: number;
    price_change: number;
    price_change_pct: number;
}

export interface HoldingsWithPrices {
    holdings: HoldingWithPrice[];
    total_value: number;
    total_cost: number;
    total_gain_loss: number;
    total_gain_loss_pct: number;
}

export const fetchHoldings = async (): Promise<Holding[]> => {
    const response = await fetchWithAuth(`${API_URL}/api/holdings`);

    if (!response.ok) {
        if (response.status === 404) {
            // Si es 404, probablemente no hay holdings aún - retornar lista vacía
            return [];
        }
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Error fetching holdings: ${response.statusText}`);
    }

    return response.json();
};

export const fetchHoldingsWithPrices = async (): Promise<HoldingsWithPrices> => {
    const response = await fetchWithAuth(`${API_URL}/api/holdings/with-prices`);

    if (!response.ok) {
        if (response.status === 404) {
            // Si es 404, retornar estructura vacía
            return {
                holdings: [],
                total_value: 0,
                total_cost: 0,
                total_gain_loss: 0,
                total_gain_loss_pct: 0,
            };
        }
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Error fetching holdings with prices: ${response.statusText}`);
    }

    return response.json();
};

export const createHolding = async (holding: {
    ticker: string;
    quantity: number;
    average_cost: number;
    purchase_date?: string;
    notes?: string;
}): Promise<Holding> => {
    const response = await fetchWithAuth(`${API_URL}/api/holdings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(holding),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || `Error creating holding: ${response.statusText}`);
    }

    return response.json();
};

export const deleteHolding = async (holdingId: string): Promise<void> => {
    const response = await fetchWithAuth(`${API_URL}/api/holdings/${holdingId}`, {
        method: 'DELETE',
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || `Error deleting holding: ${response.statusText}`);
    }
};

// ============================================================================
// LECTURES API
// ============================================================================

export interface LectureRecommendation {
    title: string;
    url: string;
    source: string;
    date: string;
}

export const fetchRecommendedLectures = async (): Promise<LectureRecommendation[]> => {
    const response = await fetchWithAuth(`${API_URL}/api/lectures/recommended`);

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || `Error fetching lectures: ${response.statusText}`);
    }

    return response.json();
};
