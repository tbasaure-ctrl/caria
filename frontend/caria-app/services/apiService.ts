// API Service for Caria
// Handles all API communication with the backend

const TOKEN_KEY = 'caria_auth_token';
const REFRESH_TOKEN_KEY = 'caria_refresh_token';

export const API_BASE_URL = (() => {
    if (typeof window !== 'undefined') {
        const envUrl = import.meta.env.VITE_API_URL;
        if (envUrl) return envUrl;
        
        // Fallback to current origin for API
        const protocol = window.location.protocol;
        const hostname = window.location.hostname;
        const port = window.location.port;
        
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return port ? `${protocol}//${hostname}:${port.replace('5173', '8000')}` : `${protocol}//${hostname}:8000`;
        }
        
        return `${protocol}//${hostname}${port ? `:${port}` : ''}`;
    }
    return 'http://localhost:8000';
})();

export const API_URL = API_BASE_URL;

// Token management
export const saveToken = (token: string, refreshToken?: string): void => {
    if (typeof window !== 'undefined') {
        localStorage.setItem(TOKEN_KEY, token);
        if (refreshToken) {
            localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
        }
    }
};

export const getToken = (): string | null => {
    if (typeof window !== 'undefined') {
        return localStorage.getItem(TOKEN_KEY);
    }
    return null;
};

export const getAuthToken = getToken;

export const getRefreshToken = (): string | null => {
    if (typeof window !== 'undefined') {
        return localStorage.getItem(REFRESH_TOKEN_KEY);
    }
    return null;
};

export const removeToken = (): void => {
    if (typeof window !== 'undefined') {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(REFRESH_TOKEN_KEY);
    }
};

export const logout = (): void => {
    removeToken();
    if (typeof window !== 'undefined') {
        window.location.href = '/';
    }
};

// Types
export interface Holding {
    id: string;
    ticker: string;
    quantity: number;
    average_cost: number;
    purchase_date?: string;
    notes?: string;
    created_at?: string;
    updated_at?: string;
}

export interface RealtimePrice {
    price: number;
    change: number;
    changesPercentage: number;
    previousClose?: number;
    close?: number;
}

export interface HoldingWithPrice extends Holding {
    current_price?: number;
    cost_basis?: number;
    current_value?: number;
    gain_loss?: number;
    gain_loss_pct?: number;
    price_change?: number;
    price_change_pct?: number;
    price_source?: string;
}

export interface HoldingsWithPrices {
    holdings: HoldingWithPrice[];
    total_value: number;
    total_cost: number;
    total_gain_loss: number;
    total_gain_loss_pct: number;
}

export interface LectureRecommendation {
    id: string;
    title: string;
    description: string;
    url: string;
    category: string;
    difficulty: 'beginner' | 'intermediate' | 'advanced';
}

// Fetch with authentication
export const fetchWithAuth = async (url: string, options: RequestInit = {}, retry = true): Promise<Response> => {
    const token = getToken();
    const headers: HeadersInit = {
        'Content-Type': 'application/json',
        ...(options.headers as HeadersInit),
    };
    
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    
    const response = await fetch(url, {
        ...options,
        headers,
    });
    
    if (response.status === 401 && retry) {
        // Try to refresh token
        const refreshToken = getRefreshToken();
        if (refreshToken) {
            try {
                const refreshResponse = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ refresh_token: refreshToken }),
                });
                
                if (refreshResponse.ok) {
                    const data = await refreshResponse.json();
                    saveToken(data.access_token, data.refresh_token);
                    // Retry original request
                    return fetchWithAuth(url, options, false);
                }
            } catch (error) {
                console.error('Token refresh failed:', error);
            }
        }
        
        // Refresh failed, logout
        logout();
        throw new Error('Authentication failed');
    }
    
    return response;
};

// Price fetching
export const fetchPrices = async (tickers: string[]): Promise<Record<string, RealtimePrice>> => {
    if (tickers.length === 0) return {};
    
    try {
        const response = await fetchWithAuth(
            `${API_URL}/api/prices/realtime?tickers=${tickers.join(',')}`
        );
        
        if (!response.ok) {
            throw new Error(`Error fetching prices: ${response.statusText}`);
        }
        
        return response.json();
    } catch (error) {
        console.error('Error fetching prices:', error);
        return {};
    }
};

// Holdings API
export const fetchHoldings = async (): Promise<Holding[]> => {
    const response = await fetchWithAuth(`${API_URL}/api/holdings`);
    
    if (!response.ok) {
        if (response.status === 404) {
            return [];
        }
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Error fetching holdings: ${response.statusText}`);
    }
    
    return response.json();
};

export const fetchHoldingsWithPrices = async (currency: string = "USD"): Promise<HoldingsWithPrices> => {
    const response = await fetchWithAuth(`${API_URL}/api/holdings/with-prices?currency=${currency}`);

    if (!response.ok) {
        if (response.status === 404) {
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
        body: JSON.stringify(holding),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Error creating holding: ${response.statusText}`);
    }

    return response.json();
};

export const deleteHolding = async (holdingId: string): Promise<void> => {
    const response = await fetchWithAuth(`${API_URL}/api/holdings/${holdingId}`, {
        method: 'DELETE',
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Error deleting holding: ${response.statusText}`);
    }
};

export const updateHolding = async (holdingId: string, updates: {
    quantity?: number;
    average_cost?: number;
    notes?: string;
}): Promise<Holding> => {
    const response = await fetchWithAuth(`${API_URL}/api/holdings/${holdingId}`, {
        method: 'PUT',
        body: JSON.stringify(updates),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Error updating holding: ${response.statusText}`);
    }

    return response.json();
};

// Lecture recommendations
export const fetchRecommendedLectures = async (): Promise<LectureRecommendation[]> => {
    const response = await fetchWithAuth(`${API_URL}/api/lectures/recommendations`);
    
    if (!response.ok) {
        return [];
    }
    
    return response.json();
};

// Economic Monitor API functions
export interface BusinessCyclePoint {
    country_code: string;
    country_name: string;
    x: number;
    y: number;
    phase: 'expansion' | 'slowdown' | 'recession' | 'recovery';
    trajectory?: Array<{ x: number; y: number }>;
}

export interface CurrencyRate {
    currency_pair: string;
    base_currency: string;
    quote_currency: string;
    rate: number;
    date: string;
    change_1d?: number;
    change_1w?: number;
    change_1m?: number;
    change_1y?: number;
    change_pct_1d?: number;
    change_pct_1w?: number;
    change_pct_1m?: number;
    change_pct_1y?: number;
}

export interface CurrencyHistory {
    currency_pair: string;
    dates: string[];
    rates: number[];
    country_code?: string;
    country_name?: string;
}

export interface HeatmapCell {
    country_code: string;
    country_name: string;
    indicator_name: string;
    indicator_category: string;
    value: number;
    z_score: number;
    normalized_value: number;
    status: 'health' | 'warning' | 'deterioration';
}

export const fetchBusinessCycle = async (): Promise<{ points: BusinessCyclePoint[]; last_updated: string }> => {
    const response = await fetchWithAuth(`${API_URL}/api/economic-monitor/business-cycle`);
    if (!response.ok) {
        throw new Error(`Error fetching business cycle: ${response.statusText}`);
    }
    return response.json();
};

export const fetchCurrencies = async (): Promise<{ rates: CurrencyRate[]; last_updated: string }> => {
    const response = await fetchWithAuth(`${API_URL}/api/economic-monitor/currencies`);
    if (!response.ok) {
        throw new Error(`Error fetching currencies: ${response.statusText}`);
    }
    return response.json();
};

export const fetchCurrencyHistory = async (currencyPair: string, days: number = 365): Promise<{ history: CurrencyHistory; last_updated: string }> => {
    const response = await fetchWithAuth(`${API_URL}/api/economic-monitor/currency/${currencyPair}?days=${days}`);
    if (!response.ok) {
        throw new Error(`Error fetching currency history: ${response.statusText}`);
    }
    return response.json();
};

export const fetchHeatmap = async (): Promise<{ cells: HeatmapCell[]; countries: string[]; indicators: string[]; last_updated: string }> => {
    const response = await fetchWithAuth(`${API_URL}/api/economic-monitor/heatmap`);
    if (!response.ok) {
        throw new Error(`Error fetching heatmap: ${response.statusText}`);
    }
    return response.json();
};

export const fetchCountryDetails = async (countryCode: string): Promise<any> => {
    const response = await fetchWithAuth(`${API_URL}/api/economic-monitor/country/${countryCode}`);
    if (!response.ok) {
        throw new Error(`Error fetching country details: ${response.statusText}`);
    }
    return response.json();
};
