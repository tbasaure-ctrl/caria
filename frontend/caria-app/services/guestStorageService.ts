/**
 * Guest Storage Service
 * Manages localStorage-based data persistence for unauthenticated users.
 * Allows full widget functionality without requiring login.
 */

// Storage keys
const STORAGE_KEYS = {
  GUEST_HOLDINGS: 'caria-guest-holdings',
  GUEST_WATCHLIST: 'caria-guest-watchlist',
  GUEST_PREFERENCES: 'caria-guest-preferences',
  GUEST_BANNER_DISMISSED: 'caria-guest-banner-dismissed',
  GUEST_BANNER_DISMISS_TIME: 'caria-guest-banner-dismiss-time',
} as const;

// Types matching the API types from apiService.ts
export interface GuestHolding {
  id: string;
  ticker: string;
  quantity: number;
  average_cost: number;
  purchase_date?: string;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface GuestHoldingInput {
  ticker: string;
  quantity: number;
  average_cost: number;
  purchase_date?: string;
  notes?: string;
}

export interface GuestWatchlistItem {
  id: string;
  ticker: string;
  added_at: string;
  notes?: string;
}

// Generate unique ID for guest entries
const generateId = (): string => {
  return `guest-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

// ============================================================================
// HOLDINGS MANAGEMENT
// ============================================================================

export const getGuestHoldings = (): GuestHolding[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.GUEST_HOLDINGS);
    if (!stored) return [];
    return JSON.parse(stored);
  } catch (error) {
    console.error('Error reading guest holdings:', error);
    return [];
  }
};

export const saveGuestHoldings = (holdings: GuestHolding[]): void => {
  try {
    localStorage.setItem(STORAGE_KEYS.GUEST_HOLDINGS, JSON.stringify(holdings));
  } catch (error) {
    console.error('Error saving guest holdings:', error);
  }
};

export const createGuestHolding = (input: GuestHoldingInput): GuestHolding => {
  const now = new Date().toISOString();
  const newHolding: GuestHolding = {
    id: generateId(),
    ticker: input.ticker.toUpperCase(),
    quantity: input.quantity,
    average_cost: input.average_cost,
    purchase_date: input.purchase_date,
    notes: input.notes,
    created_at: now,
    updated_at: now,
  };

  const holdings = getGuestHoldings();
  holdings.push(newHolding);
  saveGuestHoldings(holdings);

  return newHolding;
};

export const updateGuestHolding = (
  id: string,
  updates: Partial<GuestHoldingInput>
): GuestHolding | null => {
  const holdings = getGuestHoldings();
  const index = holdings.findIndex((h) => h.id === id);

  if (index === -1) return null;

  holdings[index] = {
    ...holdings[index],
    ...updates,
    updated_at: new Date().toISOString(),
  };

  saveGuestHoldings(holdings);
  return holdings[index];
};

export const deleteGuestHolding = (id: string): boolean => {
  const holdings = getGuestHoldings();
  const filtered = holdings.filter((h) => h.id !== id);

  if (filtered.length === holdings.length) return false;

  saveGuestHoldings(filtered);
  return true;
};

// ============================================================================
// WATCHLIST MANAGEMENT
// ============================================================================

export const getGuestWatchlist = (): GuestWatchlistItem[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.GUEST_WATCHLIST);
    if (!stored) return [];
    return JSON.parse(stored);
  } catch (error) {
    console.error('Error reading guest watchlist:', error);
    return [];
  }
};

export const addToGuestWatchlist = (ticker: string, notes?: string): GuestWatchlistItem => {
  const item: GuestWatchlistItem = {
    id: generateId(),
    ticker: ticker.toUpperCase(),
    added_at: new Date().toISOString(),
    notes,
  };

  const watchlist = getGuestWatchlist();
  // Prevent duplicates
  if (!watchlist.some((w) => w.ticker === item.ticker)) {
    watchlist.push(item);
    localStorage.setItem(STORAGE_KEYS.GUEST_WATCHLIST, JSON.stringify(watchlist));
  }

  return item;
};

export const removeFromGuestWatchlist = (ticker: string): boolean => {
  const watchlist = getGuestWatchlist();
  const filtered = watchlist.filter((w) => w.ticker !== ticker.toUpperCase());

  if (filtered.length === watchlist.length) return false;

  localStorage.setItem(STORAGE_KEYS.GUEST_WATCHLIST, JSON.stringify(filtered));
  return true;
};

// ============================================================================
// GUEST MODE BANNER MANAGEMENT
// ============================================================================

const BANNER_RESHOW_HOURS = 24; // Show banner again after 24 hours

export const isGuestBannerDismissed = (): boolean => {
  try {
    const dismissed = localStorage.getItem(STORAGE_KEYS.GUEST_BANNER_DISMISSED);
    if (dismissed !== 'true') return false;

    const dismissTime = localStorage.getItem(STORAGE_KEYS.GUEST_BANNER_DISMISS_TIME);
    if (!dismissTime) return false;

    const dismissedAt = new Date(dismissTime);
    const now = new Date();
    const hoursSinceDismiss = (now.getTime() - dismissedAt.getTime()) / (1000 * 60 * 60);

    // If more than 24 hours have passed, show the banner again
    if (hoursSinceDismiss > BANNER_RESHOW_HOURS) {
      return false;
    }

    return true;
  } catch (error) {
    return false;
  }
};

export const dismissGuestBanner = (): void => {
  localStorage.setItem(STORAGE_KEYS.GUEST_BANNER_DISMISSED, 'true');
  localStorage.setItem(STORAGE_KEYS.GUEST_BANNER_DISMISS_TIME, new Date().toISOString());
};

// ============================================================================
// GUEST MODE UTILITIES
// ============================================================================

export const isGuestMode = (token: string | null): boolean => {
  return !token;
};

export const hasGuestData = (): boolean => {
  const holdings = getGuestHoldings();
  const watchlist = getGuestWatchlist();
  return holdings.length > 0 || watchlist.length > 0;
};

export const clearAllGuestData = (): void => {
  Object.values(STORAGE_KEYS).forEach((key) => {
    localStorage.removeItem(key);
  });
};

// ============================================================================
// MIGRATE GUEST DATA TO USER ACCOUNT (after login)
// ============================================================================

export interface MigrationResult {
  holdings: GuestHolding[];
  watchlist: GuestWatchlistItem[];
}

export const getGuestDataForMigration = (): MigrationResult => {
  return {
    holdings: getGuestHoldings(),
    watchlist: getGuestWatchlist(),
  };
};

export const clearGuestDataAfterMigration = (): void => {
  localStorage.removeItem(STORAGE_KEYS.GUEST_HOLDINGS);
  localStorage.removeItem(STORAGE_KEYS.GUEST_WATCHLIST);
};

