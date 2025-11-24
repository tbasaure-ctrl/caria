/**
 * Watchlist Widget
 * Display user's watchlist with quick access to Investment Thesis and Valuation tools
 */

import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface WatchlistItem {
    id: string;
    ticker: string;
    company_name?: string;
    added_date: string;
    current_price?: number;
    change_pct?: number;
}

interface WatchlistWidgetProps {
    onOpenThesis?: (ticker: string) => void;
    onOpenValuation?: (ticker: string) => void;
}

export const WatchlistWidget: React.FC<WatchlistWidgetProps> = ({
    onOpenThesis,
    onOpenValuation
}) => {
    const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [newTicker, setNewTicker] = useState('');
    const [showAddForm, setShowAddForm] = useState(false);

    useEffect(() => {
        loadWatchlist();
    }, []);

    const loadWatchlist = async () => {
        try {
            setError(null);
            const response = await fetchWithAuth(`${API_BASE_URL}/api/watchlist`);
            if (!response.ok) {
                throw new Error('Failed to load watchlist');
            }
            const data = await response.json();
            setWatchlist(data.watchlist || []);
        } catch (err: any) {
            console.error('Error loading watchlist:', err);
            setError('Could not load watchlist.');
        } finally {
            setLoading(false);
        }
    };

    const handleAddTicker = async () => {
        if (!newTicker.trim()) return;

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/watchlist`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker: newTicker.toUpperCase() })
            });

            if (!response.ok) {
                throw new Error('Failed to add ticker');
            }

            setNewTicker('');
            setShowAddForm(false);
            loadWatchlist();
        } catch (err: any) {
            console.error('Error adding ticker:', err);
            setError('Could not add ticker to watchlist.');
        }
    };

    const handleRemoveTicker = async (id: string) => {
        if (!confirm('Remove from watchlist?')) return;

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/watchlist/${id}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error('Failed to remove ticker');
            }

            loadWatchlist();
        } catch (err: any) {
            console.error('Error removing ticker:', err);
            setError('Could not remove ticker.');
        }
    };

    if (loading) {
        return (
            <WidgetCard title="WATCHLIST" tooltip="Track stocks you're interested in with quick access to analysis tools">
                <div className="text-sm text-slate-400">Loading...</div>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard title="WATCHLIST" tooltip="Track stocks you're interested in with quick access to Investment Thesis and Valuation">
            <div className="space-y-3">
                {error && (
                    <div className="text-sm text-red-400 bg-red-900/30 p-2 rounded-md border border-red-800">
                        {error}
                    </div>
                )}

                {/* Add Ticker Button */}
                {!showAddForm && (
                    <button
                        onClick={() => setShowAddForm(true)}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold py-2 px-3 rounded transition-colors"
                    >
                        + Add to Watchlist
                    </button>
                )}

                {/* Add Form */}
                {showAddForm && (
                    <div className="bg-gray-900/50 p-3 rounded border border-slate-800 space-y-2">
                        <input
                            type="text"
                            value={newTicker}
                            onChange={(e) => setNewTicker(e.target.value)}
                            onKeyPress={(e) => {
                                if (e.key === 'Enter') {
                                    handleAddTicker();
                                }
                            }}
                            placeholder="Enter ticker symbol (e.g., AAPL)"
                            className="w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 text-sm text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
                            autoFocus
                        />
                        <div className="flex gap-2">
                            <button
                                onClick={handleAddTicker}
                                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold py-1.5 px-3 rounded transition-colors"
                            >
                                Add
                            </button>
                            <button
                                onClick={() => {
                                    setShowAddForm(false);
                                    setNewTicker('');
                                }}
                                className="flex-1 bg-slate-700 hover:bg-slate-600 text-white text-sm font-semibold py-1.5 px-3 rounded transition-colors"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                )}

                {/* Watchlist Items */}
                {watchlist.length === 0 ? (
                    <div className="text-center text-sm text-slate-500 py-4">
                        Your watchlist is empty. Add stocks to track them here.
                    </div>
                ) : (
                    <div className="space-y-2 max-h-[500px] overflow-y-auto custom-scrollbar">
                        {watchlist.map((item) => (
                            <div
                                key={item.id}
                                className="flex items-center justify-between p-3 bg-gray-900/50 rounded border border-slate-800 hover:border-slate-700 transition-colors"
                            >
                                <div className="flex-1">
                                    <div className="flex items-baseline gap-2">
                                        <span className="text-slate-200 font-semibold">{item.ticker}</span>
                                        {item.change_pct !== undefined && (
                                            <span
                                                className={`text-xs ${item.change_pct >= 0 ? 'text-green-400' : 'text-red-400'
                                                    }`}
                                            >
                                                {item.change_pct >= 0 ? '+' : ''}
                                                {item.change_pct.toFixed(2)}%
                                            </span>
                                        )}
                                    </div>
                                    {item.company_name && (
                                        <div className="text-xs text-slate-400">{item.company_name}</div>
                                    )}
                                    {item.current_price && (
                                        <div className="text-xs text-slate-500 mt-1">
                                            ${item.current_price.toFixed(2)}
                                        </div>
                                    )}
                                </div>

                                {/* Action Buttons */}
                                <div className="flex  gap-2">
                                    {onOpenThesis && (
                                        <button
                                            onClick={() => onOpenThesis(item.ticker)}
                                            className="text-xs bg-slate-700 hover:bg-slate-600 text-slate-200 px-2 py-1 rounded transition-colors"
                                            title="View Investment Thesis"
                                        >
                                            Thesis
                                        </button>
                                    )}
                                    {onOpenValuation && (
                                        <button
                                            onClick={() => onOpenValuation(item.ticker)}
                                            className="text-xs bg-slate-700 hover:bg-slate-600 text-slate-200 px-2 py-1 rounded transition-colors"
                                            title="View Valuation"
                                        >
                                            Value
                                        </button>
                                    )}
                                    <button
                                        onClick={() => handleRemoveTicker(item.id)}
                                        className="text-xs text-red-400 hover:text-red-300 px-2 py-1 rounded hover:bg-red-900/20 transition-colors"
                                        title="Remove from watchlist"
                                    >
                                        ✕
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};
