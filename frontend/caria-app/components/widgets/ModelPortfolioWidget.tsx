/**
 * ModelPortfolioWidget - Shows model-selected portfolio for validation.
 * Replaces IdealPortfolio widget with model portfolio selection functionality.
 */

import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { PortfolioPerformance } from './PortfolioPerformance';
import { getErrorMessage } from '../../src/utils/errorHandling';

interface Holding {
    ticker: string;
    allocation: number;
}

interface ModelPortfolio {
    id: string;
    created_at: string;
    selection_type: 'outlier' | 'balanced' | 'random';
    regime: string | null;
    holdings: Holding[];
    total_holdings: number;
    initial_value: number;
    status: string;
    notes: string | null;
}

export const ModelPortfolioWidget: React.FC = () => {
    const [portfolios, setPortfolios] = useState<ModelPortfolio[]>([]);
    const [selectedPortfolio, setSelectedPortfolio] = useState<ModelPortfolio | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isSelecting, setIsSelecting] = useState(false);
    const [selectionType, setSelectionType] = useState<'outlier' | 'balanced' | 'random'>('balanced');
    const [numHoldings, setNumHoldings] = useState(15);

    useEffect(() => {
        loadPortfolios();
    }, []);

    const loadPortfolios = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await fetchWithAuth(`${API_BASE_URL}/api/portfolio/model/list?status=active`);

            if (!response.ok) {
                throw new Error('Failed to load model portfolios');
            }

            const data: ModelPortfolio[] = await response.json();
            setPortfolios(data);
            if (data.length > 0 && !selectedPortfolio) {
                setSelectedPortfolio(data[0]);
            }
        } catch (err: any) {
            console.error('Error loading model portfolios:', err);
            setError('Coming soon... Model portfolios are being enhanced with better allocation strategies.');
        } finally {
            setLoading(false);
        }
    };

    const handleSelectPortfolio = async () => {
        setIsSelecting(true);
        setError(null);

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/portfolio/model/select`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    selection_type: selectionType,
                    num_holdings: numHoldings,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Failed to select portfolio' }));
                throw new Error(errorData.detail || 'Failed to select portfolio');
            }

            const newPortfolio = await response.json();
            await loadPortfolios(); // Reload list
            setSelectedPortfolio(newPortfolio);
        } catch (err: any) {
            console.error('Error selecting portfolio:', err);
            setError('Coming soon... Portfolio selection is being enhanced for better recommendations.');
        } finally {
            setIsSelecting(false);
        }
    };

    if (loading) {
        return (
            <WidgetCard
                title="Model Portfolio"
                id="model-portfolio-widget"
                tooltip="Portfolios seleccionados por el modelo basados en análisis cuantitativo. Elige entre balanced, outlier o random según tu estrategia."
            >
                <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    Loading model portfolios...
                </div>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard
            title="Model Portfolio"
            id="model-portfolio-widget"
            tooltip="Portfolios seleccionados por el modelo basados en análisis cuantitativo. Elige entre balanced, outlier o random según tu estrategia."
        >
            <div className="space-y-4">
                {/* Selection Controls */}
                <div className="space-y-3 p-3 rounded-lg" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
                    <div>
                        <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-primary)' }}>
                            Selection Type
                        </label>
                        <select
                            value={selectionType}
                            onChange={(e) => setSelectionType(e.target.value as 'outlier' | 'balanced' | 'random')}
                            className="w-full px-3 py-2 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-primary)',
                                border: '1px solid var(--color-bg-tertiary)',
                                color: 'var(--color-text-primary)',
                            }}
                        >
                            <option value="balanced">Balanced (Well-diversified)</option>
                            <option value="outlier">Outlier (Unusual allocations)</option>
                            <option value="random">Random (Baseline)</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-primary)' }}>
                            Number of Holdings: {numHoldings}
                        </label>
                        <input
                            type="range"
                            min="10"
                            max="20"
                            value={numHoldings}
                            onChange={(e) => setNumHoldings(parseInt(e.target.value))}
                            className="w-full"
                            style={{ accentColor: 'var(--color-primary)' }}
                        />
                    </div>

                    <button
                        onClick={handleSelectPortfolio}
                        disabled={isSelecting}
                        className="w-full px-4 py-2 rounded-lg font-medium transition-opacity"
                        style={{
                            backgroundColor: isSelecting ? 'var(--color-bg-tertiary)' : 'var(--color-primary)',
                            color: 'var(--color-cream)',
                            opacity: isSelecting ? 0.6 : 1,
                            cursor: isSelecting ? 'not-allowed' : 'pointer',
                        }}
                    >
                        {isSelecting ? 'Selecting...' : 'Select New Portfolio'}
                    </button>
                </div>

                {/* Error */}
                {error && (
                    <div className="p-3 rounded-lg" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#ef4444' }}>
                        {error}
                    </div>
                )}

                {/* Portfolio List */}
                {portfolios.length > 0 && (
                    <div>
                        <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-primary)' }}>
                            Active Portfolios
                        </label>
                        <select
                            value={selectedPortfolio?.id || ''}
                            onChange={(e) => {
                                const portfolio = portfolios.find((p) => p.id === e.target.value);
                                setSelectedPortfolio(portfolio || null);
                            }}
                            className="w-full px-3 py-2 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-secondary)',
                                border: '1px solid var(--color-bg-tertiary)',
                                color: 'var(--color-text-primary)',
                            }}
                        >
                            {portfolios.map((portfolio) => (
                                <option key={portfolio.id} value={portfolio.id}>
                                    {portfolio.selection_type} - {portfolio.total_holdings} holdings ({new Date(portfolio.created_at).toLocaleDateString()})
                                </option>
                            ))}
                        </select>
                    </div>
                )}

                {/* Selected Portfolio Details */}
                {selectedPortfolio && (
                    <div className="space-y-3">
                        <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm font-semibold" style={{ color: 'var(--color-cream)' }}>
                                    {selectedPortfolio.selection_type.charAt(0).toUpperCase() + selectedPortfolio.selection_type.slice(1)} Portfolio
                                </span>
                                <span className="text-xs px-2 py-1 rounded" style={{ backgroundColor: 'var(--color-bg-tertiary)', color: 'var(--color-text-secondary)' }}>
                                    {selectedPortfolio.total_holdings} holdings
                                </span>
                            </div>
                            {selectedPortfolio.regime && (
                                <div className="text-xs mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                                    Regime: {selectedPortfolio.regime}
                                </div>
                            )}
                            <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                Created: {new Date(selectedPortfolio.created_at).toLocaleDateString()}
                            </div>
                        </div>

                        {/* Holdings List */}
                        <div className="p-3 rounded-lg max-h-48 overflow-y-auto" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
                            <h4 className="text-sm font-semibold mb-2" style={{ color: 'var(--color-cream)' }}>Holdings</h4>
                            <div className="space-y-1">
                                {selectedPortfolio.holdings.map((holding, idx) => (
                                    <div key={idx} className="flex justify-between text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                        <span className="font-mono">{holding.ticker}</span>
                                        <span>{holding.allocation.toFixed(2)}%</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Performance Component */}
                        <PortfolioPerformance portfolioId={selectedPortfolio.id} />
                    </div>
                )}

                {portfolios.length === 0 && !loading && (
                    <div className="text-sm text-center py-4" style={{ color: 'var(--color-text-secondary)' }}>
                        No active portfolios. Select a new portfolio to start tracking.
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};

