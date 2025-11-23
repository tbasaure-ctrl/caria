/**
 * PortfolioPerformance - Shows performance vs benchmarks with metrics.
 */

import React, { useState, useEffect } from 'react';
import { fetchWithAuth } from '../../services/apiService';
import { API_BASE_URL } from '../../services/apiConfig';

interface PerformanceData {
    portfolio_id: string;
    date: string;
    portfolio_value: number;
    portfolio_return_pct: number;
    benchmark_sp500_return_pct: number | null;
    benchmark_qqq_return_pct: number | null;
    benchmark_vti_return_pct: number | null;
    sharpe_ratio: number | null;
    max_drawdown_pct: number | null;
    volatility_pct: number | null;
    alpha_pct: number | null;
    beta: number | null;
}

interface PortfolioPerformanceProps {
    portfolioId: string;
}

export const PortfolioPerformance: React.FC<PortfolioPerformanceProps> = ({ portfolioId }) => {
    const [performance, setPerformance] = useState<PerformanceData[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (portfolioId) {
            loadPerformance();
        }
    }, [portfolioId]);

    const loadPerformance = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await fetchWithAuth(
                `${API_BASE_URL}/api/portfolio/model/track?portfolio_id=${portfolioId}`
            );

            if (!response.ok) {
                throw new Error('Failed to load performance data');
            }

            const data: PerformanceData[] = await response.json();
            setPerformance(data);
        } catch (err: any) {
            console.error('Error loading performance:', err);
            setError('Coming soon... Performance tracking is being enhanced with historical analysis.');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
                <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>Loading performance...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-3 rounded-lg" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#ef4444' }}>
                <div className="text-xs">{error}</div>
            </div>
        );
    }

    if (performance.length === 0) {
        return (
            <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
                <div className="text-xs text-center" style={{ color: 'var(--color-text-secondary)' }}>
                    No performance data available yet. Performance will be tracked over time.
                </div>
            </div>
        );
    }

    const latest = performance[0]; // Most recent performance data

    return (
        <div className="p-3 rounded-lg space-y-3" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
            <h4 className="text-sm font-semibold" style={{ color: 'var(--color-cream)' }}>Performance Metrics</h4>

            {/* Return Comparison */}
            <div className="space-y-2">
                <div className="flex justify-between text-xs">
                    <span style={{ color: 'var(--color-text-secondary)' }}>Portfolio Return</span>
                    <span
                        className="font-semibold"
                        style={{
                            color: latest.portfolio_return_pct >= 0 ? '#10b981' : '#ef4444',
                        }}
                    >
                        {latest.portfolio_return_pct >= 0 ? '+' : ''}
                        {latest.portfolio_return_pct.toFixed(2)}%
                    </span>
                </div>
                {latest.benchmark_sp500_return_pct !== null && (
                    <div className="flex justify-between text-xs">
                        <span style={{ color: 'var(--color-text-secondary)' }}>vs S&P 500</span>
                        <span
                            style={{
                                color: latest.benchmark_sp500_return_pct >= 0 ? '#10b981' : '#ef4444',
                            }}
                        >
                            {latest.benchmark_sp500_return_pct >= 0 ? '+' : ''}
                            {latest.benchmark_sp500_return_pct.toFixed(2)}%
                        </span>
                    </div>
                )}
                {latest.alpha_pct !== null && (
                    <div className="flex justify-between text-xs">
                        <span style={{ color: 'var(--color-text-secondary)' }}>Alpha</span>
                        <span
                            className="font-semibold"
                            style={{
                                color: latest.alpha_pct >= 0 ? '#10b981' : '#ef4444',
                            }}
                        >
                            {latest.alpha_pct >= 0 ? '+' : ''}
                            {latest.alpha_pct.toFixed(2)}%
                        </span>
                    </div>
                )}
            </div>

            {/* Risk Metrics */}
            <div className="pt-2 border-t" style={{ borderColor: 'var(--color-bg-tertiary)' }}>
                <div className="grid grid-cols-2 gap-2 text-xs">
                    {latest.sharpe_ratio !== null && (
                        <div>
                            <div style={{ color: 'var(--color-text-secondary)' }}>Sharpe Ratio</div>
                            <div className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                                {latest.sharpe_ratio.toFixed(2)}
                            </div>
                        </div>
                    )}
                    {latest.beta !== null && (
                        <div>
                            <div style={{ color: 'var(--color-text-secondary)' }}>Beta</div>
                            <div className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                                {latest.beta.toFixed(2)}
                            </div>
                        </div>
                    )}
                    {latest.max_drawdown_pct !== null && (
                        <div>
                            <div style={{ color: 'var(--color-text-secondary)' }}>Max Drawdown</div>
                            <div className="font-semibold" style={{ color: '#ef4444' }}>
                                {latest.max_drawdown_pct.toFixed(2)}%
                            </div>
                        </div>
                    )}
                    {latest.volatility_pct !== null && (
                        <div>
                            <div style={{ color: 'var(--color-text-secondary)' }}>Volatility</div>
                            <div className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                                {latest.volatility_pct.toFixed(2)}%
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Latest Date */}
            <div className="text-xs pt-2 border-t" style={{ borderColor: 'var(--color-bg-tertiary)', color: 'var(--color-text-muted)' }}>
                Last updated: {new Date(latest.date).toLocaleDateString()}
            </div>
        </div>
    );
};

