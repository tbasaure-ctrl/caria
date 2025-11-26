import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { getErrorMessage } from '../../src/utils/errorHandling';

interface PortfolioMetrics {
    sharpe_ratio?: number;
    sortino_ratio?: number;
    alpha?: number;
    beta?: number;
    max_drawdown?: number;
    cagr?: number;
    volatility?: number;
    returns?: number;
}

interface PortfolioAnalyticsData {
    metrics: PortfolioMetrics;
    holdings_count: number;
    analysis_date: string;
}

export const PortfolioAnalytics: React.FC = () => {
    const [analytics, setAnalytics] = useState<PortfolioAnalyticsData | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [benchmark, setBenchmark] = useState<string>('SPY');

    useEffect(() => {
        const fetchAnalytics = async () => {
            setIsLoading(true);
            setError(null);
            try {
                const response = await fetchWithAuth(
                    `${API_BASE_URL}/api/portfolio/analysis/metrics?benchmark=${benchmark}`
                );
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch analytics' }));
                    throw new Error(errorData.detail || 'Failed to fetch portfolio analytics');
                }
                
                const data = await response.json();
                setAnalytics(data);
            } catch (err: unknown) {
                setError('Coming soon... Portfolio analytics are being enhanced with more advanced metrics.');
            } finally {
                setIsLoading(false);
            }
        };

        fetchAnalytics();
    }, [benchmark]);

    const formatMetric = (value: number | undefined, format: 'percent' | 'number' = 'number'): string => {
        if (value === undefined || value === null) return 'N/A';
        if (format === 'percent') {
            return `${(value * 100).toFixed(2)}%`;
        }
        return value.toFixed(2);
    };

    const [showReport, setShowReport] = useState(false);
    const [reportHtml, setReportHtml] = useState<string | null>(null);
    const [loadingReport, setLoadingReport] = useState(false);

    const openFullReport = async () => {
        try {
            if (showReport) {
                setShowReport(false);
                setReportHtml(null);
                return;
            }
            
            setLoadingReport(true);
            setError(null);
            
            const response = await fetchWithAuth(
                `${API_BASE_URL}/api/portfolio/analysis/report?benchmark=${benchmark}`
            );
            
            if (!response.ok) {
                throw new Error('Failed to load report');
            }
            
            const htmlContent = await response.text();
            setReportHtml(htmlContent);
            setShowReport(true);
        } catch (err: unknown) {
            setError('Failed to load report. Please ensure you have holdings in your portfolio.');
        } finally {
            setLoadingReport(false);
        }
    };

    return (
        <WidgetCard
            title="PORTFOLIO ANALYTICS"
            tooltip="Métricas avanzadas de tu cartera: Sharpe Ratio, Alpha, Beta, volatilidad, y comparación con benchmarks como S&P 500."
        >
            <div className="space-y-4">
                {/* Benchmark Selector */}
                <div className="flex items-center gap-2">
                    <label className="text-sm text-slate-400">Benchmark:</label>
                    <select
                        value={benchmark}
                        onChange={(e) => setBenchmark(e.target.value)}
                        className="bg-gray-800 border border-slate-700 rounded-md px-3 py-1 text-sm text-slate-200 focus:outline-none focus:ring-1 focus:ring-slate-600"
                    >
                        <option value="SPY">SPY (S&P 500)</option>
                        <option value="QQQ">QQQ (NASDAQ)</option>
                        <option value="DIA">DIA (Dow Jones)</option>
                    </select>
                </div>

                {isLoading && (
                    <div className="text-center text-slate-500 py-8">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-slate-500 mx-auto"></div>
                        <p className="mt-2 text-sm">Analyzing portfolio...</p>
                    </div>
                )}

                {error && (
                    <div className="bg-red-900/20 border border-red-700/50 rounded-md p-4 text-red-400 text-sm">
                        {error}
                    </div>
                )}

                {analytics && !isLoading && (
                    <>
                        <div className="grid grid-cols-2 gap-4">
                            {/* Key Metrics */}
                            <div className="space-y-3">
                                <div className="flex justify-between items-center">
                                    <span className="text-sm text-slate-400">Sharpe Ratio</span>
                                    <span className="text-sm font-bold text-slate-200">
                                        {formatMetric(analytics.metrics.sharpe_ratio)}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-sm text-slate-400">Sortino Ratio</span>
                                    <span className="text-sm font-bold text-slate-200">
                                        {formatMetric(analytics.metrics.sortino_ratio)}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-sm text-slate-400">Alpha</span>
                                    <span className={`text-sm font-bold ${
                                        (analytics.metrics.alpha || 0) > 0 ? 'text-green-400' : 'text-red-400'
                                    }`}>
                                        {formatMetric(analytics.metrics.alpha, 'percent')}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-sm text-slate-400">Beta</span>
                                    <span className="text-sm font-bold text-slate-200">
                                        {formatMetric(analytics.metrics.beta)}
                                    </span>
                                </div>
                            </div>

                            <div className="space-y-3">
                                <div className="flex justify-between items-center">
                                    <span className="text-sm text-slate-400">CAGR</span>
                                    <span className={`text-sm font-bold ${
                                        (analytics.metrics.cagr || 0) > 0 ? 'text-green-400' : 'text-red-400'
                                    }`}>
                                        {formatMetric(analytics.metrics.cagr, 'percent')}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-sm text-slate-400">Max Drawdown</span>
                                    <span className="text-sm font-bold text-red-400">
                                        {formatMetric(analytics.metrics.max_drawdown, 'percent')}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-sm text-slate-400">Volatility</span>
                                    <span className="text-sm font-bold text-slate-200">
                                        {formatMetric(analytics.metrics.volatility, 'percent')}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-sm text-slate-400">Holdings</span>
                                    <span className="text-sm font-bold text-slate-200">
                                        {analytics.holdings_count}
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Full Report Button */}
                        <button
                            onClick={openFullReport}
                            disabled={loadingReport}
                            className="w-full mt-4 bg-slate-800 text-white font-bold py-2 px-4 rounded-md hover:bg-slate-700 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {loadingReport ? 'Loading Report...' : showReport ? 'Hide Full Report' : 'View Full Report'}
                        </button>

                        {/* Report Display */}
                        {showReport && reportHtml && (
                            <div className="mt-4 rounded-lg overflow-hidden border border-slate-700" style={{ minHeight: '400px', maxHeight: '800px', overflowY: 'auto' }}>
                                <div 
                                    className="p-4"
                                    dangerouslySetInnerHTML={{ __html: reportHtml }}
                                    style={{ backgroundColor: '#ffffff', color: '#000000' }}
                                />
                            </div>
                        )}

                        {analytics.analysis_date && (
                            <p className="text-xs text-slate-500 text-center mt-2">
                                Last updated: {new Date(analytics.analysis_date).toLocaleDateString()}
                            </p>
                        )}
                    </>
                )}
            </div>
        </WidgetCard>
    );
};







