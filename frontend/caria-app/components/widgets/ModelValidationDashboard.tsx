/**
 * ModelValidationDashboard - Admin dashboard showing aggregate statistics and retraining trigger.
 */

import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface RetrainingTrigger {
    id: string;
    triggered_at: string;
    trigger_reason: string;
    portfolios_analyzed: number;
    average_underperformance_pct: number | null;
    threshold_met: boolean;
    retraining_status: string;
    retraining_completed_at: string | null;
    notes: string | null;
}

interface PerformanceAnalysis {
    should_retrain: boolean;
    reason: string;
    portfolios_analyzed: number;
    average_underperformance_pct: number | null;
    portfolio_details?: Array<{
        portfolio_id: string;
        selection_type: string;
        portfolio_return: number;
        benchmark_return: number;
        underperformance: number;
        alpha: number | null;
        date: string;
    }>;
    threshold: number;
}

interface AnalysisResponse {
    analysis: PerformanceAnalysis;
    retraining_history: RetrainingTrigger[];
}

export const ModelValidationDashboard: React.FC = () => {
    const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [daysBack, setDaysBack] = useState(90);

    useEffect(() => {
        loadAnalysis();
    }, [daysBack]);

    const loadAnalysis = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await fetchWithAuth(
                `${API_BASE_URL}/api/portfolio/model/analyze?days_back=${daysBack}`
            );

            if (!response.ok) {
                throw new Error('Failed to load analysis');
            }

            const data: AnalysisResponse = await response.json();
            setAnalysis(data);
        } catch (err: any) {
            console.error('Error loading analysis:', err);
            setError(err.message || 'Error loading analysis');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <WidgetCard title="Model Validation Dashboard" id="model-validation-dashboard">
                <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    Loading analysis...
                </div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard title="Model Validation Dashboard" id="model-validation-dashboard">
                <div className="text-sm" style={{ color: '#ef4444' }}>{error}</div>
                <button
                    onClick={loadAnalysis}
                    className="mt-2 text-xs underline"
                    style={{ color: 'var(--color-primary)' }}
                >
                    Retry
                </button>
            </WidgetCard>
        );
    }

    if (!analysis) {
        return null;
    }

    const { analysis: perfAnalysis, retraining_history } = analysis;

    return (
        <WidgetCard title="Model Validation Dashboard" id="model-validation-dashboard">
            <div className="space-y-4">
                {/* Analysis Period */}
                <div className="flex items-center gap-3">
                    <label className="text-sm" style={{ color: 'var(--color-text-primary)' }}>
                        Analysis Period:
                    </label>
                    <select
                        value={daysBack}
                        onChange={(e) => setDaysBack(parseInt(e.target.value))}
                        className="px-3 py-1 rounded-lg text-sm"
                        style={{
                            backgroundColor: 'var(--color-bg-secondary)',
                            border: '1px solid var(--color-bg-tertiary)',
                            color: 'var(--color-text-primary)',
                        }}
                    >
                        <option value={30}>30 days</option>
                        <option value={60}>60 days</option>
                        <option value={90}>90 days</option>
                        <option value={180}>180 days</option>
                        <option value={365}>1 year</option>
                    </select>
                </div>

                {/* Retraining Status */}
                <div
                    className={`p-4 rounded-lg ${
                        perfAnalysis.should_retrain
                            ? 'bg-yellow-900/20 border-yellow-500/30'
                            : 'bg-green-900/20 border-green-500/30'
                    }`}
                    style={{ border: '1px solid' }}
                >
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="text-lg font-semibold" style={{ color: 'var(--color-cream)' }}>
                            Retraining Status
                        </h3>
                        <span
                            className={`px-3 py-1 rounded text-xs font-semibold ${
                                perfAnalysis.should_retrain ? 'bg-yellow-500/20 text-yellow-300' : 'bg-green-500/20 text-green-300'
                            }`}
                        >
                            {perfAnalysis.should_retrain ? 'RETRAINING RECOMMENDED' : 'NO RETRAINING NEEDED'}
                        </span>
                    </div>
                    <p className="text-sm mb-3" style={{ color: 'var(--color-text-secondary)' }}>
                        {perfAnalysis.reason}
                    </p>
                    {perfAnalysis.average_underperformance_pct !== null && (
                        <div className="grid grid-cols-2 gap-3 text-sm">
                            <div>
                                <div style={{ color: 'var(--color-text-secondary)' }}>Portfolios Analyzed</div>
                                <div className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                                    {perfAnalysis.portfolios_analyzed}
                                </div>
                            </div>
                            <div>
                                <div style={{ color: 'var(--color-text-secondary)' }}>Avg Underperformance</div>
                                <div
                                    className="font-semibold"
                                    style={{
                                        color:
                                            perfAnalysis.average_underperformance_pct >= 0 ? '#10b981' : '#ef4444',
                                    }}
                                >
                                    {perfAnalysis.average_underperformance_pct >= 0 ? '+' : ''}
                                    {perfAnalysis.average_underperformance_pct.toFixed(2)}%
                                </div>
                            </div>
                            <div>
                                <div style={{ color: 'var(--color-text-secondary)' }}>Threshold</div>
                                <div className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                                    {perfAnalysis.threshold}%
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Portfolio Details */}
                {perfAnalysis.portfolio_details && perfAnalysis.portfolio_details.length > 0 && (
                    <div className="p-3 rounded-lg max-h-64 overflow-y-auto" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
                        <h4 className="text-sm font-semibold mb-2" style={{ color: 'var(--color-cream)' }}>
                            Portfolio Performance Details
                        </h4>
                        <div className="space-y-2">
                            {perfAnalysis.portfolio_details.map((portfolio) => (
                                <div
                                    key={portfolio.portfolio_id}
                                    className="p-2 rounded text-xs"
                                    style={{ backgroundColor: 'var(--color-bg-primary)' }}
                                >
                                    <div className="flex justify-between items-center mb-1">
                                        <span className="font-mono text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                            {portfolio.portfolio_id.substring(0, 8)}...
                                        </span>
                                        <span className="px-2 py-0.5 rounded text-xs" style={{ backgroundColor: 'var(--color-bg-tertiary)', color: 'var(--color-text-secondary)' }}>
                                            {portfolio.selection_type}
                                        </span>
                                    </div>
                                    <div className="grid grid-cols-3 gap-2">
                                        <div>
                                            <div style={{ color: 'var(--color-text-secondary)' }}>Return</div>
                                            <div
                                                style={{
                                                    color: portfolio.portfolio_return >= 0 ? '#10b981' : '#ef4444',
                                                }}
                                            >
                                                {portfolio.portfolio_return >= 0 ? '+' : ''}
                                                {portfolio.portfolio_return.toFixed(2)}%
                                            </div>
                                        </div>
                                        <div>
                                            <div style={{ color: 'var(--color-text-secondary)' }}>Benchmark</div>
                                            <div
                                                style={{
                                                    color: portfolio.benchmark_return >= 0 ? '#10b981' : '#ef4444',
                                                }}
                                            >
                                                {portfolio.benchmark_return >= 0 ? '+' : ''}
                                                {portfolio.benchmark_return.toFixed(2)}%
                                            </div>
                                        </div>
                                        <div>
                                            <div style={{ color: 'var(--color-text-secondary)' }}>Underperf.</div>
                                            <div
                                                style={{
                                                    color: portfolio.underperformance >= 0 ? '#10b981' : '#ef4444',
                                                }}
                                            >
                                                {portfolio.underperformance >= 0 ? '+' : ''}
                                                {portfolio.underperformance.toFixed(2)}%
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Retraining History */}
                <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
                    <h4 className="text-sm font-semibold mb-2" style={{ color: 'var(--color-cream)' }}>
                        Retraining History
                    </h4>
                    {retraining_history.length === 0 ? (
                        <div className="text-xs text-center py-2" style={{ color: 'var(--color-text-secondary)' }}>
                            No retraining triggers yet.
                        </div>
                    ) : (
                        <div className="space-y-2 max-h-48 overflow-y-auto">
                            {retraining_history.map((trigger) => (
                                <div
                                    key={trigger.id}
                                    className="p-2 rounded text-xs"
                                    style={{ backgroundColor: 'var(--color-bg-primary)' }}
                                >
                                    <div className="flex justify-between items-start mb-1">
                                        <span className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                            {new Date(trigger.triggered_at).toLocaleDateString()}
                                        </span>
                                        <span
                                            className={`px-2 py-0.5 rounded text-xs ${
                                                trigger.retraining_status === 'completed'
                                                    ? 'bg-green-500/20 text-green-300'
                                                    : trigger.retraining_status === 'in_progress'
                                                    ? 'bg-yellow-500/20 text-yellow-300'
                                                    : 'bg-gray-500/20 text-gray-300'
                                            }`}
                                        >
                                            {trigger.retraining_status}
                                        </span>
                                    </div>
                                    <div className="text-xs mb-1" style={{ color: 'var(--color-text-primary)' }}>
                                        {trigger.trigger_reason}
                                    </div>
                                    {trigger.average_underperformance_pct !== null && (
                                        <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                            {trigger.portfolios_analyzed} portfolios, avg underperformance:{' '}
                                            {trigger.average_underperformance_pct.toFixed(2)}%
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </WidgetCard>
    );
};

