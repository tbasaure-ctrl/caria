import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { fetchWithAuth, API_BASE_URL } from '../services/apiService';

interface Scenario {
    price: number;
    return_pct: number;
}

interface RiskRewardData {
    ticker: string;
    horizon_months: number;
    current_price: number;
    scenarios: {
        bear: Scenario;
        base: Scenario;
        bull: Scenario;
    };
    metrics: {
        upside: number;
        downside: number;
        rrr: number;
        expected_value: number;
    };
    explanations: {
        summary: string;
        ev_breakdown: string;
        analogy: string;
        position_sizing: string;
    };
    volatility_metrics: {
        volatility: number;
        max_drawdown: number;
    };
}

interface RiskRewardPanelProps {
    ticker?: string;
    onTickerChange?: (ticker: string) => void;
}

export const RiskRewardPanel: React.FC<RiskRewardPanelProps> = ({ ticker: initialTicker, onTickerChange }) => {
    const [ticker, setTicker] = useState(initialTicker || '');
    const [horizonMonths, setHorizonMonths] = useState(24);
    const [probabilities, setProbabilities] = useState({
        bear: 0.20,
        base: 0.50,
        bull: 0.30,
    });
    const [data, setData] = useState<RiskRewardData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (ticker && ticker.length >= 1) {
            const timeoutId = setTimeout(() => {
                analyzeRiskReward();
            }, 500); // Debounce
            return () => clearTimeout(timeoutId);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [ticker, horizonMonths, probabilities]);

    const analyzeRiskReward = async () => {
        if (!ticker || ticker.length < 1) return;

        setLoading(true);
        setError(null);

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/risk-reward/analyze`, {
                method: 'POST',
                body: JSON.stringify({
                    ticker: ticker.toUpperCase(),
                    horizon_months: horizonMonths,
                    probabilities: probabilities,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const result = await response.json();
            setData(result);
        } catch (err: any) {
            console.error('Risk-reward analysis error:', err);
            setError(err?.message || 'Failed to analyze risk-reward');
            setData(null);
        } finally {
            setLoading(false);
        }
    };

    const handleProbabilityChange = (scenario: 'bear' | 'base' | 'bull', value: number) => {
        const newProbs = { ...probabilities };
        newProbs[scenario] = Math.max(0, Math.min(1, value / 100));

        // Normalize to sum to 100%
        const total = newProbs.bear + newProbs.base + newProbs.bull;
        if (total > 0) {
            newProbs.bear = newProbs.bear / total;
            newProbs.base = newProbs.base / total;
            newProbs.bull = newProbs.bull / total;
        }

        setProbabilities(newProbs);
    };

    const handleTickerChange = (value: string) => {
        setTicker(value);
        if (onTickerChange) {
            onTickerChange(value);
        }
    };

    // Prepare chart data
    const chartData = data ? [
        {
            name: 'Bear',
            return_pct: data.scenarios.bear.return_pct * 100,
            price: data.scenarios.bear.price,
            probability: probabilities.bear * 100,
        },
        {
            name: 'Base',
            return_pct: data.scenarios.base.return_pct * 100,
            price: data.scenarios.base.price,
            probability: probabilities.base * 100,
        },
        {
            name: 'Bull',
            return_pct: data.scenarios.bull.return_pct * 100,
            price: data.scenarios.bull.price,
            probability: probabilities.bull * 100,
        },
    ] : [];

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div
                    style={{
                        backgroundColor: 'var(--color-bg-secondary)',
                        border: '1px solid var(--color-border-default)',
                        borderRadius: '8px',
                        padding: '12px',
                    }}
                >
                    <p style={{ margin: '0 0 4px 0', fontWeight: 'bold', color: 'var(--color-text-primary)' }}>
                        {data.name} Case
                    </p>
                    <p style={{ margin: '4px 0', color: 'var(--color-text-secondary)' }}>
                        Return: {data.return_pct > 0 ? '+' : ''}{data.return_pct.toFixed(1)}%
                    </p>
                    <p style={{ margin: '4px 0', color: 'var(--color-text-secondary)' }}>
                        Price: ${data.price.toFixed(2)}
                    </p>
                    <p style={{ margin: '4px 0', color: 'var(--color-text-secondary)' }}>
                        Probability: {data.probability.toFixed(0)}%
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <div
            className="h-full flex flex-col overflow-hidden"
            style={{
                backgroundColor: 'var(--color-bg-secondary)',
                borderLeft: '1px solid var(--color-border-subtle)',
            }}
        >
            {/* Header */}
            <div
                className="px-6 py-4 border-b"
                style={{
                    borderColor: 'var(--color-border-subtle)',
                }}
            >
                <h2
                    className="text-lg font-semibold mb-1"
                    style={{
                        fontFamily: 'var(--font-display)',
                        color: 'var(--color-text-primary)',
                    }}
                >
                    Risk–Reward Engine
                </h2>
                <p
                    className="text-xs"
                    style={{ color: 'var(--color-text-muted)' }}
                >
                    Quantify upside, downside, and expected value
                </p>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {/* Ticker Input */}
                <div>
                    <label
                        className="block text-sm font-medium mb-2"
                        style={{ color: 'var(--color-text-primary)' }}
                    >
                        Ticker
                    </label>
                    <input
                        type="text"
                        value={ticker}
                        onChange={(e) => handleTickerChange(e.target.value.toUpperCase())}
                        placeholder="Enter ticker (e.g., AAPL)"
                        className="w-full px-4 py-2 rounded-lg text-sm focus:outline-none focus:ring-2"
                        style={{
                            backgroundColor: 'var(--color-bg-tertiary)',
                            border: '1px solid var(--color-border-subtle)',
                            color: 'var(--color-text-primary)',
                        }}
                    />
                </div>

                {/* Time Horizon */}
                <div>
                    <label
                        className="block text-sm font-medium mb-2"
                        style={{ color: 'var(--color-text-primary)' }}
                    >
                        Time Horizon
                    </label>
                    <div className="flex gap-2">
                        {[12, 24, 36].map((months) => (
                            <button
                                key={months}
                                onClick={() => setHorizonMonths(months)}
                                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                    horizonMonths === months
                                        ? 'text-white'
                                        : 'text-text-secondary'
                                }`}
                                style={{
                                    backgroundColor:
                                        horizonMonths === months
                                            ? 'var(--color-accent-primary)'
                                            : 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border-subtle)',
                                }}
                            >
                                {months}M
                            </button>
                        ))}
                    </div>
                </div>

                {/* Probability Sliders */}
                {data && (
                    <div>
                        <label
                            className="block text-sm font-medium mb-3"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            Scenario Probabilities
                        </label>
                        <div className="space-y-4">
                            {(['bear', 'base', 'bull'] as const).map((scenario) => (
                                <div key={scenario}>
                                    <div className="flex justify-between items-center mb-2">
                                        <span
                                            className="text-sm capitalize font-medium"
                                            style={{ color: 'var(--color-text-primary)' }}
                                        >
                                            {scenario} Case
                                        </span>
                                        <span
                                            className="text-sm font-semibold"
                                            style={{ color: 'var(--color-text-secondary)' }}
                                        >
                                            {(probabilities[scenario] * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <input
                                        type="range"
                                        min="0"
                                        max="100"
                                        value={probabilities[scenario] * 100}
                                        onChange={(e) =>
                                            handleProbabilityChange(scenario, parseFloat(e.target.value))
                                        }
                                        className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                                        style={{
                                            backgroundColor: 'var(--color-bg-tertiary)',
                                            accentColor: 'var(--color-accent-primary)',
                                        }}
                                    />
                                </div>
                            ))}
                            <p
                                className="text-xs mt-2"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Total: {(probabilities.bear + probabilities.base + probabilities.bull) * 100}%
                            </p>
                        </div>
                    </div>
                )}

                {/* Loading State */}
                {loading && (
                    <div className="text-center py-8">
                        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-accent-primary"></div>
                        <p
                            className="text-sm mt-2"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Analyzing risk-reward...
                        </p>
                    </div>
                )}

                {/* Error State */}
                {error && (
                    <div
                        className="p-4 rounded-lg"
                        style={{
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            border: '1px solid rgba(239, 68, 68, 0.3)',
                        }}
                    >
                        <p
                            className="text-sm"
                            style={{ color: '#ef4444' }}
                        >
                            {error}
                        </p>
                    </div>
                )}

                {/* Results */}
                {data && !loading && (
                    <>
                        {/* Scenario Chart */}
                        <div>
                            <h3
                                className="text-sm font-semibold mb-3"
                                style={{ color: 'var(--color-text-primary)' }}
                            >
                                Scenario Outcomes
                            </h3>
                            <div className="w-full" style={{ minHeight: '256px', height: '256px' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                        <XAxis
                                            dataKey="name"
                                            stroke="var(--color-text-muted)"
                                            tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }}
                                        />
                                        <YAxis
                                            stroke="var(--color-text-muted)"
                                            tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }}
                                            tickFormatter={(value) => `${value > 0 ? '+' : ''}${value.toFixed(0)}%`}
                                        />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Bar dataKey="return_pct" radius={[8, 8, 0, 0]}>
                                            {chartData.map((entry, index) => (
                                                <Cell
                                                    key={`cell-${index}`}
                                                    fill={
                                                        entry.name === 'Bear'
                                                            ? '#ef4444'
                                                            : entry.name === 'Base'
                                                            ? '#fbbf24'
                                                            : '#22c55e'
                                                    }
                                                />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Metrics */}
                        <div
                            className="grid grid-cols-2 gap-4"
                        >
                            <div
                                className="p-4 rounded-lg"
                                style={{
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border-subtle)',
                                }}
                            >
                                <p
                                    className="text-xs mb-1"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Upside
                                </p>
                                <p
                                    className="text-xl font-semibold"
                                    style={{ color: '#22c55e' }}
                                >
                                    +{(data.metrics.upside * 100).toFixed(0)}%
                                </p>
                            </div>
                            <div
                                className="p-4 rounded-lg"
                                style={{
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border-subtle)',
                                }}
                            >
                                <p
                                    className="text-xs mb-1"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Downside
                                </p>
                                <p
                                    className="text-xl font-semibold"
                                    style={{ color: '#ef4444' }}
                                >
                                    -{(data.metrics.downside * 100).toFixed(0)}%
                                </p>
                            </div>
                            <div
                                className="p-4 rounded-lg"
                                style={{
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border-subtle)',
                                }}
                            >
                                <p
                                    className="text-xs mb-1"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Risk-Reward Ratio
                                </p>
                                <p
                                    className="text-xl font-semibold"
                                    style={{ color: 'var(--color-text-primary)' }}
                                >
                                    {data.metrics.rrr.toFixed(1)}:1
                                </p>
                            </div>
                            <div
                                className="p-4 rounded-lg"
                                style={{
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border-subtle)',
                                }}
                            >
                                <p
                                    className="text-xs mb-1"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Expected Value
                                </p>
                                <p
                                    className={`text-xl font-semibold ${
                                        data.metrics.expected_value >= 0 ? 'text-green-500' : 'text-red-500'
                                    }`}
                                >
                                    {data.metrics.expected_value >= 0 ? '+' : ''}
                                    {(data.metrics.expected_value * 100).toFixed(0)}%
                                </p>
                            </div>
                        </div>

                        {/* Explanations */}
                        <div className="space-y-4">
                            <div>
                                <h3
                                    className="text-sm font-semibold mb-2"
                                    style={{ color: 'var(--color-text-primary)' }}
                                >
                                    Summary
                                </h3>
                                <p
                                    className="text-sm leading-relaxed"
                                    style={{ color: 'var(--color-text-secondary)' }}
                                >
                                    {data.explanations.summary}
                                </p>
                            </div>

                            <div>
                                <h3
                                    className="text-sm font-semibold mb-2"
                                    style={{ color: 'var(--color-text-primary)' }}
                                >
                                    Expected Value Breakdown
                                </h3>
                                <p
                                    className="text-sm leading-relaxed font-mono"
                                    style={{ color: 'var(--color-text-secondary)' }}
                                >
                                    {data.explanations.ev_breakdown}
                                </p>
                            </div>

                            <div>
                                <h3
                                    className="text-sm font-semibold mb-2"
                                    style={{ color: 'var(--color-text-primary)' }}
                                >
                                    Analogy
                                </h3>
                                <p
                                    className="text-sm leading-relaxed"
                                    style={{ color: 'var(--color-text-secondary)' }}
                                >
                                    {data.explanations.analogy}
                                </p>
                            </div>

                            <div>
                                <h3
                                    className="text-sm font-semibold mb-2"
                                    style={{ color: 'var(--color-text-primary)' }}
                                >
                                    Position Sizing Guidance
                                </h3>
                                <p
                                    className="text-sm leading-relaxed"
                                    style={{ color: 'var(--color-text-secondary)' }}
                                >
                                    {data.explanations.position_sizing}
                                </p>
                            </div>
                        </div>
                    </>
                )}

                {/* Empty State */}
                {!data && !loading && !error && (
                    <div className="text-center py-12">
                        <div 
                            className="w-16 h-16 mx-auto mb-4 rounded-xl flex items-center justify-center"
                            style={{ backgroundColor: 'rgba(46, 124, 246, 0.12)' }}
                        >
                            <svg className="w-8 h-8" style={{ color: 'var(--color-accent-primary)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                        </div>
                        <h3
                            className="text-base font-semibold mb-2"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            Risk-Reward Analysis
                        </h3>
                        <p
                            className="text-sm mb-4"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            Enter a ticker symbol above to analyze Bear/Base/Bull scenarios
                        </p>
                        <p
                            className="text-xs"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Or mention a ticker in the chat to auto-populate
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

